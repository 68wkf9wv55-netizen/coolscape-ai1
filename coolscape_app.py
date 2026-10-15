"""
CoolScape AI – v3 prototip (Siemens Türkiye Climathon)
Optimizasyon tabanlı karar destek: kısıtlı soğutma suyunu saat saat ısı stresine göre dağıtır.

Ne gerçek, ne temsili:
- Hava verisi GERÇEK: Open-Meteo Historical Weather API (ERA5 tabanlı), Antalya, saatlik.
- Isı yüzeyi GERÇEK (varsa): Landsat 8/9 yüzey sıcaklığı + Sentinel-2 NDVI, H3 altıgenler
  (veri_hazirla.py ile üretilen isi_yuzeyi_h3.csv). Yoksa harita yalnız park noktalarını gösterir.
- Park değerleri TEMSİLİ: nüfus, kırılgan grup, hat kapasitesi. Yüzey sıcaklığı farkı ve yeşil alan,
  ısı yüzeyi dosyası varsa ondan, yoksa temsili değerlerden alınır.
- Soğutma eğrisi VEKİL MODEL v0: literatür ölçeğine (Paris kaldırım ıslatma, yaklaşık 1 °C UTCI)
  dayanan, azalan getirili, kalibre edilmemiş eğri. Belirsizlik aralığıyla raporlanır.

Çalıştırma:  pip install -r requirements.txt  &&  streamlit run coolscape_app.py
"""

import os
import time

import altair as alt
import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import pulp
import requests
import streamlit as st
from pythermalcomfort.models import utci

# ---------------------------------------------------------------------------
# SABİTLER VE VARSAYIMLAR
# ---------------------------------------------------------------------------
ANTALYA_LAT, ANTALYA_LON = 36.89, 30.71
SAATLER = list(range(8, 21))  # 08:00–20:00 karar ufku
DEMO_GUNLERI = {
    "25 Temmuz 2025 (resmî ölçüm 46,1 °C, son 95 yılın temmuz rekoru)": "2025-07-25",
    "10 Ağustos 2025 (MGM: Kepez 45,2 °C, Muratpaşa 43,4 °C)": "2025-08-10",
}
ISI_YUZEYI_CSV = "isi_yuzeyi_h3.csv"
UTCI_ESIK = 32.0  # "güçlü ısı stresi" alt sınırı

# UTCI ısı stresi sınıfları (resmî eşikler) ve renkleri
UTCI_SINIFLARI = [
    (-99, 26, "Isı stresi yok", [189, 215, 231]),
    (26, 32, "Orta", [254, 224, 139]),
    (32, 38, "Güçlü", [253, 174, 97]),
    (38, 46, "Çok güçlü", [215, 48, 39]),
    (46, 99, "Aşırı", [103, 0, 13]),
]

# Vekil model v0 (kalibre edilmemiş)
E_REF = 1.0        # °C: tam etkili sulamada yaya seviyesi UTCI düşüşü ölçeği
VPD_DOYMA = 2.5    # kPa
X0_NOMINAL = 8.0   # m³: azalan getiri ölçeği
X0_ARALIK = (4.0, 16.0)  # duyarlılık analizi
KIRILIM = [0, 4, 8, 16, 32]
MIN_DOZ = 2.0

# ---------------------------------------------------------------------------
# PARK VERİSİ (TEMSİLİ – koordinatları ve değerleri doğrulayın)
# ---------------------------------------------------------------------------
PARKLAR = pd.DataFrame({
    "park": ["Karaalioğlu Parkı", "Atatürk Parkı", "Düden Park", "Dokuma Park", "Konyaaltı Sahil Parkı"],
    "ilce": ["Muratpaşa", "Muratpaşa", "Muratpaşa", "Kepez", "Konyaaltı"],
    "lat": [36.8775, 36.8860, 36.8504, 36.9177, 36.8732],
    "lon": [30.7061, 30.6975, 30.7818, 30.6625, 30.6418],
    "yuzey_farki_c": [0.7, -0.3, -2.3, 2.7, -0.8],
    "yesil_skoru": [7, 6, 8, 4, 5],
    "nufus_skoru": [8, 7, 9, 5, 10],
    "hassas_yuzde": [20, 15, 25, 10, 30],
    "pompa_katsayisi": [1.2, 1.1, 1.5, 1.8, 1.0],
    "hat_kapasitesi_m3_saat": [15, 12, 20, 10, 18],
    "aritilmis_su_erisimi": [False, False, True, True, False],
})


# ---------------------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def hava_verisi_cek(tarih: str) -> pd.DataFrame:
    params = {
        "latitude": ANTALYA_LAT, "longitude": ANTALYA_LON,
        "start_date": tarih, "end_date": tarih,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms", "timezone": "Europe/Istanbul",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "zaman": pd.to_datetime(h["time"]),
        "sicaklik_c": h["temperature_2m"], "bagil_nem": h["relative_humidity_2m"],
        "ruzgar_ms": h["wind_speed_10m"], "gunes_wm2": h["shortwave_radiation"],
    })
    df["saat"] = df["zaman"].dt.hour
    return df


def csv_hava_verisi(dosya) -> pd.DataFrame:
    df = pd.read_csv(dosya)
    eksik = {"saat", "sicaklik_c", "bagil_nem", "ruzgar_ms", "gunes_wm2"} - set(df.columns)
    if eksik:
        raise ValueError(f"CSV'de eksik sütunlar: {', '.join(sorted(eksik))}")
    return df


@st.cache_data(show_spinner=False)
def isi_yuzeyi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol)
    if not {"hex", "lst_c", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lst_c"]).copy()
    df["lst_fark_c"] = df["lst_c"] - df["lst_c"].median()
    return df


def parklari_guncelle(parklar, yuzey):
    """Isı yüzeyi varsa parkların yüzey farkı ve yeşil skorunu çevresindeki altıgenlerden al."""
    p = parklar.copy()
    p[["yuzey_farki_c", "yesil_skoru"]] = p[["yuzey_farki_c", "yesil_skoru"]].astype(float)
    p["kaynak"] = "temsili"
    if yuzey is None:
        return p
    cozunurluk = h3.get_resolution(yuzey["hex"].iloc[0])
    idx = yuzey.set_index("hex")
    for i, r in p.iterrows():
        komsu = [c for c in h3.grid_disk(h3.latlng_to_cell(r["lat"], r["lon"], cozunurluk), 1) if c in idx.index]
        if not komsu:
            continue
        p.at[i, "yuzey_farki_c"] = round(float(idx.loc[komsu, "lst_fark_c"].mean()), 2)
        if "ndvi" in idx.columns and idx.loc[komsu, "ndvi"].notna().any():
            ndvi = float(idx.loc[komsu, "ndvi"].mean())
            p.at[i, "yesil_skoru"] = round(float(np.clip(1 + ndvi / 0.8 * 9, 1, 10)), 1)
        p.at[i, "kaynak"] = "Landsat/Sentinel-2"
    return p


# ---------------------------------------------------------------------------
# FİZİK
# ---------------------------------------------------------------------------
def vpd_kpa(sicaklik_c, bagil_nem):
    t = np.asarray(sicaklik_c, dtype=float)
    rh = np.asarray(bagil_nem, dtype=float)
    try:
        from metpy.calc import saturation_vapor_pressure
        from metpy.units import units
        es = saturation_vapor_pressure(t * units.degC).to("kPa").magnitude
    except Exception:
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100.0)


def utci_hesapla(tdb, mrt, ruzgar, nem):
    """MRT yaklaşımı: T + 0,02·SW + yüzey sıcaklığı farkı (kaba). Geçerlilik sınırına kırpılır."""
    tdb = np.asarray(tdb, dtype=float)
    mrt = np.clip(np.asarray(mrt, dtype=float), tdb - 30, tdb + 70)
    v = np.clip(np.asarray(ruzgar, dtype=float), 0.5, 17.0)
    sonuc = utci(tdb=tdb, tr=mrt, v=v, rh=np.asarray(nem, dtype=float))
    return np.asarray(getattr(sonuc, "utci", sonuc), dtype=float)


def park_saat_tablosu(hava, parklar):
    h = hava[hava["saat"].isin(SAATLER)]
    parcalar = []
    for _, p in parklar.iterrows():
        d = h.copy()
        d["park"] = p["park"]
        d["mrt_c"] = d["sicaklik_c"] + 0.02 * d["gunes_wm2"] + p["yuzey_farki_c"]
        parcalar.append(d)
    df = pd.concat(parcalar, ignore_index=True)
    df["utci_c"] = utci_hesapla(df["sicaklik_c"], df["mrt_c"], df["ruzgar_ms"], df["bagil_nem"])
    df["vpd_kpa"] = vpd_kpa(df["sicaklik_c"], df["bagil_nem"])
    df["tehlike"] = (df["utci_c"] - UTCI_ESIK).clip(lower=0).fillna(0)
    return df


def yuzey_utci(yuzey, hava_satiri):
    t, sw = hava_satiri["sicaklik_c"], hava_satiri["gunes_wm2"]
    n = len(yuzey)
    return utci_hesapla(np.full(n, t), t + 0.02 * sw + yuzey["lst_fark_c"].values,
                        np.full(n, hava_satiri["ruzgar_ms"]), np.full(n, hava_satiri["bagil_nem"]))


def utci_renk(deger):
    if not np.isfinite(deger):
        return [200, 200, 200, 120]
    for alt, ust, _, renk in UTCI_SINIFLARI:
        if alt <= deger < ust:
            return renk + [200]
    return UTCI_SINIFLARI[-1][3] + [200]


def utci_sinif(deger):
    for alt, ust, ad, _ in UTCI_SINIFLARI:
        if alt <= deger < ust:
            return ad
    return "Aşırı"


# ---------------------------------------------------------------------------
# OPTİMİZASYON (MILP)
# ---------------------------------------------------------------------------
def sogutma_egrisi(x_m3, emax, x0=X0_NOMINAL):
    return emax * (1 - np.exp(-np.asarray(x_m3, dtype=float) / x0))


def girdileri_hazirla(tablo, parklar, w_maruziyet, w_kirilganlik):
    p = parklar.set_index("park")
    t = tablo.copy()
    t["oncelik"] = t["tehlike"] * (1 + w_maruziyet * t["park"].map(p["nufus_skoru"]) / 10.0
                                   + w_kirilganlik * t["park"].map(p["hassas_yuzde"]) / 30.0)
    t["emax"] = E_REF * np.minimum(t["vpd_kpa"] / VPD_DOYMA, 1.0) * (0.7 + 0.06 * t["park"].map(p["yesil_skoru"]))
    t["kapasite"] = t["park"].map(p["hat_kapasitesi_m3_saat"])
    t["pompa"] = t["park"].map(p["pompa_katsayisi"])
    return t


def degisken(model, ad, alt=None, ust=None, tur="Continuous"):
    if hasattr(model, "add_variable"):  # PuLP 4.x
        return model.add_variable(ad, lowBound=alt, upBound=ust, cat=tur)
    return pulp.LpVariable(ad, lowBound=alt, upBound=ust, cat=tur)


def cozucu_sec():
    for ad in ("PULP_CBC_CMD", "HiGHS", "HiGHS_CMD", "COIN_CMD"):
        sinif = getattr(pulp, ad, None)
        if sinif is None:
            continue
        try:
            c = sinif(msg=False)
            if c.available():
                return c
        except Exception:
            continue
    raise RuntimeError("MILP çözücüsü bulunamadı (pulp==2.9.0 kurulu olmalı).")


def optimize_et(t, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0=X0_NOMINAL):
    model = pulp.LpProblem("CoolScape", pulp.LpMaximize)
    s, amac, toplam, kirilgan = {}, [], [], []
    for idx, r in t.iterrows():
        y = degisken(model, f"y_{idx}", tur="Binary")
        segs = []
        for k in range(len(KIRILIM) - 1):
            a, b = KIRILIM[k], min(KIRILIM[k + 1], r["kapasite"])
            if b <= a:
                break
            v = degisken(model, f"s_{idx}_{k}", alt=0, ust=float(b - a))
            egim = (sogutma_egrisi(b, r["emax"], x0) - sogutma_egrisi(a, r["emax"], x0)) / (b - a)
            amac.append(r["oncelik"] * egim * v)
            if enerji_opt:
                amac.append(-0.01 * r["pompa"] * v)
            segs.append(v)
        x = pulp.lpSum(segs)
        s[idx] = segs
        model += x <= r["kapasite"] * y
        model += x >= MIN_DOZ * y
        if r["vpd_kpa"] < vpd_min:
            model += y == 0
        toplam.append(x)
        if r["park"] in kirilgan_parklar:
            kirilgan.append(x)
    model += pulp.lpSum(amac)
    model += pulp.lpSum(toplam) <= butce
    if kirilgan and kirilgan_pay > 0:
        model += pulp.lpSum(kirilgan) >= kirilgan_pay * pulp.lpSum(toplam)
    model.solve(cozucu_sec())
    durum = pulp.LpStatus[model.status] if hasattr(pulp, "LpStatus") else str(model.status)
    sonuc = t.copy()
    sonuc["su_m3"] = [round(sum((v.value() or 0) for v in s[i]), 2) for i in t.index]
    return sonuc, durum


def esit_dagitim(t, butce):
    """Karşılaştırma: aynı bütçe parklara ve saatlere eşit bölünür; nem ve öncelik gözetilmez."""
    sonuc = t.copy()
    sonuc["su_m3"] = np.minimum(butce / len(t), sonuc["kapasite"])
    return sonuc


def etki(sonuc, x0=X0_NOMINAL, emax_carpan=1.0):
    s = sonuc.copy()
    s["utci_dusus_c"] = sogutma_egrisi(s["su_m3"], s["emax"] * emax_carpan, x0)
    s["agirlikli_fayda"] = s["oncelik"] * s["utci_dusus_c"]
    s["utci_sonra_c"] = s["utci_c"] - s["utci_dusus_c"]
    return s


def operator_komutlari(sonuc, parklar, aritilmis_senaryo):
    p = parklar.set_index("park")
    satirlar = []
    for park, g in sonuc[sonuc["su_m3"] > 0.01].sort_values("saat").groupby("park"):
        miktar = g.set_index("saat")["su_m3"]
        saatler = list(miktar.index)
        if aritilmis_senaryo and p.loc[park, "aritilmis_su_erisimi"]:
            kaynak = "Arıtılmış su (senaryo)"
        else:
            kaynak = "Mevcut kaynak"
        bas = onceki = saatler[0]
        for sa in saatler[1:] + [None]:
            if sa is None or sa != onceki + 1:
                satirlar.append({"Park": park, "Başlangıç": f"{bas:02d}:00", "Bitiş": f"{onceki + 1:02d}:00",
                                 "Su (m³)": round(float(miktar.loc[bas:onceki].sum()), 1), "Kaynak": kaynak})
                bas = sa
            onceki = sa if sa is not None else onceki
    return pd.DataFrame(satirlar).sort_values(["Başlangıç", "Park"]) if satirlar else pd.DataFrame()


# ---------------------------------------------------------------------------
# GÖRSELLEŞTİRME
# ---------------------------------------------------------------------------
SICAK_RENKLER = [[255, 255, 178, 0], [254, 217, 118, 110], [254, 178, 76, 160],
                 [253, 141, 60, 195], [240, 59, 32, 220], [189, 0, 38, 235]]


def gun_katsayisi(satir):
    """Kent geneli ısı stresinin o saatteki şiddeti (0–1): sabah/akşam sıcak noktalar soluklaşır."""
    u = utci_hesapla([satir["sicaklik_c"]], [satir["sicaklik_c"] + 0.02 * satir["gunes_wm2"]],
                     [satir["ruzgar_ms"]], [satir["bagil_nem"]])[0]
    return float(np.clip((u - 26) / 20, 0, 1)) if np.isfinite(u) else 0.0


def harita_ciz(yer, saat, hava, yuzey, parklar, plan, sicak_pay):
    katmanlar = []
    satir = hava[hava["saat"] == saat].iloc[0]
    if yuzey is not None:
        # Yalnız kentin en sıcak %sicak_pay'lık alanı renklenir; ağırlık = eşiğin üstündeki yüzey sıcaklığı farkı
        esik = yuzey["lst_fark_c"].quantile(1 - sicak_pay)
        ust = max(yuzey["lst_fark_c"].quantile(0.99) - esik, 0.1)
        y = yuzey[["lat", "lon", "lst_fark_c"]].copy()
        y["agirlik"] = (y["lst_fark_c"] - esik).clip(lower=0) * gun_katsayisi(satir)
        y = y[y["agirlik"] > 0]
        if not y.empty:
            katmanlar.append(pdk.Layer(
                "HeatmapLayer", y, id="isi", get_position="[lon, lat]", get_weight="agirlik",
                aggregation=pdk.types.String("MEAN"), radius_pixels=35, intensity=1, threshold=0.03,
                color_range=SICAK_RENKLER, color_domain=[0, ust]))
    p = parklar[["park", "lat", "lon"]].copy()
    if plan is not None:
        h = plan[plan["saat"] == saat].set_index("park")
        p["su"] = p["park"].map(h["su_m3"]).fillna(0)
        p["once"] = p["park"].map(h["utci_c"])
        p["sonra"] = p["park"].map(h["utci_sonra_c"])
        p["yaricap"] = np.sqrt(p["su"]) * 90 + 60
        p["renk"] = [[0, 120, 255, 220] if s > 0 else [120, 120, 120, 160] for s in p["su"]]
        p["bilgi"] = [f"{s:.1f} m³ · UTCI {a:.1f} → {b:.1f} °C" for s, a, b in zip(p["su"], p["once"], p["sonra"])]
    else:
        p["yaricap"] = 120
        p["renk"] = [[30, 30, 30, 200]] * len(p)
        p["bilgi"] = "Optimizasyon henüz çalışmadı"
    p["ad"] = p["park"]
    katmanlar.append(pdk.Layer("ScatterplotLayer", p, id="parklar", get_position="[lon, lat]",
                               get_radius="yaricap", get_fill_color="renk", pickable=True,
                               stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2))
    yer.pydeck_chart(pdk.Deck(layers=katmanlar,
                              initial_view_state=pdk.ViewState(latitude=36.89, longitude=30.71, zoom=11.3, pitch=0),
                              tooltip={"text": "{ad}\n{bilgi}"}))


def su_an_karti(yer, saat, hava, plan, parklar):
    satir = hava[hava["saat"] == saat].iloc[0]
    with yer.container(border=True):
        st.markdown(f"### 🕐 {saat:02d}:00")
        st.caption(f"Hava {satir['sicaklik_c']:.1f} °C · nem %{satir['bagil_nem']:.0f} · "
                   f"rüzgâr {satir['ruzgar_ms']:.1f} m/s · güneş {satir['gunes_wm2']:.0f} W/m²")
        if plan is None:
            st.write("Optimizasyon çalışınca bu saatin su planı burada görünecek.")
            return
        h = plan[plan["saat"] == saat].sort_values("su_m3", ascending=False)
        sulanan = h[h["su_m3"] > 0.01]
        if sulanan.empty:
            if (h["vpd_kpa"] < st.session_state.get("vpd_min", 0)).all():
                st.write("Bu saatte sulama yok: hava nemli, su az serinletir (nem kapısı kapalı).")
            else:
                st.write("Bu saatte sulama yok: su daha verimli olacağı saatlere ayrıldı.")
        for _, r in sulanan.iterrows():
            st.markdown(f"**{r['park']}** · {r['su_m3']:.1f} m³ · UTCI {r['utci_c']:.1f} → {r['utci_sonra_c']:.1f} °C  \n"
                        f"<span style='color:gray'>Neden: {utci_sinif(r['utci_c']).lower()} ısı stresi, "
                        f"VPD {r['vpd_kpa']:.2f} kPa, öncelik {r['oncelik']:.1f}</span>", unsafe_allow_html=True)


def plan_matrisi(plan):
    d = plan[["park", "saat", "su_m3", "utci_dusus_c"]].copy()
    d["saat_et"] = d["saat"].map(lambda s: f"{s:02d}:00")
    d["etiket"] = d["su_m3"].map(lambda v: f"{v:.0f}" if v >= 0.5 else "")
    taban = alt.Chart(d).encode(
        x=alt.X("saat_et:O", title="Saat"),
        y=alt.Y("park:N", title=None),
    )
    hucre = taban.mark_rect().encode(
        color=alt.Color("su_m3:Q", title="Su (m³)", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("park:N", title="Park"), alt.Tooltip("saat_et:O", title="Saat"),
                 alt.Tooltip("su_m3:Q", title="Su (m³)", format=".1f"),
                 alt.Tooltip("utci_dusus_c:Q", title="UTCI düşüşü (°C)", format=".2f")])
    yazi = taban.mark_text(fontSize=11).encode(
        text="etiket:N",
        color=alt.condition("datum.su_m3 > 8", alt.value("white"), alt.value("black")))
    return (hucre + yazi).properties(height=40 * d["park"].nunique() + 30)


def lejant(yuzey, sicak_pay):
    renkler = ", ".join(f"rgba({r},{g},{b},{max(a, 60) / 255:.2f})" for r, g, b, a in SICAK_RENKLER[1:])
    bar = (f"<span style='display:inline-block;width:160px;height:12px;vertical-align:middle;border-radius:3px;"
           f"background:linear-gradient(to right, {renkler})'></span>")
    if yuzey is not None:
        esik = yuzey["lst_fark_c"].quantile(1 - sicak_pay)
        metin = (f"**Sıcak noktalar:** kentin en sıcak %{sicak_pay * 100:.0f}'lik yüzeyi "
                 f"(kent medyanından en az {esik:+.1f} °C sıcak) &nbsp; daha az {bar} daha çok · "
                 f"renk yoğunluğu o saatteki ısı stresiyle değişir · "
                 f"<span style='color:#0078ff'>●</span> sulanan park · <span style='color:#777'>●</span> sulanmayan park")
    else:
        metin = ("**Isı yüzeyi yok:** yalnızca park noktaları gösteriliyor. "
                 f"Sıcak noktalar için önce veri_hazirla.py çalıştırın ({ISI_YUZEYI_CSV}).")
    st.markdown(metin, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CoolScape AI", layout="wide", initial_sidebar_state="expanded")
st.title("🌍 CoolScape AI")
st.markdown("**Kentsel serinlik ve su tahsis motoru** · optimizasyon tabanlı karar destek")

for anahtar, deger in {"plan": None, "esit": None, "ozet": None, "saat": 14}.items():
    st.session_state.setdefault(anahtar, deger)


def sifirla():
    st.session_state.plan = st.session_state.esit = st.session_state.ozet = None


st.sidebar.header("⚙️ Belediyenin girdileri")
gun = st.sidebar.selectbox("Yeniden oynatılacak gün", list(DEMO_GUNLERI), on_change=sifirla)
butce = st.sidebar.number_input("Günlük su bütçesi (m³, 08:00–20:00)", value=300, step=25, min_value=25, on_change=sifirla)
kirilgan_pay = st.sidebar.slider("Kırılgan parklara asgari su payı (%)", 0, 80, 40, on_change=sifirla) / 100
st.sidebar.header("Kurallar")
vpd_min = st.sidebar.slider("Nem kapısı: en düşük VPD (kPa)", 0.0, 2.0, 0.8, 0.1, on_change=sifirla,
                            help="Buhar basıncı açığı bunun altındaysa hava nemlidir, su az serinletir; o saatte sulama yapılmaz.")
st.session_state.vpd_min = vpd_min
enerji_opt = st.sidebar.checkbox("Pompalama enerjisini gözet", value=True, on_change=sifirla)
aritilmis_senaryo = st.sidebar.checkbox("Arıtılmış su senaryosu (gelecek)", value=False,
                                        help="Bugün arıtılmış su büyük ölçüde denize deşarj ediliyor. Bu anahtar, altyapı kurulursa "
                                             "hangi parkların arıtılmış suyla beslenebileceğini gösterir.")
with st.sidebar.expander("Öncelik ağırlıkları"):
    w_maruziyet = st.slider("Maruziyet (nüfus)", 0.0, 2.0, 0.5, 0.1, on_change=sifirla)
    w_kirilganlik = st.slider("Kırılganlık (65+ ve 0–4 yaş)", 0.0, 2.0, 0.5, 0.1, on_change=sifirla)
sicak_pay = st.sidebar.slider("Haritada gösterilecek en sıcak alan (%)", 5, 50, 20, 5,
                              help="Yalnızca kentin bu yüzdelik en sıcak yüzeyi renklendirilir; geri kalanı şeffaf kalır.") / 100

# --- Veri yükleme ---
try:
    hava = hava_verisi_cek(DEMO_GUNLERI[gun])
    hava_kaynak = "Open-Meteo (ERA5 tabanlı), gerçek"
except Exception as hata:
    st.error(f"Open-Meteo'ya erişilemedi ({hata}). Uydurma veriyle devam edilmez. Saatlik veriyi CSV olarak "
             "yükleyebilirsiniz (sütunlar: saat, sicaklik_c, bagil_nem, ruzgar_ms, gunes_wm2).")
    yuklenen = st.file_uploader("Saatlik hava verisi (CSV)", type="csv")
    if yuklenen is None:
        st.stop()
    hava = csv_hava_verisi(yuklenen)
    hava_kaynak = "yüklenen CSV"

yuzey = isi_yuzeyi_oku(ISI_YUZEYI_CSV, os.path.getmtime(ISI_YUZEYI_CSV) if os.path.exists(ISI_YUZEYI_CSV) else 0)
parklar = parklari_guncelle(PARKLAR, yuzey)
tablo = park_saat_tablosu(hava, parklar)
girdi = girdileri_hazirla(tablo, parklar, w_maruziyet, w_kirilganlik)
kirilgan_parklar = set(parklar.loc[parklar["hassas_yuzde"] >= parklar["hassas_yuzde"].median(), "park"])

if yuzey is not None:
    tarih_bilgi = yuzey["landsat_tarih"].iloc[0] if "landsat_tarih" in yuzey else "?"
    yuzey_kaynak = f"Landsat yüzey sıcaklığı ({tarih_bilgi}, yerel sabah geçişi), {len(yuzey)} altıgen"
else:
    yuzey_kaynak = f"yok – ısı haritası için önce veri_hazirla.py çalıştırın ({ISI_YUZEYI_CSV})"
st.caption(f"Hava: {hava_kaynak} · Isı yüzeyi: {yuzey_kaynak} · Nüfus, kırılganlık ve hat kapasitesi temsili · "
           "Soğutma eğrisi literatür tabanlı vekil model, kalibre edilmemiş.")

# --- Optimizasyon ---
c_btn, c_bilgi = st.columns([1, 3])
if c_btn.button("🚀 Günün su planını üret", type="primary"):
    with st.spinner("MILP çözülüyor (nominal + duyarlılık)..."):
        plan, durum = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt)
        if durum != "Optimal":
            st.error(f"Çözücü durumu: {durum}. Kısıtları gevşetin (kırılgan pay veya nem kapısı).")
            st.stop()
        esit = esit_dagitim(girdi, butce)
        plan_e, esit_e = etki(plan), etki(esit)
        artislar = []
        for x0 in X0_ARALIK:  # doz-yanıt belirsizliği: her senaryoda yeniden optimize et
            p2, d2 = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0=x0)
            if d2 == "Optimal":
                fz = etki(esit, x0)["agirlikli_fayda"].sum()
                if fz > 0:
                    artislar.append(etki(p2, x0)["agirlikli_fayda"].sum() / fz - 1)
        sulanan = plan_e[plan_e["su_m3"] > 0.01]
        st.session_state.plan, st.session_state.esit = plan_e, esit_e
        st.session_state.ozet = {
            "artis": plan_e["agirlikli_fayda"].sum() / max(esit_e["agirlikli_fayda"].sum(), 1e-9) - 1,
            "artis_aralik": (min(artislar), max(artislar)) if artislar else None,
            "dusus_aralik": (sogutma_egrisi(sulanan["su_m3"], sulanan["emax"] * 0.5).mean(),
                             sogutma_egrisi(sulanan["su_m3"], sulanan["emax"] * 1.5).mean()) if len(sulanan) else None,
        }
c_bilgi.caption("Saatleri ve miktarları sistem belirler. Siz yalnızca bütçeyi, eşitlik payını ve kuralları girer, "
                "önerilen planı onaylarsınız.")

plan, esit, ozet = st.session_state.plan, st.session_state.esit, st.session_state.ozet

# --- Özet metrikler ---
if ozet is not None:
    pay = lambda d: d.loc[d["park"].isin(kirilgan_parklar), "su_m3"].sum() / max(d["su_m3"].sum(), 1e-9) * 100
    m1, m2, m3, m4 = st.columns(4)
    aralik = ozet["artis_aralik"]
    m1.metric("Öncelik ağırlıklı serinlik kazancı", f"{ozet['artis'] * 100:+.0f}%", "eşit dağıtıma göre", delta_color="off")
    if aralik:
        m1.caption(f"Belirsizlik aralığı: {aralik[0] * 100:+.0f}% … {aralik[1] * 100:+.0f}% (doz-yanıt varsayımına göre)")
    m2.metric("Kullanılan su", f"{plan['su_m3'].sum():.0f} m³", f"bütçe {butce} m³", delta_color="off")
    m3.metric("Kırılgan parklara giden pay", f"%{pay(plan):.0f}", f"{pay(plan) - pay(esit):+.0f} puan")
    if ozet["dusus_aralik"]:
        a, b = ozet["dusus_aralik"]
        m4.metric("Sulanan noktada UTCI düşüşü", f"{a:.1f}–{b:.1f} °C", "tahmini aralık", delta_color="off")

# --- Harita + şu an kartı + oynatma ---
st.subheader("Isı haritası ve saatlik plan")
lejant(yuzey, sicak_pay)
c_harita, c_kart = st.columns([3, 1.3])
harita_yeri = c_harita.empty()
kart_yeri = c_kart.empty()

c_oyna, c_git = st.columns([1, 3])
oynat = c_oyna.button("▶ Günü oynat")
st.session_state.saat = c_git.select_slider("Saate git", options=SAATLER, value=st.session_state.saat,
                                            format_func=lambda s: f"{s:02d}:00", label_visibility="collapsed")
if oynat:
    for s in SAATLER:
        harita_ciz(harita_yeri, s, hava, yuzey, parklar, plan, sicak_pay)
        su_an_karti(kart_yeri, s, hava, plan, parklar)
        time.sleep(0.9)
    st.session_state.saat = SAATLER[-1]
else:
    harita_ciz(harita_yeri, st.session_state.saat, hava, yuzey, parklar, plan, sicak_pay)
    su_an_karti(kart_yeri, st.session_state.saat, hava, plan, parklar)

if plan is None:
    st.info("Planı görmek için “Günün su planını üret” düğmesine basın.")
    st.stop()

# --- Plan matrisi ve operatör komutları ---
st.subheader("🗓️ Günün su planı (park × saat, m³)")
st.altair_chart(plan_matrisi(plan))

st.subheader("🧾 Operatör onay listesi")
komut = operator_komutlari(plan, parklar, aritilmis_senaryo)
if komut.empty:
    st.info("Bu koşullarda sulama önerilmiyor.")
else:
    komut.insert(0, "Onay", False)
    st.data_editor(komut, hide_index=True, disabled=["Park", "Başlangıç", "Bitiş", "Su (m³)", "Kaynak"])
    st.download_button("⬇️ Planı CSV olarak indir", komut.drop(columns="Onay").to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"coolscape_plan_{DEMO_GUNLERI[gun]}.csv", mime="text/csv")

# --- Bütçe – serinlik eğrisi ---
with st.expander("📈 Bütçe–serinlik eğrisi: ek su ne kazandırır?"):
    if st.button("Eğriyi hesapla"):
        satirlar = []
        for b in sorted({int(butce * k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}):
            p_b, d_b = optimize_et(girdi, b, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt)
            if d_b == "Optimal":
                satirlar.append({"Bütçe (m³)": b, "Senaryo": "CoolScape", "Fayda": etki(p_b)["agirlikli_fayda"].sum()})
            satirlar.append({"Bütçe (m³)": b, "Senaryo": "Eşit dağıtım", "Fayda": etki(esit_dagitim(girdi, b))["agirlikli_fayda"].sum()})
        egri = pd.DataFrame(satirlar)
        st.altair_chart(alt.Chart(egri).mark_line(point=True).encode(
            x="Bütçe (m³):Q", y=alt.Y("Fayda:Q", title="Öncelik ağırlıklı serinlik"),
            color=alt.Color("Senaryo:N", scale=alt.Scale(range=["#0078ff", "#999999"]))))
        st.caption("Eğri düzleştiği yerde ek su az ek serinlik getirir: bütçe kararını destekler.")

with st.expander("🔎 Neden? (teknik ayrıntı ve varsayımlar)"):
    s = st.session_state.saat
    d = plan[plan["saat"] == s][["park", "utci_c", "vpd_kpa", "oncelik", "emax", "su_m3", "utci_dusus_c"]].round(2)
    d.columns = ["Park", "UTCI (°C)", "VPD (kPa)", "Öncelik", "Azami etki (°C)", "Su (m³)", "UTCI düşüşü (°C)"]
    st.dataframe(d, hide_index=True)
    st.dataframe(parklar[["park", "yuzey_farki_c", "yesil_skoru", "nufus_skoru", "hassas_yuzde", "kaynak"]]
                 .rename(columns={"park": "Park", "yuzey_farki_c": "Yüzey farkı (°C)", "yesil_skoru": "Yeşil skoru",
                                  "nufus_skoru": "Nüfus skoru (temsili)", "hassas_yuzde": "Kırılgan % (temsili)",
                                  "kaynak": "Yüzey/yeşil kaynağı"}), hide_index=True)
    st.markdown(
        "- **Öncelik** = max(UTCI − 32, 0) × (1 + maruziyet + kırılganlık ağırlıkları)\n"
        "- **Vekil model v0:** UTCI düşüşü = azami etki × (1 − e^(−su/x₀)), x₀ = 8 m³ (duyarlılık: 4–16 m³). "
        "Azami etki ≈ 1 °C ölçeğinde (Paris kaldırım ıslatma ölçümleri), VPD ve yeşil alanla ölçeklenir.\n"
        "- **MILP:** öncelik ağırlıklı UTCI düşüşünü maksimize eder; kısıtlar su bütçesi, hat kapasitesi, en az doz, "
        "nem kapısı ve kırılgan parklara asgari pay.\n"
        "- **MRT** kaba yaklaşımdır (hava sıcaklığı + 0,02 × güneş + yüzey sıcaklığı farkı); gölge ve binalar hesaba katılmaz.\n"
        "- **Landsat** yerel sabah geçer; harita öğleden sonraki pik sıcaklığı değil, yüzeyin göreli ısınma desenini gösterir.\n"
        "- Konyaaltı Sahil Parkı pilot bölgenin (Muratpaşa–Kepez) dışındadır; karşılaştırma için tutulmuştur."
    )





