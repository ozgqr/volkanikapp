# -*- coding: utf-8 -*-
"""
Volkanik Alan Gözlem Paneli — Microsoft Planetary Computer verileriyle
======================================================================

Çalıştırma:
    pip install -r requirements.txt
    streamlit run volkanik_pc_app.py

Veriler çalışma anında Planetary Computer'dan çekilir (hesap / API anahtarı gerekmez):
    cop-dem-glo-30   Copernicus DEM 30 m      -> yükseklik, eğim, pürüzlülük, çöküntüler, aydınlanma
    sentinel-2-l2a   Sentinel-2 yüzey yansıması -> mineral indeksleri, spektral birimler
    esa-worldcover   ESA WorldCover 10 m (2021) -> yüzey örtüsü
    landsat-c2-l2    Landsat 8/9 yüzey sıcaklığı -> 30 m yaz LST haritası
    modis-11A2-061   MODIS 8 günlük LST         -> yıllık gündüz/gece sıcaklık serisi, aşırılıklar
    aster-l1t        ASTER VNIR+SWIR+TIR (2000–2008) -> kil, karbonat, kuvars, mafik mineral indeksleri
    jrc-gsw          JRC Global Surface Water   -> su görülme sıklığı ve değişimi (1984–2020)
    modis-16A3GF-061 MODIS yıllık buharlaşma    -> gerçek / potansiyel buharlaşma-terleme
    modis-10A2-061   MODIS 8 günlük kar örtüsü  -> karla kaplı alan serisi
    modis-13Q1-061   MODIS 16 günlük NDVI/EVI   -> bitki örtüsünün yıl içi seyri (fenoloji)
    modis-17A3HGF-061 MODIS yıllık üretkenlik   -> net birincil üretim

Mars için (hesap gerekmez): NASA Mars Trek servisleri — trek.nasa.gov — üzerinden MOLA + HRSC yükseklik
modeli, Viking mozaiği, TES ve THEMIS ürünleri (ASU), OMEGA + CRISM sulu mineral tespitleri (ESA haritasının
dayandığı tarama), jeoloji haritası ve yapı katalogları.

Sol menü: beş volkanik alan, Mars, iki mekânı yan yana koyan Karşılaştırma ve Mars'ta W tuşuyla sürülen
gezici araç simülasyonu (2D Simulation) ile Karapınar ve Divlit analoglarına dayalı üç boyutlu A.R.E.S.
rota simülasyonu (3D Simulation).
Sekiz inceleme başlığı: jeoloji-mineraloji · topoğrafya · sıcaklık-aydınlanma · su-buz-kaynaklar ·
toz-rüzgâr-erozyon · biyoloji-yaşanabilirlik · mağara-lav tüpü · iniş-hareketlilik-inşaat.

Yalnızca planetarycomputer.microsoft.com adresine istek atılır; ham dosyalar indirilmez,
kırpma işlemi sunucu tarafında yapılır. Bu yüzden rasterio/GDAL kurulumu gerekmez.
"""
from __future__ import annotations

import heapq
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.colors import ListedColormap, LightSource
from matplotlib.patches import Patch
from scipy import ndimage as ndi

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
DATA = "https://planetarycomputer.microsoft.com/api/data/v1"

# --------------------------------------------------------------------------------------
# BÖLGELER  (bbox = [batı, güney, doğu, kuzey], derece).  Sınırları buradan değiştirebilirsiniz.
# --------------------------------------------------------------------------------------
REGIONS = {
    "Acıgöl Volkanik Kompleksi – Nevşehir": {
        "bbox": [34.42, 38.47, 34.66, 38.64],
        "tip": "Kaldera · riyolitik domlar · maarlar",
        "ozet": "Kapadokya'da Kuvaterner yaşlı, iki uçlu (riyolit + bazalt) bir volkanik kompleks. "
                "Obsidiyenli riyolit domları, maarlar, cüruf konileri ve genç bazalt akıntıları bir arada bulunur.",
    },
    "Karacadağ – Diyarbakır / Şanlıurfa": {
        "bbox": [39.55, 37.45, 40.10, 37.90],
        "tip": "Bazaltik kalkan volkanı",
        "ozet": "Güneydoğu Anadolu'da geniş tabanlı, yayvan bir bazaltik kalkan volkanı (zirve ≈ 1950 m). "
                "Kuzey–güney uzanımlı çatlak sistemi boyunca dizilmiş koniler ve çok geniş alkali bazalt akıntıları tipiktir.",
    },
    "Kula Volkanik Jeoparkı – Manisa": {
        "bbox": [28.40, 38.48, 28.76, 38.70],
        "tip": "Cüruf konileri · genç lav akıntıları",
        "ozet": "Batı Anadolu'da üç evreli Kuvaterner alkali bazalt volkanizması. Onlarca cüruf konisi, "
                "taze 'yanık' lav akıntıları ve UNESCO Küresel Jeoparkı statüsü ile bilinir.",
    },
    "Meke Gölü ve Karapınar – Konya": {
        "bbox": [33.55, 37.62, 33.76, 37.78],
        "tip": "Maar · maar içi cüruf konisi",
        "ozet": "Karapınar volkanik alanında bir maar çukuru ve ortasında sonradan oluşmuş cüruf konisi. "
                "Yakınında Acıgöl maarı, Meke Dağı ve yarı kurak ovada cüruf konileri yer alır.",
    },
    "Hassa Lav Tüpleri – Hatay": {
        "bbox": [36.46, 36.62, 36.70, 36.84],
        "tip": "Bazalt platosu · lav tüpleri",
        "ozet": "Karasu Vadisi (Ölü Deniz Fay Zonu) boyunca çıkmış Kuvaterner bazaltlarının oluşturduğu plato. "
                "Lav tüpü mağaraları, tümülüsler ve akıntı yapıları görülür. "
                "Sınır kutusu Hassa bazalt platosunu yaklaşık olarak kapsar.",
    },
}

WORLDCOVER = {  # sınıf: (ad, renk)
    10: ("Ağaç örtüsü", "#006400"), 20: ("Çalılık", "#ffbb22"), 30: ("Otlak", "#ffff4c"),
    40: ("Tarım alanı", "#f096ff"), 50: ("Yapılaşma", "#fa0000"), 60: ("Çıplak / seyrek bitkili", "#b4b4b4"),
    70: ("Kar / buz", "#f0f0f0"), 80: ("Su", "#0064c8"), 90: ("Sulak alan", "#0096a0"),
    95: ("Mangrov", "#00cf75"), 100: ("Yosun / liken", "#fae6a0"),
}
ELEV_CMAP = ListedColormap(plt.cm.terrain(np.linspace(0.25, 1.0, 256)))  # mavi (su gibi görünen) uç atıldı
UNIT_COLORS = ["#2b2b33", "#7a4a36", "#b9794a", "#d8c08a", "#f3ecd6"]  # koyudan açığa spektral birimler

# --------------------------------------------------------------------------------------
# PLANETARY COMPUTER ERİŞİMİ
# --------------------------------------------------------------------------------------
_session = requests.Session()
_session.headers["User-Agent"] = "volkanik-pc-app/1.0"


def _request(method: str, url: str, tries: int = 4, **kw) -> requests.Response:
    """Geçici hatalarda (429/5xx/zaman aşımı) yeniden dener."""
    last = None
    for i in range(tries):
        try:
            r = _session.request(method, url, timeout=120, **kw)
            if r.status_code < 400:
                return r
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:  # ağ hatası
            last = e
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Planetary Computer isteği başarısız: {last}")


def stac_search(body: dict, max_items: int = 500) -> list[dict]:
    items, url = [], STAC
    while url and len(items) < max_items:
        j = _request("POST", url, json=body).json()
        items += j.get("features", [])
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt:
            break
        url, body = nxt[0]["href"], nxt[0].get("body", body)
    return items


# --------------------------------------------------------------------------------------
# MARS VERİ ERİŞİMİ — NASA Mars Trek (WMTS karoları, ArcGIS görüntü ve öznitelik servisleri)
# --------------------------------------------------------------------------------------
# Kaynaklar:
#   trek.nasa.gov/mars  -> MOLA+HRSC yükseklik modeli (sayısal), MOLA pürüzlülük, Viking renk mozaiği,
#                          küresel jeoloji haritası, vadi ağları, kumul alanları, buzul benzeri yapılar vb.
#   mars.asu.edu        -> TES albedo / ısıl atalet / toz indeksi / mineral haritaları ve THEMIS gündüz-gece
#                          kızılötesi mozaikleri ASU ürünleridir; uygulama bunları Mars Trek servisinden çeker.
#   ESA sulu mineral haritası -> OMEGA + CRISM tespitleri (Carter vd.); nokta verisi Mars Trek'ten çekilir,
#                          ESA'nın 2022 görseli ayrıca referans olarak gösterilir.
TREK_TILES = "https://trek.nasa.gov/tiles/Mars/EQ"
TREK_ARC = "https://trek.nasa.gov/mars/trekarcgis/rest/services"
ESA_HYDRATED_PNG = ("https://media.esa.int/legacy/web/images/esa_multimedia/images/2022/08/"
                    "global_map_of_hydrated_minerals_on_mars/24409513-1-eng-GB/Global_map_of_hydrated_minerals_on_Mars.png")
ESA_HYDRATED_PAGE = "https://www.esa.int/media/global-map-of-hydrated-minerals-on-mars"

MARS_REGIONS = {
    "Jezero Krateri (Perseverance)": {
        "bbox": [76.0, 17.0, 79.0, 20.0], "tip": "Krater gölü · delta · karbonat ve kil",
        "ozet": "Eski bir krater gölü ve iyi korunmuş nehir deltası. Karbonat, kil ve olivince zengin birimler "
                "bir arada; Perseverance aracı 2021'den beri burada örnek topluyor."},
    "Gale Krateri (Curiosity)": {
        "bbox": [136.3, -6.9, 139.3, -3.9], "tip": "Krater · katmanlı çökel dağı (Aeolis Mons)",
        "ozet": "Ortasında 5 km yüksekliğinde katmanlı bir çökel dağı bulunan çarpma krateri. Alt katmanlarda "
                "kil ve sülfat, eski göl ve akarsu çökelleri; Curiosity 2012'den beri inceliyor."},
    "Arsia Mons (lav tüpü çukurları)": {
        "bbox": [-123.0, -11.5, -117.0, -5.5], "tip": "Kalkan volkanı · kaldera · tavan pencereleri",
        "ozet": "Tharsis'in dev kalkan volkanlarından biri. Yamaçlarında lav tüplerine açılan çökme çukurları "
                "('yedi kız kardeş' tavan pencereleri) ve kanal dizileri bilinir."},
    "Valles Marineris – Melas Chasma": {
        "bbox": [-75.5, -13.0, -69.5, -7.0], "tip": "Kanyon sistemi · katmanlı sülfatlar",
        "ozet": "Güneş Sistemi'nin en büyük kanyon sisteminin orta kesimi. 8–9 km derin yarlar, heyelanlar ve "
                "tabanda sülfatça zengin katmanlı çökeller."},
    "Mawrth Vallis (kil yatakları)": {
        "bbox": [-21.0, 22.0, -17.0, 26.0], "tip": "Taşkın kanalı · Fe/Mg ve Al killeri",
        "ozet": "Mars'ın en geniş ve en çeşitli kil yüzleklerinden biri. Uzun süreli sulu alterasyonun kaydı; "
                "ExoMars için aday iniş alanlarındandı."},
    "Elysium – Cerberus Fossae": {
        "bbox": [160.0, 7.5, 165.0, 12.5], "tip": "Genç lav ovaları · çatlak sistemi",
        "ozet": "Mars'ın en genç lav akıntıları ve hâlâ sismik olarak etkin çatlak sistemi. Geniş, düz bazalt "
                "ovaları; InSight bu bölgenin batısına indi."},
    "Deuteronilus Mensae (buzul benzeri yapılar)": {
        "bbox": [21.0, 41.5, 25.0, 45.5], "tip": "Orta enlem buzulları · dolgu vadileri",
        "ozet": "Kuzey orta enlemlerde, toz örtüsü altında su buzu barındıran loblu enkaz önlükleri ve vadi "
                "dolguları. Radar ölçümleri kalın, neredeyse saf buz gösterir."},
}

# gezegen sabitleri: derece başına metre, eksen eğikliği, güneş sabiti, yörünge dış merkezliği, günberi Ls'i
BODIES = {
    "earth": {"m_lon": 111_320.0, "m_lat": 110_574.0, "obliq": 23.44, "S0": 1361.0, "ecc": 0.0167, "ls_peri": 283.0,
              "dem_label": "Copernicus DEM 30 m", "ad": "Dünya"},
    "mars": {"m_lon": 59_274.0, "m_lat": 59_274.0, "obliq": 25.19, "S0": 589.0, "ecc": 0.0934, "ls_peri": 251.0,
             "dem_label": "MOLA + HRSC, 200 m", "ad": "Mars"},
}
BODY = dict(BODIES["earth"])  # etkin gezegen; set_body ile değişir


def set_body(name: str):
    BODY.clear()
    BODY.update(BODIES[name])
    BODY["key"] = name


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def trek_layer_info(layer: str):
    """Karo katmanının en yüksek yakınlaştırma düzeyi ve dosya uzantısı."""
    xml = _request("GET", f"{TREK_TILES}/{layer}/1.0.0/WMTSCapabilities.xml").text
    import re
    ids = [int(x) for x in re.findall(r"<TileMatrix>\s*<ows:Identifier>(\d+)</ows:Identifier>", xml)]
    ext = "jpg" if "image/jpeg" in xml else "png"
    return (max(ids) if ids else 3), ext


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def trek_image(layer: str, bbox: tuple, W: int, H: int, smooth: bool = True):
    """WMTS karolarını birleştirip bbox'a kırpar. Dönen: RGBA dizi [H, W, 4] (0–255)."""
    from PIL import Image
    maxz, ext = trek_layer_info(layer)
    need = (bbox[2] - bbox[0]) / W  # derece / piksel
    z = int(min(maxz, max(0, math.ceil(math.log2(180 / (256 * need))))))
    td = 180 / 2 ** z  # karo genişliği (derece)
    c0, c1 = int((bbox[0] + 180) // td), int(min(2 ** (z + 1) - 1, (bbox[2] + 180 - 1e-9) // td))
    r0, r1 = int(max(0, (90 - bbox[3]) // td)), int(min(2 ** z - 1, (90 - bbox[1] - 1e-9) // td))
    jobs = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def one(rc):
        url = f"{TREK_TILES}/{layer}/1.0.0/default/default028mm/{z}/{rc[0]}/{rc[1]}.{ext}"
        try:
            return rc, Image.open(io.BytesIO(_request("GET", url, tries=2).content)).convert("RGBA")
        except Exception:
            return rc, None

    canvas = Image.new("RGBA", ((c1 - c0 + 1) * 256, (r1 - r0 + 1) * 256), (0, 0, 0, 0))
    with ThreadPoolExecutor(8) as ex:
        for (r, c), im in ex.map(one, jobs):
            if im is not None:
                canvas.paste(im.resize((256, 256)), ((c - c0) * 256, (r - r0) * 256))
    x0 = (bbox[0] + 180 - c0 * td) / td * 256
    x1 = (bbox[2] + 180 - c0 * td) / td * 256
    y0 = (90 - bbox[3] - r0 * td) / td * 256
    y1 = (90 - bbox[1] - r0 * td) / td * 256
    out = canvas.crop((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
    out = out.resize((W, H), Image.BILINEAR if smooth else Image.NEAREST)
    return np.asarray(out)


def rainbow_value(rgba):
    """Gökkuşağı renk ölçekli haritayı (mor = düşük, kırmızı = yüksek) 0–1 göreli değere çevirir."""
    from matplotlib.colors import rgb_to_hsv
    hsv = rgb_to_hsv(rgba[..., :3].astype("float32") / 255)
    h = hsv[..., 0].copy()
    h[h > 0.88] = 0.0  # kırmızının mor tarafa taşan ucu
    val = np.clip((0.78 - h) / 0.78, 0, 1)
    ok = (rgba[..., 3] > 0) & (hsv[..., 1] > 0.35) & (hsv[..., 2] > 0.2)
    return np.where(ok, val, np.nan).astype("float32")


def gray_value(rgba):
    v = rgba[..., :3].astype("float32").mean(axis=-1) / 255
    return np.where((rgba[..., 3] > 0) & (v > 0.004), v, np.nan)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_mars_dem(bbox: tuple, n: int):
    """MOLA + HRSC harmanlanmış yükseklik modeli (sayısal, metre) — ArcGIS görüntü servisinden."""
    from PIL import Image
    set_body("mars")
    W, H, px = grid_shape(bbox, n)
    url = f"{TREK_ARC}/Mars_MOLA_blend200ppx_HRSC_DEM_clon0dd_200mpp_lzw/ImageServer/exportImage"
    r = _request("GET", url, params={"bbox": ",".join(map(str, bbox)), "size": f"{W},{H}", "format": "tiff",
                                     "pixelType": "S16", "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    dem = np.asarray(Image.open(io.BytesIO(r.content))).astype("float32")
    if dem.shape != (H, W):
        raise RuntimeError("Mars yükseklik servisi beklenen boyutta veri döndürmedi.")
    return dem, px


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def mars_features(service: str, bbox: tuple, geometry: bool = True):
    """ArcGIS öznitelik sorgusu: bbox ile kesişen kayıtlar. Dönen: [{'a': öznitelikler, 'g': geometri}]"""
    out, last = [], -1
    for _ in range(8):  # servis sayfalama desteklemediği için FID üzerinden ilerlenir (≤ 1000 kayıt / istek)
        j = _request("GET", f"{TREK_ARC}/{service}/query", params={
            "where": f"FID > {last}", "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "orderByFields": "FID",
            "returnGeometry": "true" if geometry else "false", "f": "json",
            "maxAllowableOffset": (bbox[2] - bbox[0]) / 1500}).json()  # geometriyi sunucuda sadeleştir
        feats = j.get("features", [])
        out += [{"a": f.get("attributes", {}), "g": f.get("geometry")} for f in feats]
        if len(feats) < 1000:
            break
        last = max(f["attributes"]["FID"] for f in feats)
    return out


def feature_points(feats):
    """Nokta katmanı -> (boylam, enlem) dizileri."""
    xy = [(f["g"]["x"], f["g"]["y"]) for f in feats if f.get("g") and "x" in f["g"]]
    return (np.array([p[0] for p in xy]), np.array([p[1] for p in xy])) if xy else (np.array([]), np.array([]))


def rasterize_polygons(feats, bbox, shape, key=None):
    """Poligonları ızgaraya işler. key verilirse o özniteliğin sınıf indeksini, yoksa 1 yazar. Dönen: (ızgara, sınıf adları)"""
    from matplotlib.path import Path
    H, W = shape
    lon = bbox[0] + (np.arange(W) + 0.5) / W * (bbox[2] - bbox[0])
    lat = bbox[3] - (np.arange(H) + 0.5) / H * (bbox[3] - bbox[1])
    names = sorted({str(f["a"].get(key)) for f in feats}) if key else ["1"]
    grid = np.zeros(shape, dtype="int16")
    for f in feats:
        g = f.get("g") or {}
        val = names.index(str(f["a"].get(key))) + 1 if key else 1
        for ring in g.get("rings", []):
            r = np.asarray(ring)
            c0, c1 = np.searchsorted(lon, [r[:, 0].min(), r[:, 0].max()])
            r1, r0 = H - np.searchsorted(lat[::-1], [r[:, 1].min(), r[:, 1].max()])
            r0, r1 = max(0, r0), min(H, r1)
            if c1 <= c0 or r1 <= r0:
                continue
            gx, gy = np.meshgrid(lon[c0:c1], lat[r0:r1])
            inside = Path(r).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            sub = grid[r0:r1, c0:c1]
            sub[inside] = np.where(sub[inside] == val, 0, val)  # iç halkalar (delikler) aynı sınıfı geri siler
    return grid, names


def grid_shape(bbox, n: int):
    """bbox için yaklaşık kare pikselli ızgara: (genişlik, yükseklik, piksel boyu [m])."""
    w, s, e, nn = bbox
    latc = math.radians((s + nn) / 2)
    wm = (e - w) * BODY["m_lon"] * math.cos(latc)
    hm = (nn - s) * BODY["m_lat"]
    k = n / max(wm, hm)
    W, H = max(8, round(wm * k)), max(8, round(hm * k))
    return W, H, wm / W


def crop(collection: str, item_id: str, assets: list[str], bbox, W: int, H: int, smooth=True):
    """Bir STAC öğesini sunucu tarafında bbox'a kırpar. Dönen: (bantlar[b,H,W] float32, geçerlilik[H,W] bool)."""
    url = f"{DATA}/item/bbox/{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/{W}x{H}.npy"
    rs = "bilinear" if smooth else "nearest"
    params = [("collection", collection), ("item", item_id), ("resampling", rs), ("reproject", rs)]
    params += [("assets", a) for a in assets]
    arr = np.load(io.BytesIO(_request("GET", url, params=params).content))
    return arr[:-1].astype("float32"), arr[-1] > 0


def mosaic(collection, item_ids, assets, bbox, W, H, smooth=True, nodata=None):
    """Birden çok öğeyi (ör. komşu DEM karoları) geçerli piksellere göre birleştirir."""
    out, filled = None, None
    for iid in item_ids:
        data, ok = crop(collection, iid, assets, bbox, W, H, smooth)
        if nodata is not None:
            ok &= data[0] != nodata
        if out is None:
            out, filled = np.where(ok, data, np.nan), ok.copy()
        else:
            take = ok & ~filled
            out[:, take] = data[:, take]
            filled |= ok
        if filled.all():
            break
    return out, filled


def _best_scenes(collection, items, probe_asset, bbox, nodata=0, max_try=12):
    """Bulutu en az sahnelerden bbox'ı en iyi kaplayanları seçer (küçük bir deneme kırpmasıyla)."""
    cand = items[:max_try]

    def cover(it):
        try:
            d, ok = crop(collection, it["id"], [probe_asset], bbox, 48, 48, smooth=False)
            return float((ok & (d[0] != nodata)).mean())
        except Exception:
            return 0.0

    with ThreadPoolExecutor(8) as ex:
        cov = list(ex.map(cover, cand))
    for it, c in zip(cand, cov):  # liste bulutluluğa göre sıralı: ilk tam kaplayan en iyisidir
        if c >= 0.995:
            return [it]
    order = sorted(range(len(cand)), key=lambda i: -cov[i])
    return [cand[i] for i in order if cov[i] > 0.02][:4]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_dem(bbox: tuple, n: int):
    W, H, px = grid_shape(bbox, n)
    items = stac_search({"collections": ["cop-dem-glo-30"], "bbox": list(bbox), "limit": 20})
    if not items:
        raise RuntimeError("Bu alan için DEM bulunamadı.")
    # DEM kendi ızgarasında (1 yay saniyesi, 2'nin katı seyreltme ile) okunur ve kare piksele burada
    # örneklenir; sunucunun doğrudan yeniden boyutlandırması eğim haritasında çizgi izi bırakıyor.
    nw, nh = round((bbox[2] - bbox[0]) * 3600), round((bbox[3] - bbox[1]) * 3600)
    k = 1
    while max(nw, nh) // (2 * k) >= max(W, H):
        k *= 2
    w, h = nw // k, nh // k
    dem, ok = mosaic("cop-dem-glo-30", [i["id"] for i in items], ["data"], bbox, w, h)
    dem = dem[0]
    if np.isnan(dem).any():  # küçük boşlukları en yakın değerle doldur
        idx = ndi.distance_transform_edt(np.isnan(dem), return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]
    f = max(w / W, h / H)
    if f > 1.2:
        dem = ndi.uniform_filter(dem, size=int(round(f)), mode="nearest")
    dem = ndi.zoom(dem, (H / h, W / w), order=1, mode="nearest")[:H, :W]
    return dem.astype("float32"), px


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_s2(bbox: tuple, n: int, year: int):
    W, H, _ = grid_shape(bbox, n)
    items = stac_search({
        "collections": ["sentinel-2-l2a"], "bbox": list(bbox),
        "datetime": f"{year}-06-15T00:00:00Z/{year}-09-30T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 15}}, "limit": 60,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }, max_items=60)
    if not items:
        raise RuntimeError(f"{year} yazı için bulutsuz Sentinel-2 sahnesi bulunamadı.")
    scenes = _best_scenes("sentinel-2-l2a", items, "B04", bbox)
    bands = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
    refl, scl = None, None
    for it in scenes:
        d, ok = crop("sentinel-2-l2a", it["id"], bands, bbox, W, H)
        s, _ = crop("sentinel-2-l2a", it["id"], ["SCL"], bbox, W, H, smooth=False)
        ok &= d[0] > 0
        # İşleme sürümü 04.00 ve sonrası sahnelerde sayısal değerlere +1000 eklenmiştir
        off = 1000.0 if float(it["properties"].get("s2:processing_baseline", "0") or 0) >= 4 else 0.0
        r = np.where(ok, (d - off) / 10000.0, np.nan)
        if refl is None:
            refl, scl = r, np.where(ok, s[0], 0)
        else:
            take = ok & np.isnan(refl[0])
            refl[:, take], scl[take] = r[:, take], s[0][take]
    refl = np.clip(refl, 1e-4, 1.5)
    return {
        "refl": dict(zip(bands, refl)), "scl": scl.astype("uint8"),
        "tarih": ", ".join(sorted({i["properties"]["datetime"][:10] for i in scenes})),
        "bulut": min(i["properties"]["eo:cloud_cover"] for i in scenes),
    }


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_worldcover(bbox: tuple, n: int):
    W, H, _ = grid_shape(bbox, n)
    items = stac_search({"collections": ["esa-worldcover"], "bbox": list(bbox), "limit": 40})
    ids = [i["id"] for i in items if "_2021_" in i["id"]] or [i["id"] for i in items]
    if not ids:
        raise RuntimeError("WorldCover verisi bulunamadı.")
    wc, _ = mosaic("esa-worldcover", ids, ["map"], bbox, W, H, smooth=False, nodata=0)
    return np.nan_to_num(wc[0]).astype("uint8")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_landsat_lst(bbox: tuple, n: int, year: int):
    W, H, _ = grid_shape(bbox, n)
    items = stac_search({
        "collections": ["landsat-c2-l2"], "bbox": list(bbox),
        "datetime": f"{year}-06-15T00:00:00Z/{year}-09-15T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 15}, "platform": {"in": ["landsat-8", "landsat-9"]}},
        "limit": 60, "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }, max_items=60)
    if not items:
        raise RuntimeError(f"{year} yazı için bulutsuz Landsat sahnesi bulunamadı.")
    scenes = _best_scenes("landsat-c2-l2", items, "lwir11", bbox, max_try=16)
    lst = None
    for it in scenes:  # tek sahne kutuyu kaplamıyorsa komşu sahnelerle tamamlanır
        d, ok = crop("landsat-c2-l2", it["id"], ["lwir11"], bbox, W, H)
        qa, _ = crop("landsat-c2-l2", it["id"], ["qa_pixel"], bbox, W, H, smooth=False)
        cloudy = (qa[0].astype("uint16") & 0b11110) > 0  # genişletilmiş bulut, sirrus, bulut, bulut gölgesi
        t = d[0] * 0.00341802 + 149.0 - 273.15  # Kelvin ölçeği -> °C
        t[~ok | (d[0] == 0) | cloudy] = np.nan
        lst = t if lst is None else np.where(np.isnan(lst), t, lst)
        if np.isfinite(lst).mean() > 0.98:
            break
    p = scenes[0]["properties"]
    tarih = ", ".join(sorted({i["properties"]["datetime"][:10] for i in scenes}))
    return {"lst": lst, "tarih": tarih, "gunes_yuk": p.get("view:sun_elevation"),
            "uydu": "Landsat 8/9", "sahne_sayisi": len(scenes)}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_modis_series(bbox: tuple, year: int) -> pd.DataFrame:
    """MODIS/Terra 8 günlük LST: alan ortalaması, en sıcak ve en soğuk piksel."""
    items = stac_search({"collections": ["modis-11A2-061"], "bbox": list(bbox),
                         "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 250})
    items = [i for i in items if i["id"].startswith("MOD11A2")]  # Terra (≈10:30 / 22:30 geçişi)
    if not items:
        raise RuntimeError(f"{year} için MODIS LST verisi bulunamadı.")
    km_w = (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    km_h = (bbox[3] - bbox[1]) * 110.57
    W, H = max(8, round(km_w)), max(8, round(km_h))  # ≈ 1 km piksel

    def one(it):
        try:
            d, ok = crop("modis-11A2-061", it["id"], ["LST_Day_1km", "LST_Night_1km"], bbox, W, H, smooth=False)
        except Exception:
            return None
        day = np.where(ok & (d[0] > 0), d[0] * 0.02 - 273.15, np.nan)
        ngt = np.where(ok & (d[1] > 0), d[1] * 0.02 - 273.15, np.nan)
        if np.isfinite(day).mean() < 0.2 and np.isfinite(ngt).mean() < 0.2:
            return None
        f = lambda fn, a: float(fn(a)) if np.isfinite(a).any() else np.nan
        return {"tarih": pd.Timestamp(it["properties"]["start_datetime"][:10]),
                "gunduz": f(np.nanmean, day), "gece": f(np.nanmean, ngt),
                "gunduz_maks": f(np.nanmax, day), "gece_min": f(np.nanmin, ngt)}

    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, items) if r]
    if not rows:
        raise RuntimeError(f"{year} için kullanılabilir MODIS LST verisi yok.")
    df = pd.DataFrame(rows).groupby("tarih").mean()
    grid = pd.date_range(f"{year}-01-01", periods=46, freq="8D")  # eksik dönemler boş kalır
    df = df.reindex(grid).rename_axis("tarih").reset_index()
    df["genlik"] = df["gunduz"] - df["gece"]
    return df


def modis_best_year(bbox: tuple, year: int):
    """Seçilen yılda katalogda eksik dönem çoksa, en yakın eksiksiz yıla düşer."""
    best = None
    for y in (year, year - 1, year - 2):
        try:
            df = get_modis_series(bbox, y)
        except Exception:
            continue
        nvalid = int(df["gunduz"].notna().sum())
        if nvalid >= 40:
            return df, y, nvalid
        if best is None or nvalid > best[2]:
            best = (df, y, nvalid)
    if best is None:
        raise RuntimeError("MODIS LST verisi alınamadı.")
    return best


# --------------------------------------------------------------------------------------
# TOPOĞRAFYA ANALİZİ
# --------------------------------------------------------------------------------------
def slope_normals(dem, px):
    g_s, g_e = np.gradient(dem, px)  # satır ekseni güneye doğru artar
    dzdx, dzdy = g_e, -g_s           # doğu ve kuzey yönünde eğim
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    norm = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1)
    return slope, (-dzdx / norm, -dzdy / norm, 1 / norm)


def roughness_tri(dem):
    """Arazi Pürüzlülük İndeksi (Riley TRI): 8 komşuyla yükseklik farklarının karekök ortalaması [m]."""
    p = np.pad(dem, 1, mode="edge")
    acc = np.zeros_like(dem)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                acc += (p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc] - dem) ** 2
    return np.sqrt(acc / 8)


@st.cache_data(show_spinner=False)
def fill_depressions(dem):
    """Öncelikli taşkın (priority-flood) ile kapalı çukurları doldurur; dolgu − DEM = çukur derinliği."""
    H, W = dem.shape
    z = dem.ravel().tolist()
    done = bytearray(H * W)
    heap = []
    for r in range(H):
        for c in ((0, W - 1) if 0 < r < H - 1 else range(W)):
            i = r * W + c
            done[i] = 1
            heap.append((z[i], i))
    heapq.heapify(heap)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        h, i = pop(heap)
        r, c = divmod(i, W)
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= H:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= W:
                    continue
                j = rr * W + cc
                if done[j]:
                    continue
                done[j] = 1
                if z[j] < h:
                    z[j] = h
                push(heap, (z[j], j))
    return np.asarray(z, dtype="float32").reshape(H, W)


def find_depressions(dem, px, bbox, min_depth=5.0, min_area_km2=0.01):
    depth = fill_depressions(dem) - dem
    lab, n = ndi.label(depth > 1.0)
    rows = []
    if n:
        idx = np.arange(1, n + 1)
        area = ndi.sum(np.ones_like(depth), lab, idx) * px * px / 1e6
        dmax = ndi.maximum(depth, lab, idx)
        pos = ndi.maximum_position(depth, lab, idx)
        for k in range(n):
            if dmax[k] >= min_depth and area[k] >= min_area_km2:
                r, c = pos[k]
                lon, lat = rc_to_lonlat(r, c, dem.shape, bbox)
                rows.append({"Enlem": round(lat, 4), "Boylam": round(lon, 4),
                             "Derinlik (m)": round(float(dmax[k]), 1),
                             "Alan (km²)": round(float(area[k]), 3),
                             "Eşdeğer çap (m)": int(round(2 * math.sqrt(area[k] * 1e6 / math.pi))),
                             "Taban yüks. (m)": int(round(float(dem[r, c]))), "_r": r, "_c": c, "_lab": k + 1})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("Derinlik (m)", ascending=False).head(15).reset_index(drop=True)
        df.index = [f"Ç{i + 1}" for i in range(len(df))]
    return depth, lab, df


def find_peaks(dem, px, bbox, radius_m=1000, min_height=30.0):
    """Çevresinden belirgin biçimde yüksek tepeler (koni / dom adayları)."""
    radius_m = max(radius_m, 3 * px)  # kaba ızgarada (Mars) pencere en az birkaç piksel olsun
    k = max(3, int(round(2 * radius_m / px)) | 1)
    tpi = dem - ndi.uniform_filter(dem, size=k, mode="nearest")
    win = max(3, int(round(max(1200, 4 * px) / px)) | 1)
    is_max = (dem == ndi.maximum_filter(dem, size=win, mode="nearest")) & (tpi >= min_height)
    b = win // 2  # çerçeve kenarındaki sahte zirveleri at
    is_max[:b], is_max[-b:], is_max[:, :b], is_max[:, -b:] = False, False, False, False
    rr, cc = np.nonzero(is_max)
    rows = []
    for r, c in zip(rr, cc):
        lon, lat = rc_to_lonlat(r, c, dem.shape, bbox)
        rows.append({"Enlem": round(lat, 4), "Boylam": round(lon, 4), "Zirve (m)": int(round(float(dem[r, c]))),
                     "Çevreden yükseklik (m)": int(round(float(tpi[r, c]))), "_r": int(r), "_c": int(c)})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("Çevreden yükseklik (m)", ascending=False).head(15).reset_index(drop=True)
        df.index = [f"T{i + 1}" for i in range(len(df))]
    return tpi, df


def rc_to_lonlat(r, c, shape, bbox):
    H, W = shape
    return bbox[0] + (c + 0.5) / W * (bbox[2] - bbox[0]), bbox[3] - (r + 0.5) / H * (bbox[3] - bbox[1])


# --------------------------------------------------------------------------------------
# AYDINLANMA (güneş geometrisi + arazi gölgesi)
# --------------------------------------------------------------------------------------
def sun_position(lat_deg, doy, solar_hour):
    """Güneş yüksekliği ve azimutu (derece; azimut kuzeyden saat yönünde). Yerel güneş saati kullanılır."""
    phi = math.radians(lat_deg)
    dec = math.radians(BODY["obliq"]) * math.sin(2 * math.pi * (284 + doy) / 365)  # 172. gün = kuzey yazı, 355 = kuzey kışı
    ha = math.radians(15 * (solar_hour - 12))
    alt = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha))
    az = math.atan2(math.sin(ha), math.cos(ha) * math.sin(phi) - math.tan(dec) * math.cos(phi)) + math.pi
    return math.degrees(alt), math.degrees(az) % 360


def _shift(a, dr, dc):
    """out[r,c] = a[r+dr, c+dc]; ızgara dışı = -inf (gölge üretmez)."""
    out = np.full_like(a, -np.inf)
    H, W = a.shape
    r0, r1 = max(0, -dr), min(H, H - dr)
    c0, c1 = max(0, -dc), min(W, W - dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def lit_fraction(dem, px, normals, alt, az):
    """Verilen güneş konumu için: (doğrudan ışık alan pikseller, cos(geliş açısı))."""
    a, z = math.radians(alt), math.radians(az)
    sx, sy, sz = math.cos(a) * math.sin(z), math.cos(a) * math.cos(z), math.sin(a)
    cosi = normals[0] * sx + normals[1] * sy + normals[2] * sz
    lit = cosi > 0
    relief = float(dem.max() - dem.min())
    kmax = min(max(dem.shape), int(relief / (px * math.tan(a))) + 1)
    k, steps = 1.0, []
    while k <= kmax:
        steps.append(int(round(k)))
        k = max(k + 1, k * 1.15)
    for k in sorted(set(steps)):  # güneş yönünde ufka bak: daha yüksek arazi varsa gölgede
        dr, dc = -int(round(k * math.cos(z))), int(round(k * math.sin(z)))
        if dr == 0 and dc == 0:
            continue
        lit &= _shift(dem, dr, dc) <= dem + math.hypot(dr, dc) * px * math.tan(a)
    return lit, np.clip(cosi, 0, None)


@st.cache_data(show_spinner=False)
def illumination(dem, px, lat, doy, step_h=0.5):
    """Bir gün için doğrudan güneş alma süresi [saat], göreli ışınım ve saatlik gölge oranı."""
    f = max(1, int(math.ceil(max(dem.shape) / 280)))  # hız için kabalaştır
    if f > 1:
        H, W = (dem.shape[0] // f) * f, (dem.shape[1] // f) * f
        d = dem[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))
    else:
        d = dem
    p = px * f
    _, normals = slope_normals(d, p)
    hours = np.zeros_like(d)
    energy = np.zeros_like(d)
    rows = []
    for t in np.arange(3.0, 21.01, step_h):
        alt, az = sun_position(lat, doy, t)
        if alt <= 1.0:
            continue
        lit, cosi = lit_fraction(d, p, normals, alt, az)
        hours += lit * step_h
        energy += np.where(lit, cosi, 0) * step_h
        rows.append({"saat": t, "golge_%": 100 * (1 - lit.mean()), "gunes_yuk": alt})
    flat = sum(max(0.0, math.sin(math.radians(sun_position(lat, doy, t)[0]))) * step_h
               for t in np.arange(3.0, 21.01, step_h))
    return hours, energy / max(flat, 1e-6), pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# SPEKTRAL ANALİZ (Sentinel-2)
# --------------------------------------------------------------------------------------
def spectral_products(s2):
    R = s2["refl"]
    scl = s2["scl"]
    ndvi = (R["B08"] - R["B04"]) / (R["B08"] + R["B04"])
    bad = np.isin(scl, [0, 1, 3, 8, 9, 10, 11]) | np.isnan(R["B04"])  # veri yok, gölge, bulut, kar
    water = (scl == 6) | ((R["B03"] - R["B08"]) / (R["B03"] + R["B08"]) > 0.1)
    veg = ndvi > 0.30
    rock = ~(bad | water | veg)  # mineral yorumuna uygun çıplak yüzey
    idx = {
        "Ferrik demir oksit  (B04/B02)": R["B04"] / R["B02"],
        "Ferröz demir  (B12/B8A + B03/B04)": R["B12"] / R["B8A"] + R["B03"] / R["B04"],
        "Kil / hidroksil  (B11/B12)": R["B11"] / R["B12"],
    }
    return ndvi, rock, water, veg, bad, idx


def stretch(a, lo=2, hi=98, mask=None):
    v = a[mask] if mask is not None and mask.any() else a[np.isfinite(a)]
    p0, p1 = np.nanpercentile(v, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-6), 0, 1)


def rgb(R, bands):
    return np.nan_to_num(np.dstack([stretch(R[b]) for b in bands]))


@st.cache_data(show_spinner=False)
def spectral_units(stack, rock, k=5, seed=7):
    """Çıplak yüzeyi k-ortalamalar ile spektral birimlere ayırır; birimler koyudan açığa sıralanır."""
    X = np.log(stack[:, rock].T)
    if len(X) < 50 * k:
        return None, None
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
    cen = sub[rng.choice(len(sub), k, replace=False)]
    for _ in range(25):
        lab = np.argmin(((sub[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        new = np.array([sub[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
        if np.allclose(new, cen, atol=1e-4):
            break
        cen = new
    cen = cen[np.argsort(cen.mean(1))]  # parlaklığa göre sırala
    full = np.concatenate([np.argmin(((X[i:i + 50000, None, :] - cen[None]) ** 2).sum(-1), axis=1)
                           for i in range(0, len(X), 50000)])
    out = np.full(rock.shape, -1, dtype="int8")
    out[rock] = full
    return out, np.exp(cen)


def unit_hint(spec):
    """Birim ortalama spektrumundan temkinli bir yorum üretir (bands: B02,B03,B04,B8A,B11,B12)."""
    albedo = float(spec.mean())
    ferric, clay = spec[2] / spec[0], spec[4] / spec[5]
    tags = []
    if albedo < 0.13:
        tags.append("koyu yüzey — taze bazalt / cüruf / lav akıntısı olabilir")
    elif albedo > 0.30:
        tags.append("açık renkli yüzey — tüf, pomza/riyolit, karbonat veya tuzlu-kuru zemin olabilir")
    else:
        tags.append("orta parlaklık — ayrışmış kaya, toprak veya karışık örtü")
    if ferric > 1.9:
        tags.append("demir oksitçe zengin (kızıl cüruf / oksitlenmiş toprak)")
    if clay > 1.25:
        tags.append("kil-hidroksil sinyali (ayrışma / alterasyon)")
    return "; ".join(tags)


# --------------------------------------------------------------------------------------
# ÇİZİM YARDIMCILARI
# --------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def _axes(bbox, figsize=(5.2, 4.6)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_aspect(1 / math.cos(math.radians((bbox[1] + bbox[3]) / 2)))
    ax.set_xlabel("Boylam (°D)")
    ax.set_ylabel("Enlem (°K)")
    ax.tick_params(labelsize=7)
    return fig, ax


def map_fig(arr, bbox, title, cmap="viridis", label="", vmin=None, vmax=None, shade=None, under=None):
    fig, ax = _axes(bbox)
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    if arr.ndim == 3:
        ax.imshow(arr, extent=ext, interpolation="nearest")
    else:
        if under is not None:
            ax.imshow(under, extent=ext, cmap="gray", vmin=-1.2, vmax=1.3)
        if vmin is None:
            vmin, vmax = np.nanpercentile(arr, [2, 98])
        im = ax.imshow(arr, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if shade is not None:
            ax.imshow(shade, extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.35)
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_title(title, loc="left")
    return fig, ax


def class_fig(arr, bbox, title, classes: dict):
    """classes = {değer: (ad, renk)} — sınıflı harita + lejant."""
    fig, ax = _axes(bbox, figsize=(5.2, 5.4))
    vals = [v for v in classes if (arr == v).any()]
    lut = np.zeros(256, dtype=int)
    for i, v in enumerate(vals):
        lut[v] = i + 1
    cmap = ListedColormap(["#ffffff"] + [classes[v][1] for v in vals])
    ax.imshow(lut[arr.astype(int) % 256], extent=[bbox[0], bbox[2], bbox[1], bbox[3]], cmap=cmap,
              vmin=0, vmax=max(len(vals), 1), interpolation="nearest")
    ax.legend(handles=[Patch(facecolor=classes[v][1], edgecolor="#555", label=classes[v][0]) for v in vals],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
    ax.set_title(title, loc="left")
    return fig


def mark(ax, df, bbox, color, marker):
    for name, row in df.iterrows():
        ax.plot(row["Boylam"], row["Enlem"], marker, ms=5, mfc="none", mec=color, mew=1.2)
        ax.annotate(name, (row["Boylam"], row["Enlem"]), xytext=(3, 3), textcoords="offset points",
                    fontsize=6.5, color=color, weight="bold")


def section_fig(dem, px, bbox, row, wc=None, lst=None):
    """Batı–doğu şematik kesit: yükseklik profili + yüzey örtüsü şeridi + yaz yüzey sıcaklığı."""
    H, W = dem.shape
    x = (np.arange(W) + 0.5) * px / 1000
    z = dem[row]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 3.6), height_ratios=[8, 1], sharex=True, layout="constrained")
    ax.fill_between(x, z, z.min() - 0.08 * np.ptp(z) - 5, color="#8d7b6c", alpha=0.85, lw=0)
    ax.plot(x, z, color="#3b2f2f", lw=1.2)
    ax.set_ylabel("Yükseklik (m)")
    ax.set_ylim(z.min() - 0.08 * np.ptp(z) - 5, z.max() + 0.25 * np.ptp(z) + 5)
    ve = (np.ptp(x) * 1000 / max(np.ptp(z), 1)) * (3.0 / 9.2)
    lat = bbox[3] - (row + 0.5) / H * (bbox[3] - bbox[1])
    ax.set_title(f"Şematik B–D kesiti  ·  {lat:.3f}°K  ·  düşey abartı ≈ {ve:.0f}×", loc="left")
    if lst is not None and np.isfinite(lst[row]).any():
        a = ax.twinx()
        a.plot(x, lst[row], color="#d62728", lw=1, alpha=0.9)
        a.set_ylabel("Yaz yüzey sıcaklığı (°C)", color="#d62728")
        a.tick_params(axis="y", colors="#d62728")
        a.spines["right"].set_visible(True)
    if wc is not None:
        vals = sorted(set(np.unique(wc[row])) & set(WORLDCOVER))
        lut = np.zeros(256, dtype=int)
        for i, v in enumerate(vals):
            lut[v] = i
        ax2.imshow(lut[wc[row]][None, :], extent=[x[0], x[-1], 0, 1], aspect="auto",
                   cmap=ListedColormap([WORLDCOVER[v][1] for v in vals] or ["#fff"]), vmin=0, vmax=max(len(vals) - 1, 1),
                   interpolation="nearest")
        ax2.legend(handles=[Patch(facecolor=WORLDCOVER[v][1], edgecolor="#555", label=WORLDCOVER[v][0]) for v in vals],
                   loc="upper center", bbox_to_anchor=(0.5, -1.6), ncol=len(vals), fontsize=7, frameon=False)
    ax2.set_yticks([])
    ax2.set_ylabel("Örtü", rotation=0, ha="right", va="center")
    ax2.set_xlabel("Batıdan uzaklık (km)")
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# SEKMELER
# --------------------------------------------------------------------------------------
def tab_geology(bbox, n, year, wc):
    st.caption("Planetary Computer'da hazır bir jeoloji / mineral haritası yoktur. Buradaki ürünler uydu "
               "spektrumundan türetilmiş **dolaylı göstergelerdir**; kesin litoloji için saha ve laboratuvar verisi gerekir.")
    try:
        s2 = get_s2(bbox, n, year)
    except Exception as e:
        st.warning(f"Sentinel-2 verisi alınamadı: {e}")
        return None
    R = s2["refl"]
    ndvi, rock, water, veg, bad, idx = spectral_products(s2)
    st.markdown(f"**Sentinel-2 L2A** · sahne tarihi: {s2['tarih']} · sahne bulutluluğu: %{s2['bulut']:.1f} · "
                f"çıplak yüzey oranı: %{100 * rock.mean():.0f}")

    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(rgb(R, ["B04", "B03", "B02"]), bbox, "Doğal renk (B04-B03-B02)")[0])
    with c2:
        show(map_fig(rgb(R, ["B12", "B8A", "B03"]), bbox, "Jeoloji bileşimi (B12-B8A-B03)")[0])
    st.caption("Jeoloji bileşiminde bitki yeşil, çıplak kaya/toprak pembe-kahve tonlarında, taze koyu lavlar çok koyu, "
               "su siyah-lacivert görünür.")

    st.subheader("Mineral gösterge indeksleri")
    grey = stretch(R["B04"] + R["B03"] + R["B02"])
    cols = st.columns(3)
    for col, (name, arr), cmap in zip(cols, idx.items(), ["YlOrRd", "PuBuGn", "YlGnBu"]):
        with col:
            a = np.where(rock, arr, np.nan)
            show(map_fig(a, bbox, name, cmap=cmap, label="oran", under=grey)[0])
    st.caption("İndeksler yalnızca çıplak yüzeyde (NDVI < 0,30; su, bulut ve gölge hariç) gösterilir; gri alanlar maskelenmiştir. "
               "Yüksek B04/B02: hematit-götit gibi ferrik oksitler · yüksek ferröz indeks: olivin-piroksen gibi Fe²⁺ taşıyan "
               "mafik mineraller · yüksek B11/B12: kil, mika, karbonat gibi hidroksilli/ayrışma mineralleri.")

    st.subheader("Yüzey malzemeleri")
    c1, c2 = st.columns(2)
    stack = np.stack([R[b] for b in ["B02", "B03", "B04", "B8A", "B11", "B12"]])
    units, cen = spectral_units(stack, rock) if rock.mean() > 0.02 else (None, None)
    with c1:
        if units is not None:
            cls = {i + 1: (f"Birim {i + 1}", UNIT_COLORS[i]) for i in range(len(cen))}
            cls[100], cls[101] = ("Bitki örtüsü (maskeli)", "#9fcf8f"), ("Su", "#3b7dd8")
            u = (units + 1).astype("uint8")
            u[veg & (units < 0)] = 100
            u[water] = 101
            show(class_fig(u, bbox, "Spektral birimler (çıplak yüzey, k-ortalamalar)", cls))
        else:
            st.info("Çıplak yüzey oranı spektral birim ayrımı için çok düşük.")
    with c2:
        if wc is not None:
            show(class_fig(wc, bbox, "Yüzey örtüsü — ESA WorldCover 2021", WORLDCOVER))

    c1, c2 = st.columns(2)
    with c1:
        if units is not None:
            wl = [490, 560, 665, 865, 1610, 2190]
            fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
            for i, s in enumerate(cen):
                ax.plot(wl, s, "-o", ms=3, color=UNIT_COLORS[i], mec="#444", mew=0.4, label=f"Birim {i + 1}")
            ax.set_facecolor("#e9eef2")
            ax.set_xlabel("Dalga boyu (nm)")
            ax.set_ylabel("Yansıma")
            ax.set_title("Birimlerin ortalama spektrumu", loc="left")
            ax.legend(fontsize=7, frameon=False, ncol=2)
            show(fig)
            share = [(units == i).sum() / max((units >= 0).sum(), 1) for i in range(len(cen))]
            st.dataframe(pd.DataFrame({
                "Çıplak yüzeydeki pay (%)": [round(100 * s, 1) for s in share],
                "Ort. yansıma": [round(float(s.mean()), 3) for s in cen],
                "B04/B02": [round(float(s[2] / s[0]), 2) for s in cen],
                "B11/B12": [round(float(s[4] / s[5]), 2) for s in cen],
                "Olası yorum": [unit_hint(s) for s in cen],
            }, index=[f"Birim {i + 1}" for i in range(len(cen))]))
    with c2:
        if wc is not None:
            vals, cnt = np.unique(wc[wc > 0], return_counts=True)
            order = np.argsort(cnt)
            fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
            ax.barh([WORLDCOVER.get(int(v), (str(v),))[0] for v in vals[order]], 100 * cnt[order] / cnt.sum(),
                    color=[WORLDCOVER.get(int(v), ("", "#999"))[1] for v in vals[order]], edgecolor="#555", lw=0.5)
            ax.set_xlabel("Alan payı (%)")
            ax.set_title("Yüzey örtüsü dağılımı", loc="left")
            show(fig)
    mineral_extra(bbox, n, R, rock, units, idx)
    return {"rock": rock, "units": units}


def tab_topo(bbox, dem, px, wc, lst):
    slope, normals = slope_normals(dem, px)
    tri = roughness_tri(dem)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    with st.spinner("Kapalı çöküntüler aranıyor…"):
        depth, lab, dep = find_depressions(dem, px, bbox)
    tpi, peaks = find_peaks(dem, px, bbox)

    m = st.columns(5)
    m[0].metric("Yükseklik aralığı", f"{dem.min():.0f} – {dem.max():.0f} m")
    m[1].metric("Ortalama yükseklik", f"{dem.mean():.0f} m")
    m[2].metric("Ortalama eğim", f"{slope.mean():.1f}°")
    m[3].metric("Dik alan (eğim > 15°)", f"%{100 * (slope > 15).mean():.1f}")
    m[4].metric("Ort. pürüzlülük (TRI)", f"{tri.mean():.1f} m")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = map_fig(dem, bbox, f"Yükseklik ({BODY['dem_label']})", cmap=ELEV_CMAP, label="m",
                          vmin=float(dem.min()), vmax=float(dem.max()), shade=shade)
        if len(peaks):
            mark(ax, peaks, bbox, "#b00020", "^")
        if len(dep):
            mark(ax, dep, bbox, "#0b3d91", "o")
        show(fig)
        st.caption("▲ T: belirgin tepeler (koni / dom adayı) · ○ Ç: kapalı çöküntüler (krater / maar adayı)")
    with c2:
        show(map_fig(slope, bbox, "Eğim", cmap="magma_r", label="derece", vmin=0,
                     vmax=float(max(10, np.percentile(slope, 99))))[0])
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(tri, bbox, f"Pürüzlülük — TRI ({px:.0f} m piksel)", cmap="cividis", label="m", vmin=0,
                     vmax=float(max(2, np.percentile(tri, 99))))[0])
    with c2:
        d = np.where(depth > 1, depth, np.nan)
        fig, ax = map_fig(d, bbox, "Kapalı çöküntü derinliği", cmap="Blues", label="m", vmin=0,
                          vmax=float(max(5, np.nanmax(depth))), under=shade)
        if len(dep):
            mark(ax, dep, bbox, "#0b3d91", "o")
        show(fig)

    st.subheader("Şematik kesit")
    H = dem.shape[0]
    if len(dep) and dep["Derinlik (m)"].iloc[0] >= 20:
        default = int(dep["_r"].iloc[0])
    elif len(peaks):
        default = int(peaks["_r"].iloc[0])
    else:
        default = H // 2
    pos = st.slider("Kesit enlemi (kuzey → güney, %)", 0, 100, int(round(100 * default / (H - 1))),
                    help="Varsayılan konum en derin çöküntüden ya da en belirgin tepeden geçer.")
    row = min(H - 1, int(round(pos / 100 * (H - 1))))
    show(section_fig(dem, px, bbox, row, wc, lst))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Kapalı çöküntüler (krater / maar adayları)**")
        if len(dep):
            st.dataframe(dep.drop(columns=["_r", "_c", "_lab"]))
        else:
            st.info("En az 5 m derinlikte ve 0,01 km² alanda kapalı çöküntü bulunamadı.")
        st.caption("Çukurlar, DEM'in öncelikli-taşkın yöntemiyle doldurulup özgün yüzeyden çıkarılmasıyla bulunur. "
                   "Kenarı yarılmış kraterler ve çerçeve kenarına açılan çukurlar yakalanmaz; göl yüzeyleri taban sayılır.")
    with c2:
        st.markdown("**Belirgin tepeler (koni / dom adayları)**")
        if len(peaks):
            st.dataframe(peaks.drop(columns=["_r", "_c"]))
        else:
            st.info("Çevresinden en az 30 m yüksek belirgin tepe bulunamadı.")
        st.caption("'Çevreden yükseklik' = zirve ile 1 km yarıçaplı çevrenin ortalaması arasındaki fark. "
                   "Liste volkanik olmayan tepeleri de içerebilir.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3), layout="constrained")
        ax.hist(dem.ravel(), bins=50, color="#8d7b6c", edgecolor="white", lw=0.3,
                weights=np.full(dem.size, 100 / dem.size))
        ax.set_xlabel("Yükseklik (m)")
        ax.set_ylabel("Alan payı (%)")
        ax.set_title("Yükseklik dağılımı", loc="left")
        show(fig)
    with c2:
        edges = [0, 2, 5, 10, 15, 25, 90]
        names = ["0–2° düz", "2–5° hafif", "5–10° orta", "10–15° eğimli", "15–25° dik", ">25° çok dik"]
        share = np.histogram(slope, bins=edges)[0] / slope.size * 100
        fig, ax = plt.subplots(figsize=(5.2, 3), layout="constrained")
        ax.bar(names, share, color=plt.cm.magma_r(np.linspace(0.1, 0.85, 6)), edgecolor="#555", lw=0.5)
        ax.set_ylabel("Alan payı (%)")
        ax.set_title("Eğim sınıfları", loc="left")
        ax.tick_params(axis="x", labelsize=7, rotation=20)
        show(fig)
    topo_extra(bbox, dem, px, slope, shade, dep, peaks)


def tab_thermal(bbox, dem, px, year, wc, ls):
    lat = (bbox[1] + bbox[3]) / 2
    st.subheader("Yüzey sıcaklığı")
    try:
        with st.spinner("MODIS yıllık sıcaklık serisi indiriliyor…"):
            ser, myear, nvalid = modis_best_year(bbox, year)
    except Exception as e:
        ser = None
        st.warning(f"MODIS sıcaklık serisi alınamadı: {e}")
    if ser is not None:
        if myear != year:
            st.info(f"{year} yılı için Planetary Computer kataloğunda MODIS dönemlerinin çoğu eksik; "
                    f"yıllık seri için en yakın eksiksiz yıl olan **{myear}** kullanıldı.")
        elif nvalid < 40:
            st.warning(f"{myear} için 46 dönemin yalnızca {nvalid} tanesi mevcut; yıllık uç değerler eksik olabilir.")
        frost = int((ser["gece_min"] < 0).sum())
        m = st.columns(5)
        m[0].metric("En sıcak gündüz (alan ort.)", f"{ser['gunduz'].max():.1f} °C",
                    help=f"{ser.loc[ser['gunduz'].idxmax(), 'tarih']:%d.%m.%Y} ile başlayan 8 gün")
        m[1].metric("En sıcak piksel", f"{ser['gunduz_maks'].max():.1f} °C")
        m[2].metric("En soğuk gece (alan ort.)", f"{ser['gece'].min():.1f} °C",
                    help=f"{ser.loc[ser['gece'].idxmin(), 'tarih']:%d.%m.%Y} ile başlayan 8 gün")
        m[3].metric("En soğuk piksel", f"{ser['gece_min'].min():.1f} °C")
        m[4].metric("En büyük gündüz–gece farkı", f"{ser['genlik'].max():.1f} °C")
        fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
        ax.fill_between(ser["tarih"], ser["gece"], ser["gunduz"], color="#f4a261", alpha=0.25, lw=0,
                        label="Gündüz–gece aralığı")
        ax.plot(ser["tarih"], ser["gunduz"], "-o", ms=2.5, color="#d62728", lw=1.2, label="Gündüz (≈10:30)")
        ax.plot(ser["tarih"], ser["gece"], "-o", ms=2.5, color="#1f4e9c", lw=1.2, label="Gece (≈22:30)")
        ax.plot(ser["tarih"], ser["gunduz_maks"], color="#d62728", lw=0.7, ls=":", label="En sıcak piksel")
        ax.plot(ser["tarih"], ser["gece_min"], color="#1f4e9c", lw=0.7, ls=":", label="En soğuk piksel")
        ax.axhline(0, color="#555", lw=0.6)
        ax.set_ylabel("Yüzey sıcaklığı (°C)")
        ax.set_title(f"{myear} — MODIS/Terra 8 günlük yüzey sıcaklığı (1 km)", loc="left")
        ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        show(fig)
        st.caption(f"Yıllık aralık (en sıcak gündüz − en soğuk gece, alan ortalaması): "
                   f"**{ser['gunduz'].max() - ser['gece'].min():.1f} °C** · gece yüzey sıcaklığının 0 °C altına indiği "
                   f"8 günlük dönem sayısı: **{frost}** / {nvalid}. Değerler hava değil **zemin yüzeyi** sıcaklığıdır ve "
                   "8 günlük açık-gökyüzü ortalamasıdır; tekil günlerdeki uç değerler daha yüksek/düşük olabilir.")

    if ls is not None:
        lst = ls["lst"]
        c1, c2 = st.columns(2)
        with c1:
            sun = f" · güneş yüksekliği {ls['gunes_yuk']:.0f}°" if ls.get("gunes_yuk") else ""
            show(map_fig(lst, bbox, "Yaz yüzey sıcaklığı — Landsat (30 m)", cmap="inferno", label="°C")[0])
            st.caption(f"{ls['uydu']} · {ls['tarih']}{sun} · ortalama {np.nanmean(lst):.1f} °C · "
                       f"%2–%98 aralığı {np.nanpercentile(lst, 2):.1f} – {np.nanpercentile(lst, 98):.1f} °C")
        with c2:
            if wc is not None:
                rows = [(WORLDCOVER[v][0], WORLDCOVER[v][1], np.nanmean(lst[wc == v]))
                        for v in WORLDCOVER if (wc == v).mean() > 0.005 and np.isfinite(lst[wc == v]).any()]
                rows.sort(key=lambda r: r[2])
                fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
                ax.barh([r[0] for r in rows], [r[2] for r in rows], color=[r[1] for r in rows], edgecolor="#555", lw=0.5)
                if rows:
                    ax.set_xlim(min(r[2] for r in rows) - 3, max(r[2] for r in rows) + 2)
                ax.set_xlabel("Ortalama yüzey sıcaklığı (°C)")
                ax.set_title("Yüzey örtüsüne göre yaz sıcaklığı", loc="left")
                show(fig)

    st.subheader("Aydınlanma ve gölge")
    with st.spinner("Güneş geometrisi ve arazi gölgeleri hesaplanıyor…"):
        hs, es, ts = illumination(dem, px, lat, 172)
        hw, ew, tw = illumination(dem, px, lat, 355)
    m = st.columns(5)
    m[0].metric("Öğle güneş yüksekliği · 21 Haziran", f"{sun_position(lat, 172, 12)[0]:.1f}°")
    m[1].metric("Öğle güneş yüksekliği · 21 Aralık", f"{sun_position(lat, 355, 12)[0]:.1f}°")
    m[2].metric("Ort. doğrudan güneş · yaz", f"{hs.mean():.1f} sa")
    m[3].metric("Ort. doğrudan güneş · kış", f"{hw.mean():.1f} sa")
    m[4].metric("Kışın < 4 sa güneş alan", f"%{100 * (hw < 4).mean():.1f}")
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(hw, bbox, "Doğrudan güneş alma süresi — 21 Aralık", cmap="YlOrBr_r", label="saat/gün",
                     vmin=0, vmax=float(hw.max()))[0])
    with c2:
        show(map_fig(ew, bbox, "Göreli güneş ışınımı — 21 Aralık", cmap="RdYlBu_r",
                     label="düz zemine oran", vmin=0, vmax=2)[0])
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(hs, bbox, "Doğrudan güneş alma süresi — 21 Haziran", cmap="YlOrBr_r", label="saat/gün",
                     vmin=float(min(hs.min(), hs.max() - 4)), vmax=float(hs.max()))[0])
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        ax.plot(ts["saat"], ts["golge_%"], color="#e9a23b", lw=1.6, label="21 Haziran")
        ax.plot(tw["saat"], tw["golge_%"], color="#34558b", lw=1.6, label="21 Aralık")
        ax.set_xlabel("Yerel güneş saati")
        ax.set_ylabel("Gölgedeki alan (%)")
        ax.set_title("Gün içinde gölgede kalan alan", loc="left")
        ax.legend(fontsize=8, frameon=False)
        show(fig)
    st.caption("Aydınlanma, DEM üzerinde güneş konumu yarım saatlik adımlarla izlenerek (yamaç yönelimi + komşu arazinin "
               "düşürdüğü gölge) hesaplanır; bulut ve atmosfer etkisi içermez. Göreli ışınım 1,0 = aynı gün düz bir zemin.")


# --------------------------------------------------------------------------------------
# EK VERİ SETLERİ: ASTER (mineraloji), JRC yüzey suyu, MODIS buharlaşma ve kar
# --------------------------------------------------------------------------------------
# ASTER L1T sayısal değer -> radyans katsayıları (normal kazanç) ve güneş ışınımı (ESUN)
ASTER_UCC = {"VNIR": [1.688, 1.415, 0.862], "SWIR": [0.2174, 0.0696, 0.0625, 0.0597, 0.0417, 0.0318],
             "TIR": [0.006822, 0.006780, 0.006590, 0.005693, 0.005225]}
ASTER_ESUN = {"VNIR": [1848.0, 1549.0, 1114.0], "SWIR": [225.4, 86.63, 81.85, 74.85, 66.49, 59.85]}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_aster(bbox: tuple, n: int):
    """ASTER L1T (2000–2008; SWIR dedektörü 2008'de devre dışı kaldı). Dönen: bant sözlüğü B1..B14."""
    W, H, _ = grid_shape(bbox, n)
    items = stac_search({
        "collections": ["aster-l1t"], "bbox": list(bbox),
        "datetime": "2000-03-01T00:00:00Z/2008-03-31T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": 10}}, "limit": 100,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }, max_items=100)
    items = [i for i in items if all(a in i["assets"] for a in ("VNIR", "SWIR", "TIR"))]
    dry = [i for i in items if 5 <= int(i["properties"]["datetime"][5:7]) <= 10]  # kurak dönem: az bitki, kar yok
    items = dry or items
    if not items:
        raise RuntimeError("Bu alan için uygun (VNIR+SWIR+TIR, az bulutlu) ASTER sahnesi yok.")
    scenes = _best_scenes("aster-l1t", items, "SWIR", bbox, max_try=14)
    if not scenes:
        raise RuntimeError("ASTER sahneleri bu alanı kaplamıyor.")
    out, used = None, []
    for it in scenes[:3]:
        parts, ok_all = [], np.ones((H, W), bool)
        for sub in ("VNIR", "SWIR", "TIR"):
            d, ok = crop("aster-l1t", it["id"], [sub], bbox, W, H)
            ok &= (d > 0).all(axis=0)
            rad = (d - 1.0) * np.array(ASTER_UCC[sub], dtype="float32")[:, None, None]
            if sub in ASTER_ESUN:  # yaklaşık atmosfer-üstü yansıma (oranlarda güneş açısı sadeleşir)
                rad = rad / np.array(ASTER_ESUN[sub], dtype="float32")[:, None, None]
            parts.append(rad)
            ok_all &= ok
        ok_all = ndi.binary_erosion(ok_all, iterations=3)  # sahne kenarındaki karışık pikselleri at
        stack = np.where(ok_all, np.concatenate(parts), np.nan)
        if out is None:
            out = stack
        else:
            both = ok_all & np.isfinite(out[0])
            if both.sum() > 500:  # farklı tarih/kazançlı sahneyi ortak alandaki medyana göre ilkine eşitle
                stack *= np.nanmedian(out[:, both] / stack[:, both], axis=1)[:, None, None]
            take = ok_all & np.isnan(out[0])
            out[:, take] = stack[:, take]
        used.append(it["properties"]["datetime"][:10])
        if np.isfinite(out[0]).mean() > 0.97:
            break
    out = np.clip(out, 1e-6, None)
    return {"B": {f"B{i + 1}": out[i] for i in range(14)}, "tarih": ", ".join(sorted(set(used))),
            "kaplama": float(np.isfinite(out[0]).mean())}


def aster_indices(B):
    return {
        "AlOH grubu  (B5+B7)/B6": (B["B5"] + B["B7"]) / B["B6"],
        "Kaolin grubu  B6/B5": B["B6"] / B["B5"],
        "MgOH / karbonat  (B6+B9)/(B7+B8)": (B["B6"] + B["B9"]) / (B["B7"] + B["B8"]),
        "FeOH grubu  (B6+B8)/B7": (B["B6"] + B["B8"]) / B["B7"],
        "Ferrik oksit  B2/B1": B["B2"] / B["B1"],
        "Ferröz demir  B5/B4": B["B5"] / B["B4"],
        "Kuvars indeksi  B11²/(B10·B12)": B["B11"] ** 2 / (B["B10"] * B["B12"]),
        "Karbonat indeksi  B13/B14": B["B13"] / B["B14"],
        "Mafik indeks  B12/B13": B["B12"] / B["B13"],
    }


def s2_indices_extra(R):
    return {
        "Demir oksit geneli  (B11/B8A)": R["B11"] / R["B8A"],
        "Gossan / Fe-hidroksit  (B11/B04)": R["B11"] / R["B04"],
        "Ferröz silikat  (B12/B11)": R["B12"] / R["B11"],
    }


JRC_TRANS = {1: ("Kalıcı su", "#0000ff"), 2: ("Yeni kalıcı", "#22b14c"), 3: ("Kaybolan kalıcı", "#d1102d"),
             4: ("Mevsimlik", "#99d9ea"), 5: ("Yeni mevsimlik", "#b5e61d"), 6: ("Kaybolan mevsimlik", "#e6a1aa"),
             7: ("Mevsimlikten kalıcıya", "#ff7f27"), 8: ("Kalıcıdan mevsimliğe", "#ffc90e"),
             9: ("Geçici kalıcı", "#7f7f7f"), 10: ("Geçici mevsimlik", "#c3c3c3")}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_jrc(bbox: tuple, n: int):
    """JRC Global Surface Water 1984–2020: su görülme sıklığı (%) ve değişim sınıfları."""
    W, H, _ = grid_shape(bbox, n)
    ids = [i["id"] for i in stac_search({"collections": ["jrc-gsw"], "bbox": list(bbox), "limit": 10})]
    if not ids:
        raise RuntimeError("JRC yüzey suyu verisi bulunamadı.")
    occ, _ = mosaic("jrc-gsw", ids, ["occurrence"], bbox, W, H, smooth=False)
    tr, _ = mosaic("jrc-gsw", ids, ["transitions"], bbox, W, H, smooth=False)
    occ = np.nan_to_num(occ[0])
    occ[occ > 100] = 0
    return occ.astype("float32"), np.nan_to_num(tr[0]).astype("uint8")


def _km_grid(bbox, km):
    w = (bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    h = (bbox[3] - bbox[1]) * 110.57
    return max(8, round(w / km)), max(8, round(h / km))


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_modis_et(bbox: tuple, year: int):
    """MODIS MOD16A3GF yıllık gerçek (ET) ve potansiyel (PET) buharlaşma-terleme [mm/yıl], 500 m."""
    items = stac_search({"collections": ["modis-16A3GF-061"], "bbox": list(bbox),
                         "datetime": f"{year - 6}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 100})
    items = [i for i in items if i["id"].startswith("MOD16A3GF")]
    if not items:
        raise RuntimeError("MODIS buharlaşma verisi bulunamadı.")
    W, H = _km_grid(bbox, 0.5)

    def one(it):
        try:
            d, ok = crop("modis-16A3GF-061", it["id"], ["ET_500m", "PET_500m"], bbox, W, H, smooth=False)
        except Exception:
            return None
        et = np.where(ok & (d[0] < 65500), d[0] * 0.1, np.nan)
        pet = np.where(ok & (d[1] < 65500), d[1] * 0.1, np.nan)
        return int(it["id"].split(".")[1][1:5]), et, pet

    with ThreadPoolExecutor(6) as ex:
        res = sorted([r for r in ex.map(one, items) if r and np.isfinite(r[1]).any()], key=lambda r: r[0])
    if not res:
        raise RuntimeError("MODIS buharlaşma verisi bu alanda boş (çıplak/yapay yüzeyler ürün dışında tutulur).")
    df = pd.DataFrame({"yil": [r[0] for r in res], "ET": [float(np.nanmean(r[1])) for r in res],
                       "PET": [float(np.nanmean(r[2])) for r in res]}).groupby("yil", as_index=False).mean()
    return df, res[-1][1], res[-1][0]


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_snow_series(bbox: tuple, year: int) -> pd.DataFrame:
    """MODIS MOD10A2 8 günlük en geniş kar örtüsü: karla kaplı alan oranı (%)."""
    items = stac_search({"collections": ["modis-10A2-061"], "bbox": list(bbox),
                         "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 250})
    items = [i for i in items if i["id"].startswith("MOD10A2")]
    if not items:
        raise RuntimeError(f"{year} için MODIS kar verisi bulunamadı.")
    W, H = _km_grid(bbox, 0.5)

    def one(it):
        try:
            d, ok = crop("modis-10A2-061", it["id"], ["Maximum_Snow_Extent"], bbox, W, H, smooth=False)
        except Exception:
            return None
        v = d[0]
        clear = ok & np.isin(v, [25, 37, 200])  # kar yok, göl, kar
        if clear.mean() < 0.3:
            return None
        doy = int(it["id"].split(".")[1][5:8])
        return {"tarih": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1),
                "kar_%": 100 * float((v == 200)[clear].mean())}

    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, items) if r]
    if not rows:
        raise RuntimeError(f"{year} için kullanılabilir MODIS kar verisi yok.")
    df = pd.DataFrame(rows).groupby("tarih").mean()
    return df.reindex(pd.date_range(f"{year}-01-01", periods=46, freq="8D")).rename_axis("tarih").reset_index()


# --------------------------------------------------------------------------------------
# HİDROLOJİ ANALİZİ (DEM üzerinden akış yönü, birikim, akarsu ağı, havzalar)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def flow_routing(dem, px, thr_km2, sinks):
    """Öncelikli-taşkın + D8 ile akış yönlendirme.

    Küçük çukurlar doldurulur; `sinks` ile verilen belirgin kapalı çöküntülerin (krater, maar) tabanı ise
    akışın sonlandığı iç drenaj noktası sayılır. Her hücre en dik komşusuna, düzlüklerde taşma noktasına akar.
    Dönen sözlük: birikim alanı [km²], Strahler derecesi, havza kimliği, kapalı havza kimliği, ana akarsu profili.
    """
    H, W = dem.shape
    N = H * W
    z = dem.ravel().astype("float64").tolist()
    done = bytearray(N)
    parent = [-1] * N
    pop_order = [0] * N
    heap = []
    for r in range(H):
        for c in ((0, W - 1) if 0 < r < H - 1 else range(W)):
            i = r * W + c
            done[i] = 1
            heap.append((z[i], i))
    for i in sinks:
        if not done[i]:
            done[i] = 1
            heap.append((z[i], i))
    sink_set = set(sinks)
    heapq.heapify(heap)
    pop, push = heapq.heappop, heapq.heappush
    k = 0
    while heap:
        h, i = pop(heap)
        pop_order[i] = k
        k += 1
        r, c = divmod(i, W)
        for dr in (-1, 0, 1):
            rr = r + dr
            if rr < 0 or rr >= H:
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if cc < 0 or cc >= W:
                    continue
                j = rr * W + cc
                if done[j]:
                    continue
                done[j] = 1
                parent[j] = i
                if z[j] < h:
                    z[j] = h
                push(heap, (z[j], j))
    filled = np.asarray(z).reshape(H, W)

    # doldurulmuş yüzeyde en dik iniş; yoksa (düzlük) taşkın sırasındaki üst hücre
    rec = np.asarray(parent, dtype="int64").reshape(H, W)
    best = np.zeros((H, W))
    pad = np.pad(filled, 1, constant_values=np.inf)
    idx = np.arange(N).reshape(H, W)
    idx_not_sink = ~np.isin(idx, list(sink_set)) if sink_set else None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc):
                continue
            nb = pad[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc]
            drop = (filled - nb) / math.hypot(dr, dc)
            better = drop > best
            if sink_set:  # çöküntü tabanı akışın sonlandığı yerdir
                better &= idx_not_sink
            best[better] = drop[better]
            rec[better] = (idx + dr * W + dc)[better]
    rec_l = rec.ravel().tolist()
    order = np.lexsort((-np.asarray(pop_order), -filled.ravel())).tolist()  # yukarıdan aşağıya

    thr = thr_km2 * 1e6 / (px * px)
    acc = [1.0] * N
    bestdon, bestacc = [-1] * N, [0.0] * N
    smax, scnt, strah = [0] * N, [0] * N, [0] * N
    for i in order:
        a = acc[i]
        r = rec_l[i]
        o = 0
        if a >= thr:
            o = 1 if smax[i] == 0 else (smax[i] + 1 if scnt[i] >= 2 else smax[i])
            strah[i] = o
        if r >= 0:
            acc[r] += a
            if a > bestacc[r]:
                bestacc[r], bestdon[r] = a, i
            if o:
                if o > smax[r]:
                    smax[r], scnt[r] = o, 1
                elif o == smax[r]:
                    scnt[r] += 1

    # havzalar (çıkış hücresine göre) ve kapalı çöküntülere boşalan alanlar: aşağıdan yukarıya yayılır
    basin = [0] * N
    closed = [0] * N
    nb_id = 0
    for i in reversed(order):
        r = rec_l[i]
        if r < 0:
            nb_id += 1
            basin[i] = nb_id
            closed[i] = 1 if i in sink_set else 0
        else:
            basin[i] = basin[r]
            closed[i] = closed[r]

    acc_a = np.asarray(acc).reshape(H, W)
    strah_a = np.asarray(strah, dtype="int16").reshape(H, W)
    # akarsu uzunluğu: her akarsu hücresinden alıcısına olan adım
    ii = np.nonzero(strah_a.ravel() > 0)[0]
    rr = rec.ravel()[ii]
    step = np.where(rr < 0, 0.0, np.where(np.isin(np.abs(ii - rr), (1, W)), px, px * math.sqrt(2)))
    so = strah_a.ravel()[ii]
    length_km = {int(o): float(step[so == o].sum() / 1000) for o in np.unique(so)}

    # ana akarsu: en büyük birikimli çıkıştan kaynağa doğru, en büyük kolu izleyerek
    path = [int(np.argmax(acc_a))]
    while bestdon[path[-1]] >= 0 and len(path) < N:
        path.append(bestdon[path[-1]])
    path = np.asarray(path[::-1])
    pr, pc = np.divmod(path, W)
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(pr), np.diff(pc)) * px)]) / 1000
    return {"acc_km2": acc_a * px * px / 1e6, "strahler": strah_a, "basin": np.asarray(basin).reshape(H, W),
            "closed": np.asarray(closed).reshape(H, W), "length_km": length_km,
            "main": {"dist": dist, "z": dem[pr, pc], "r": pr, "c": pc}, "thr_km2": thr_km2}


# --------------------------------------------------------------------------------------
# EK ÇİZİM YARDIMCILARI
# --------------------------------------------------------------------------------------
def hist_grid(named: dict, mask, color="#7a6a8c", ncols=3):
    """Birden çok indeksin dağılımını küçük panellerde gösterir."""
    n = len(named)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 2.3 * nrows), layout="constrained", squeeze=False)
    for ax, (name, arr) in zip(axs.ravel(), named.items()):
        v = arr[mask & np.isfinite(arr)]
        if v.size:
            lo, hi = np.percentile(v, [0.5, 99.5])
            ax.hist(v, bins=60, range=(lo, hi), color=color, edgecolor="none", weights=np.full(v.size, 100 / v.size))
            ax.axvline(np.median(v), color="#222", lw=0.9, ls="--")
        ax.set_title(name, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_ylabel("Alan %", fontsize=7)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    return fig


def unit_heatmap(named: dict, units, k):
    """Spektral birim × indeks: birim ortalamasının tüm çıplak yüzeye göre standart sapma cinsinden farkı."""
    names = list(named)
    M = np.full((k, len(names)), np.nan)
    for j, nm in enumerate(names):
        a = named[nm]
        base = a[(units >= 0) & np.isfinite(a)]
        if base.size < 100:
            continue
        mu, sd = base.mean(), base.std() + 1e-9
        for u in range(k):
            v = a[(units == u) & np.isfinite(a)]
            if v.size > 20:
                M[u, j] = (v.mean() - mu) / sd
    fig, ax = plt.subplots(figsize=(10, 0.55 * k + 2.4), layout="constrained")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(names)), [n.split("  ")[0] for n in names], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(k), [f"Birim {u + 1}" for u in range(k)])
    for u in range(k):
        for j in range(len(names)):
            if np.isfinite(M[u, j]):
                ax.text(j, u, f"{M[u, j]:+.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(M[u, j]) > 0.9 else "#222")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.9, label="ortalamadan fark (σ)")
    ax.set_title("Spektral birimlerin mineral gösterge profili", loc="left")
    return fig


# --------------------------------------------------------------------------------------
# EK BÖLÜMLER: mineraloji, topoğrafya
# --------------------------------------------------------------------------------------
def mineral_extra(bbox, n, R, rock, units, idx):
    st.subheader("Genişletilmiş mineraloji — Sentinel-2")
    grey = stretch(R["B04"] + R["B03"] + R["B02"])
    extra = s2_indices_extra(R)
    cols = st.columns(3)
    for col, (name, arr), cmap in zip(cols, extra.items(), ["OrRd", "YlOrBr", "BuPu"]):
        with col:
            show(map_fig(np.where(rock, arr, np.nan), bbox, name, cmap=cmap, label="oran", under=grey)[0])
    all_s2 = {**idx, **extra}
    show(hist_grid(all_s2, rock, color="#b5651d"))
    st.caption("Histogramlar çıplak yüzeydeki piksellerin dağılımıdır; kesikli çizgi medyanı gösterir. "
               "Sağa uzun kuyruk, o mineral grubunun alanın küçük bir kısmında yoğunlaştığına işaret eder.")

    st.subheader("ASTER mineral indeksleri (SWIR + termal)")
    st.caption("ASTER, Sentinel-2'de bulunmayan dar SWIR bantları ve termal bantlarıyla kil türlerini, karbonatı, "
               "kuvarsı ve mafik kayaları ayırabilen tek açık uydu verisidir. Planetary Computer'daki ürün (L1T) "
               "atmosfer düzeltmesi yapılmamış sensör radyansıdır ve 2000–2008 arasına aittir; değerler mutlak "
               "bolluk değil, **alan içi göreli fark** olarak okunmalıdır. Alan tek sahneye sığmadığında farklı tarihli "
               "sahneler birleştirilir; birleşme çizgisinde ince bir basamak, termal bantlarda yatay şerit izi görülebilir.")
    aster = None
    try:
        with st.spinner("ASTER sahneleri indiriliyor…"):
            aster = get_aster(bbox, n)
    except Exception as e:
        st.info(f"ASTER bölümü atlandı: {e}")
    a_idx = {}
    if aster is not None:
        a_idx = aster_indices(aster["B"])
        ok = rock & np.isfinite(aster["B"]["B1"])
        st.markdown(f"**ASTER L1T** · sahne tarihleri: {aster['tarih']} · alan kaplaması: %{100 * aster['kaplama']:.0f}")
        cmaps = ["YlGnBu", "PuRd", "GnBu", "YlOrBr", "OrRd", "PuBuGn", "YlOrRd", "BuGn", "RdPu"]
        items = list(a_idx.items())
        for row in range(0, len(items), 3):
            cols = st.columns(3)
            for col, (name, arr), cmap in zip(cols, items[row:row + 3], cmaps[row:row + 3]):
                with col:
                    show(map_fig(np.where(ok, arr, np.nan), bbox, name, cmap=cmap, label="oran", under=grey)[0])
        st.caption("AlOH: muskovit-illit-simektit · Kaolin: kaolinit-dikit-alunit (ileri arjilik alterasyon) · "
                   "MgOH/karbonat: klorit, epidot, amfibol, kalsit-dolomit · FeOH: jarosit, nontronit · "
                   "Kuvars indeksi: yüksek = silisçe zengin (riyolit, kumtaşı, silisleşme) · "
                   "Mafik indeks: yüksek = silisçe fakir (bazalt, ultramafik) · Karbonat: kalsit, dolomit.")
        show(hist_grid(a_idx, ok, color="#5b7c99"))
        c1, c2 = st.columns(2)
        with c1:
            qi, mi = a_idx["Kuvars indeksi  B11²/(B10·B12)"][ok], a_idx["Mafik indeks  B12/B13"][ok]
            if qi.size > 200:
                fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
                hb = ax.hexbin(mi, qi, gridsize=45, cmap="magma_r", mincnt=1, bins="log",
                               extent=(*np.percentile(mi, [0.5, 99.5]), *np.percentile(qi, [0.5, 99.5])))
                fig.colorbar(hb, ax=ax, label="piksel sayısı (log)")
                ax.set_xlabel("Mafik indeks  (→ bazaltik)")
                ax.set_ylabel("Kuvars indeksi  (→ silisik)")
                ax.set_title("Silisik – mafik ayrımı (ASTER termal)", loc="left")
                show(fig)
        with c2:
            al, mg = a_idx["AlOH grubu  (B5+B7)/B6"][ok], a_idx["MgOH / karbonat  (B6+B9)/(B7+B8)"][ok]
            if al.size > 200:
                fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")
                hb = ax.hexbin(mg, al, gridsize=45, cmap="viridis_r", mincnt=1, bins="log",
                               extent=(*np.percentile(mg, [0.5, 99.5]), *np.percentile(al, [0.5, 99.5])))
                fig.colorbar(hb, ax=ax, label="piksel sayısı (log)")
                ax.set_xlabel("MgOH / karbonat  (→ klorit, epidot, kalsit)")
                ax.set_ylabel("AlOH  (→ kil, mika)")
                ax.set_title("Alterasyon türü ayrımı (ASTER SWIR)", loc="left")
                show(fig)
    if units is not None:
        k = int(units.max()) + 1
        a_ok = {nm: np.where(np.isfinite(aster["B"]["B1"]), a, np.nan) for nm, a in a_idx.items()} if aster else {}
        show(unit_heatmap({**all_s2, **a_ok}, units, k))
        st.caption("Her hücre, o spektral birimin ilgili indeksteki ortalamasının tüm çıplak yüzey ortalamasından kaç "
                   "standart sapma saptığını gösterir: kırmızı = birimde belirgin biçimde yüksek, mavi = düşük.")


def curvatures(dem, px):
    """Profil ve plan eğriliği (1/100 m). Negatif profil = dışbükey yamaç, pozitif = içbükey."""
    z = ndi.gaussian_filter(dem, 1.0)
    q, p = np.gradient(z, px)
    q = -q
    r = np.gradient(p, px, axis=1)
    t = -np.gradient(q, px, axis=0)
    s = -np.gradient(p, px, axis=0)
    g2 = p * p + q * q + 1e-12
    prof = (p * p * r + 2 * p * q * s + q * q * t) / (g2 * (1 + g2) ** 1.5)
    plan = (q * q * r - 2 * p * q * s + p * p * t) / (g2 ** 1.5)
    plan[g2 < 1e-4] = 0  # düzlüklerde plan eğriliği tanımsız
    return prof * 100, np.clip(plan, -5, 5) * 100


LANDFORMS = {1: ("Vadi / çukur", "#2c5aa0"), 2: ("Alt yamaç", "#8fb8de"), 3: ("Düzlük", "#eee8c8"),
             4: ("Orta yamaç", "#c9b27c"), 5: ("Üst yamaç", "#c97b4a"), 6: ("Sırt / zirve", "#7a2e1d")}


def landforms(dem, px, slope, radius_m=300):
    """Topoğrafik Konum İndeksi (TPI) ve eğime göre yeryüzü şekli sınıfları."""
    k = max(3, int(round(2 * radius_m / px)) | 1)
    tpi = dem - ndi.uniform_filter(dem, size=k, mode="nearest")
    zt = tpi / (tpi.std() + 1e-9)
    lf = np.full(dem.shape, 3, dtype="uint8")
    lf[(np.abs(zt) <= 0.5) & (slope > 5)] = 4
    lf[(zt > -1) & (zt < -0.5)] = 2
    lf[zt <= -1] = 1
    lf[(zt > 0.5) & (zt < 1)] = 5
    lf[zt >= 1] = 6
    return lf


def topo_extra(bbox, dem, px, slope, shade, dep, peaks):
    st.subheader("Genişletilmiş topoğrafya")
    g_s, g_e = np.gradient(dem, px)
    aspect = (np.degrees(np.arctan2(-g_e, g_s)) + 360) % 360  # yamacın baktığı yön, kuzeyden saat yönünde
    aspect_m = np.where(slope >= 2, aspect, np.nan)
    prof, plan = curvatures(dem, px)
    lf = landforms(dem, px, slope)
    win = max(3, int(round(1000 / px)) | 1)
    relief = ndi.maximum_filter(dem, size=win) - ndi.minimum_filter(dem, size=win)
    # hipsometrik integral: (ortalama − min) / (maks − min)
    hi = float((dem.mean() - dem.min()) / max(dem.max() - dem.min(), 1e-6))
    dirs = ["K", "KD", "D", "GD", "G", "GB", "B", "KB"]
    sector = (np.round(aspect_m[np.isfinite(aspect_m)] / 45).astype(int)) % 8
    share = np.bincount(sector, minlength=8) / max(sector.size, 1) * 100

    m = st.columns(5)
    m[0].metric("Hipsometrik integral", f"{hi:.2f}",
                help="0'a yakın: aşınmış / düzleşmiş; 1'e yakın: genç, az aşınmış kütle. Tek volkan konisi ≈ 0,3–0,5.")
    m[1].metric("Ort. yerel rölyef (1 km)", f"{relief.mean():.0f} m")
    m[2].metric("Baskın bakı", dirs[int(np.argmax(share))] if sector.size else "—")
    m[3].metric("İçbükey yamaç payı", f"%{100 * (prof > 0.02).mean():.0f}")
    m[4].metric("Dışbükey yamaç payı", f"%{100 * (prof < -0.02).mean():.0f}")

    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(aspect_m, bbox, "Bakı (yamaç yönü)", cmap="twilight_shifted", label="derece (0 = K, 90 = D)",
                     vmin=0, vmax=360, under=shade)[0])
    with c2:
        fig = plt.figure(figsize=(5.2, 4.6), layout="constrained")
        ax = fig.add_subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.bar(np.radians(np.arange(8) * 45), share, width=np.radians(40),
               color=plt.cm.twilight_shifted(np.arange(8) / 8), edgecolor="#444", lw=0.5)
        ax.set_xticks(np.radians(np.arange(8) * 45), dirs)
        ax.tick_params(labelsize=7)
        ax.set_title("Bakı gülü — eğimli alanın yönlere dağılımı (%)", loc="left", pad=14)
        show(fig)
    c1, c2 = st.columns(2)
    lim = float(max(0.05, np.percentile(np.abs(prof), 98)))
    with c1:
        show(map_fig(prof, bbox, "Profil eğriliği (yamaç boyunca)", cmap="RdBu", label="1/100 m  (− dışbükey, + içbükey)",
                     vmin=-lim, vmax=lim)[0])
    with c2:
        lim2 = float(max(0.05, np.percentile(np.abs(plan), 95)))
        show(map_fig(plan, bbox, "Plan eğriliği (yamaca dik)", cmap="PuOr", label="1/100 m  (− sırt, + oluk)",
                     vmin=-lim2, vmax=lim2)[0])
    c1, c2 = st.columns(2)
    with c1:
        show(class_fig(lf, bbox, "Yeryüzü şekli sınıfları (TPI, 300 m)", LANDFORMS))
    with c2:
        show(map_fig(relief, bbox, "Yerel rölyef (1 km pencerede en yüksek − en alçak)", cmap="copper_r", label="m",
                     vmin=0, vmax=float(np.percentile(relief, 99)))[0])

    c1, c2 = st.columns(2)
    with c1:
        zs = np.sort(dem.ravel())[::-1]
        a = np.linspace(0, 1, zs.size)
        rel = (zs - zs[-1]) / max(zs[0] - zs[-1], 1e-6)
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.fill_between(a, rel, color="#c9b27c", alpha=0.6, lw=0)
        ax.plot(a, rel, color="#5a4526", lw=1.4)
        ax.plot([0, 1], [1, 0], color="#888", lw=0.7, ls=":")
        ax.set_xlabel("Göreli alan (a / A)")
        ax.set_ylabel("Göreli yükseklik (h / H)")
        ax.set_title(f"Hipsometrik eğri  ·  integral = {hi:.2f}", loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        show(fig)
    with c2:
        vals, cnt = np.unique(lf, return_counts=True)
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.barh([LANDFORMS[int(v)][0] for v in vals], 100 * cnt / cnt.sum(),
                color=[LANDFORMS[int(v)][1] for v in vals], edgecolor="#555", lw=0.5)
        ax.set_xlabel("Alan payı (%)")
        ax.set_title("Yeryüzü şekli dağılımı", loc="left")
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        bins = np.linspace(dem.min(), dem.max(), 21)
        which = np.clip(np.digitize(dem.ravel(), bins) - 1, 0, 19)
        mean_s = np.array([slope.ravel()[which == b].mean() if (which == b).any() else np.nan for b in range(20)])
        p90 = np.array([np.percentile(slope.ravel()[which == b], 90) if (which == b).any() else np.nan for b in range(20)])
        mid = (bins[:-1] + bins[1:]) / 2
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.fill_betweenx(mid, mean_s, p90, color="#d9534f", alpha=0.18, lw=0, label="ortalama – %90")
        ax.plot(mean_s, mid, "-o", ms=3, color="#a12a25", lw=1.3, label="ortalama eğim")
        ax.set_xlabel("Eğim (°)")
        ax.set_ylabel("Yükseklik (m)")
        ax.set_title("Yüksekliğe göre eğim", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        if len(dep):
            ax.scatter(dep["Eşdeğer çap (m)"], dep["Derinlik (m)"], s=38, c="#1f4e9c", edgecolor="white", lw=0.6,
                       label="Çöküntü: derinlik", zorder=3)
            for name, row in dep.iterrows():
                ax.annotate(name, (row["Eşdeğer çap (m)"], row["Derinlik (m)"]), xytext=(4, 3),
                            textcoords="offset points", fontsize=6.5, color="#1f4e9c")
            x = np.array([dep["Eşdeğer çap (m)"].min(), dep["Eşdeğer çap (m)"].max()])
            for ratio, ls_ in ((0.2, "--"), (0.05, ":")):
                ax.plot(x, ratio * x, color="#888", lw=0.7, ls=ls_)
                ax.annotate(f"d/D = {ratio}", (x[-1], ratio * x[-1]), fontsize=6.5, color="#666", ha="right", va="bottom")
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "Kapalı çöküntü bulunamadı", ha="center", va="center", transform=ax.transAxes, color="#777")
        ax.set_xlabel("Eşdeğer çap D (m)")
        ax.set_ylabel("Derinlik d (m)")
        ax.set_title("Çöküntü morfometrisi (derinlik / çap)", loc="left")
        show(fig)
    st.caption("Derinlik/çap oranı taze, yalın çarpma ya da patlama kraterlerinde ≈ 0,2 dolayındadır; dolmuş, aşınmış "
               "ya da göl tabanlı çukurlarda belirgin biçimde küçülür. Maarlarda göl ve çökel dolgusu oranı düşürür.")


# --------------------------------------------------------------------------------------
# HİDROLOJİ SEKMESİ
# --------------------------------------------------------------------------------------
def hydro_dem_section(bbox, dem, px, wc):
    """Yalnızca yükseklik modelinden: akarsu ağı, havzalar, nemlilik indeksi (Dünya ve Mars için ortak)."""
    slope, _ = slope_normals(dem, px)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    depth, lab, dep = find_depressions(dem, px, bbox)
    sinks = tuple(int(r["_r"]) * dem.shape[1] + int(r["_c"]) for _, r in dep.iterrows())
    thr = max(0.3, round(150 * px * px / 1e6, 2))
    with st.spinner("Akış yönleri ve akarsu ağı hesaplanıyor…"):
        fr = flow_routing(dem, px, thr, sinks)
    acc, strah, main = fr["acc_km2"], fr["strahler"], fr["main"]
    area_km2 = dem.size * px * px / 1e6
    total_len = sum(fr["length_km"].values())
    twi = np.log(np.maximum(acc * 1e6 / px, px) / np.maximum(np.tan(np.radians(slope)), 1e-3))
    closed_share = 100 * (fr["closed"] > 0).mean()

    st.subheader("Drenaj ağı ve havzalar (yükseklik modelinden)")
    m = st.columns(5)
    m[0].metric("Drenaj yoğunluğu", f"{total_len / area_km2:.2f} km/km²")
    m[1].metric("En yüksek Strahler derecesi", f"{int(strah.max())}")
    m[2].metric("Ana akarsu uzunluğu", f"{main['dist'][-1]:.1f} km")
    m[3].metric("Kapalı havzaya boşalan alan", f"%{closed_share:.1f}",
                help="Yüzey akışı dışarı çıkmadan bir krater / maar / çöküntüde sonlanan alanın payı.")
    m[4].metric("Ortalama nemlilik indeksi (TWI)", f"{twi.mean():.1f}")

    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = _axes(bbox)
        ax.imshow(shade, extent=ext, cmap="gray", vmin=-0.6, vmax=1.2)
        omax = max(int(strah.max()), 1)
        s = ndi.maximum_filter(strah, size=2 if max(dem.shape) < 500 else 3)
        layer = np.where(s > 0, s, np.nan)
        im = ax.imshow(layer, extent=ext, cmap=ListedColormap(plt.cm.Blues(np.linspace(0.45, 1, omax))),
                       vmin=0.5, vmax=omax + 0.5, interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Strahler derecesi", ticks=range(1, omax + 1))
        lon = bbox[0] + (main["c"] + 0.5) / dem.shape[1] * (bbox[2] - bbox[0])
        lat = bbox[3] - (main["r"] + 0.5) / dem.shape[0] * (bbox[3] - bbox[1])
        ax.plot(lon, lat, color="#e4572e", lw=1.1, label="Ana akarsu")
        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
        ax.set_title(f"Akarsu ağı (eşik: {thr:g} km² katkı alanı)", loc="left")
        show(fig)
    with c2:
        b = fr["basin"]
        ids, cnt = np.unique(b, return_counts=True)
        top = ids[np.argsort(cnt)[::-1][:7]]
        cls = np.full(b.shape, 9, dtype="uint8")
        for i, bid in enumerate(top):
            cls[b == bid] = i + 1
        cls[fr["closed"] > 0] = 8
        pal = ["#f4d6a0", "#c7e0b0", "#f2b8a2", "#d5c6e8", "#b9dbe4", "#f1e39b", "#e3c0cf"]
        legend = {i + 1: (f"Havza {i + 1}  ({cnt[ids == bid][0] * px * px / 1e6:.0f} km²)", pal[i])
                  for i, bid in enumerate(top)}
        legend[8] = (f"Kapalı havza — iç drenaj  ({(fr['closed'] > 0).sum() * px * px / 1e6:.0f} km²)", "#2b6cb0")
        legend[9] = ("Diğer küçük havzalar", "#e9e9e9")
        show(class_fig(cls, bbox, "Drenaj havzaları", legend))
    c1, c2 = st.columns(2)
    with c1:
        a = np.where(acc >= 0.02, np.log10(acc), np.nan)
        show(map_fig(a, bbox, "Akış birikimi (katkı alanı ≥ 0,02 km²)", cmap="GnBu", label="log₁₀ km²",
                     vmin=math.log10(0.02), vmax=float(np.log10(acc.max())), under=shade)[0])
    with c2:
        show(map_fig(twi, bbox, "Topoğrafik nemlilik indeksi (TWI)", cmap="YlGnBu", label="ln(a / tan β)")[0])
    st.caption("Akış her hücreden en dik komşusuna yönlendirilir (D8). 5 m'den sığ çukurlar doldurulur; daha derin kapalı "
               "çöküntüler (krater, maar) akışın sonlandığı iç drenaj havzası sayılır. Havzalar çıkış noktasına göre ayrılır; kutu dışına taşan havzaların alanı eksik görünür. "
               "TWI yüksek yerler suyun toplandığı, doygunluğa yatkın tabanlardır. Bazalt platolarında ve genç lavlarda "
               "gerçek akış büyük oranda çatlak ve tüplerden yeraltına sızar; bu ağ **yüzey topoğrafyasının öngördüğü** ağdır.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.fill_between(main["dist"], main["z"], main["z"].min() - 5, color="#b9dbe4", alpha=0.7, lw=0)
        ax.plot(main["dist"], main["z"], color="#1f4e9c", lw=1.4)
        ax.set_xlabel("Kaynaktan uzaklık (km)")
        ax.set_ylabel("Yükseklik (m)")
        grad = (main["z"][0] - main["z"][-1]) / max(main["dist"][-1] * 1000, 1) * 100
        ax.set_title(f"Ana akarsu boyuna profili  ·  ort. eğim %{grad:.1f}", loc="left")
        show(fig)
    with c2:
        orders = sorted(fr["length_km"])
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.bar([str(o) for o in orders], [fr["length_km"][o] for o in orders],
               color=plt.cm.Blues(np.linspace(0.45, 1, max(len(orders), 1))), edgecolor="#555", lw=0.5)
        for i, o in enumerate(orders):
            ax.text(i, fr["length_km"][o], f"{fr['length_km'][o]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xlabel("Strahler derecesi")
        ax.set_ylabel("Toplam uzunluk (km)")
        ax.set_title("Akarsu uzunluğunun derecelere dağılımı", loc="left")
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
        ax.hist(twi.ravel(), bins=60, color="#4f9bb5", edgecolor="none", weights=np.full(twi.size, 100 / twi.size))
        ax.axvline(np.percentile(twi, 90), color="#222", lw=0.9, ls="--")
        ax.set_xlabel("TWI")
        ax.set_ylabel("Alan payı (%)")
        ax.set_title("Nemlilik indeksi dağılımı (kesikli: en ıslak %10 eşiği)", loc="left")
        show(fig)
    with c2:
        if wc is not None:
            rows = [(WORLDCOVER[v][0], WORLDCOVER[v][1], float(twi[wc == v].mean()))
                    for v in WORLDCOVER if (wc == v).mean() > 0.005]
            rows.sort(key=lambda r: r[2])
            fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
            ax.barh([r[0] for r in rows], [r[2] for r in rows], color=[r[1] for r in rows], edgecolor="#555", lw=0.5)
            if rows:
                ax.set_xlim(min(r[2] for r in rows) - 0.5, max(r[2] for r in rows) + 0.3)
            ax.set_xlabel("Ortalama TWI")
            ax.set_title("Yüzey örtüsüne göre nemlilik indeksi", loc="left")
            show(fig)



def tab_hydro(bbox, n, year, dem, px, wc):
    hydro_dem_section(bbox, dem, px, wc)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    # ---------------------------------------------------------------- yüzey suyu
    st.subheader("Yüzey suyu ve nem (uydu gözlemi)")
    s2 = None
    try:
        s2 = get_s2(bbox, n, year)
    except Exception as e:
        st.info(f"Sentinel-2 su/nem indeksleri atlandı: {e}")
    occ = tr = None
    try:
        with st.spinner("JRC yüzey suyu verisi indiriliyor…"):
            occ, tr = get_jrc(bbox, n)
    except Exception as e:
        st.info(f"JRC yüzey suyu bölümü atlandı: {e}")
    px_km2 = px * px / 1e6
    if occ is not None:
        ever = float((occ > 0).sum() * px_km2)
        perm = float((occ >= 75).sum() * px_km2)
        lost = float(np.isin(tr, [3, 6]).sum() * px_km2)
        new = float(np.isin(tr, [2, 5]).sum() * px_km2)
        m = st.columns(4)
        m[0].metric("Su görülmüş alan (1984–2020)", f"{ever:.2f} km²")
        m[1].metric("Çoğunlukla sulu (≥ %75)", f"{perm:.2f} km²")
        m[2].metric("Kaybolan su yüzeyi", f"{lost:.2f} km²")
        m[3].metric("Yeni oluşan su yüzeyi", f"{new:.2f} km²")
    c1, c2 = st.columns(2)
    with c1:
        if occ is not None and (occ > 0).any():
            show(map_fig(np.where(occ > 0, occ, np.nan), bbox, "Su görülme sıklığı 1984–2020 (JRC)", cmap="Blues",
                         label="gözlemlerin %'si", vmin=0, vmax=100, under=shade)[0])
        elif occ is not None:
            st.info("JRC kayıtlarına göre bu alanda 1984–2020 arasında açık su yüzeyi gözlenmemiş.")
    with c2:
        if tr is not None and (tr > 0).any():
            show(class_fig(tr, bbox, "Su yüzeyi değişim sınıfları (JRC)", JRC_TRANS))
    if s2 is not None:
        R = s2["refl"]
        c1, c2 = st.columns(2)
        with c1:
            show(map_fig((R["B03"] - R["B11"]) / (R["B03"] + R["B11"]), bbox, f"Su indeksi MNDWI — {s2['tarih']}",
                         cmap="BrBG", label="(B03−B11)/(B03+B11)", vmin=-0.6, vmax=0.6)[0])
        with c2:
            show(map_fig((R["B8A"] - R["B11"]) / (R["B8A"] + R["B11"]), bbox, f"Nem indeksi NDMI — {s2['tarih']}",
                         cmap="RdYlBu", label="(B8A−B11)/(B8A+B11)", vmin=-0.4, vmax=0.4)[0])
        st.caption("MNDWI > 0 açık su yüzeyini, NDMI ise bitki ve toprak nem içeriğini gösterir (mavi = nemli, kırmızı = kuru). "
                   "Görüntü kurak mevsime aittir; sulanan tarım alanları NDMI'da mavi lekeler olarak öne çıkar.")
    c1, c2 = st.columns(2)
    with c1:
        if tr is not None and (tr > 0).any():
            vals, cnt = np.unique(tr[tr > 0], return_counts=True)
            order = np.argsort(cnt)
            fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
            ax.barh([JRC_TRANS.get(int(v), (str(v),))[0] for v in vals[order]], cnt[order] * px_km2,
                    color=[JRC_TRANS.get(int(v), ("", "#999"))[1] for v in vals[order]], edgecolor="#555", lw=0.5)
            ax.set_xlabel("Alan (km²)")
            ax.set_title("Su yüzeyi değişimi — sınıf alanları", loc="left")
            show(fig)
    with c2:
        if occ is not None and (occ > 0).any():
            fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
            edges = [1, 10, 25, 50, 75, 90, 101]
            names = ["%1–10", "%10–25", "%25–50", "%50–75", "%75–90", "%90–100"]
            a = np.histogram(occ[occ > 0], bins=edges)[0] * px_km2
            ax.bar(names, a, color=plt.cm.Blues(np.linspace(0.3, 1, 6)), edgecolor="#555", lw=0.5)
            ax.set_ylabel("Alan (km²)")
            ax.set_xlabel("Su görülme sıklığı")
            ax.set_title("Suyun kalıcılığına göre alan", loc="left")
            ax.tick_params(axis="x", labelsize=7)
            show(fig)

    # ---------------------------------------------------------------- su bütçesi bileşenleri
    st.subheader("Buharlaşma ve kar örtüsü")
    c1, c2 = st.columns(2)
    with c1:
        try:
            with st.spinner("MODIS buharlaşma verisi indiriliyor…"):
                et_df, et_map, et_year = get_modis_et(bbox, year)
            fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
            x = np.arange(len(et_df))
            ax.bar(x - 0.2, et_df["PET"], width=0.4, color="#f0b27a", edgecolor="#555", lw=0.5, label="Potansiyel (PET)")
            ax.bar(x + 0.2, et_df["ET"], width=0.4, color="#4f9bb5", edgecolor="#555", lw=0.5, label="Gerçek (ET)")
            ax.set_xticks(x, et_df["yil"].astype(int))
            ax.set_ylabel("mm / yıl")
            ax.set_title("Yıllık buharlaşma-terleme (MODIS, alan ort.)", loc="left")
            ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
            show(fig)
            last = et_df.iloc[-1]
            st.caption(f"{int(last['yil'])}: gerçek ET **{last['ET']:.0f} mm**, potansiyel **{last['PET']:.0f} mm** · "
                       f"ET/PET = **{last['ET'] / last['PET']:.2f}** (küçük oran = su kısıtlı, kurak ortam). "
                       "Ürün çıplak ve yapay yüzeyleri kapsamaz; ortalama yalnızca bitkili piksellerdendir.")
        except Exception as e:
            et_map = None
            st.info(f"Buharlaşma bölümü atlandı: {e}")
    with c2:
        if et_map is not None and np.isfinite(et_map).any():
            show(map_fig(et_map, bbox, f"Gerçek buharlaşma-terleme {et_year} (500 m)", cmap="YlGn", label="mm / yıl")[0])
    try:
        with st.spinner("MODIS kar örtüsü serisi indiriliyor…"):
            snow, syear = None, year
            for y in (year, year - 1, year - 2):
                try:
                    cand = get_snow_series(bbox, y)
                except Exception:
                    continue
                if snow is None or cand["kar_%"].notna().sum() > snow["kar_%"].notna().sum():
                    snow, syear = cand, y
                if snow["kar_%"].notna().sum() >= 36:
                    break
        if snow is None:
            raise RuntimeError("kullanılabilir kar verisi yok")
        fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
        ax.fill_between(snow["tarih"], snow["kar_%"], color="#9ecae1", alpha=0.8, lw=0, step="mid")
        ax.plot(snow["tarih"], snow["kar_%"], color="#2b6cb0", lw=1, drawstyle="steps-mid")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Karla kaplı alan (%)")
        ax.set_title(f"{syear} — 8 günlük en geniş kar örtüsü (MODIS/Terra, 500 m)", loc="left")
        show(fig)
        nsnow = int((snow["kar_%"] > 50).sum())
        st.caption(f"Alanın yarısından fazlasının karla kaplı olduğu 8 günlük dönem sayısı: **{nsnow}** "
                   f"(≈ {nsnow * 8} gün). Bulutlu dönemler boş bırakılmıştır.")
    except Exception as e:
        st.info(f"Kar örtüsü bölümü atlandı: {e}")
    st.caption("Yağış verileri (ERA5, TerraClimate, GPM) Planetary Computer'da yalnızca toplu Zarr/NetCDF dosyaları olarak "
               "sunulduğundan bu panelin kullandığı kırpma servisiyle alınamıyor; bu yüzden yağış grafiği yoktur.")


# --------------------------------------------------------------------------------------
# EK VERİ: MODIS bitki indeksi serisi ve yıllık net birincil üretim
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_ndvi_series(bbox: tuple, year: int) -> pd.DataFrame:
    """MODIS MOD13Q1 16 günlük NDVI / EVI (250 m): alan ortalaması ve %10–%90 aralığı."""
    items = stac_search({"collections": ["modis-13Q1-061"], "bbox": list(bbox),
                         "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 250})
    items = [i for i in items if i["id"].startswith("MOD13Q1") and i["id"].split(".")[1][1:5] == str(year)]
    if not items:
        raise RuntimeError(f"{year} için MODIS bitki indeksi bulunamadı.")
    W, H = _km_grid(bbox, 0.25)

    def one(it):
        try:
            d, ok = crop("modis-13Q1-061", it["id"], ["250m_16_days_NDVI", "250m_16_days_EVI"], bbox, W, H, smooth=False)
        except Exception:
            return None
        nd = np.where(ok & (d[0] > -2000), d[0] * 1e-4, np.nan)
        ev = np.where(ok & (d[1] > -2000), d[1] * 1e-4, np.nan)
        if np.isfinite(nd).mean() < 0.3:
            return None
        doy = int(it["id"].split(".")[1][5:8])
        return {"tarih": pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1), "ndvi": float(np.nanmean(nd)),
                "evi": float(np.nanmean(ev)), "p10": float(np.nanpercentile(nd, 10)), "p90": float(np.nanpercentile(nd, 90))}

    with ThreadPoolExecutor(8) as ex:
        rows = [r for r in ex.map(one, items) if r]
    if not rows:
        raise RuntimeError(f"{year} için kullanılabilir MODIS bitki indeksi yok.")
    return pd.DataFrame(rows).groupby("tarih", as_index=False).mean().sort_values("tarih")


def ndvi_best_year(bbox, year):
    best = None
    for y in (year, year - 1, year - 2):
        try:
            df = get_ndvi_series(bbox, y)
        except Exception:
            continue
        if len(df) >= 20:
            return df, y
        if best is None or len(df) > len(best[0]):
            best = (df, y)
    if best is None:
        raise RuntimeError("MODIS bitki indeksi serisi alınamadı.")
    return best


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def get_npp(bbox: tuple, year: int) -> pd.DataFrame:
    """MODIS MOD17A3HGF yıllık net birincil üretim [gC/m²/yıl]."""
    items = stac_search({"collections": ["modis-17A3HGF-061"], "bbox": list(bbox),
                         "datetime": f"{year - 6}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", "limit": 100})
    items = [i for i in items if i["id"].startswith("MOD17A3HGF")]
    if not items:
        raise RuntimeError("MODIS net birincil üretim verisi bulunamadı.")
    W, H = _km_grid(bbox, 0.5)
    rows = []
    for it in items:
        try:
            d, ok = crop("modis-17A3HGF-061", it["id"], ["Npp_500m"], bbox, W, H, smooth=False)
        except Exception:
            continue
        v = np.where(ok & (d[0] < 32700), d[0] * 0.1, np.nan)  # 0,0001 kgC/m² -> gC/m²
        if np.isfinite(v).any():
            rows.append({"yil": int(it["id"].split(".")[1][1:5]), "npp": float(np.nanmean(v)),
                         "kapsama": float(np.isfinite(v).mean())})
    if not rows:
        raise RuntimeError("Net birincil üretim bu alanda boş (çıplak yüzeyler ürün dışında).")
    return pd.DataFrame(rows).groupby("yil", as_index=False).mean()


# --------------------------------------------------------------------------------------
# EK ANALİZ: açıklık, erozyon, iniş alanı, en düşük maliyetli güzergâh
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def openness(dem, px, radius_m=1500):
    """Topoğrafik açıklık [°]: 8 yönde ufuk açısının ortalamasının 90°'den farkı. 90 = tamamen açık düzlük."""
    f = max(1, int(math.ceil(max(dem.shape) / 320)))
    d = dem[::f, ::f]
    p = px * f
    kmax = max(2, int(radius_m / p))
    total = np.zeros_like(d)
    for az in range(0, 360, 45):
        z = math.radians(az)
        hor = np.zeros_like(d)
        for k in range(1, kmax + 1):
            dr, dc = -int(round(k * math.cos(z))), int(round(k * math.sin(z)))
            s = _shift(d, dr, dc)
            ang = np.degrees(np.arctan((s - d) / (math.hypot(dr, dc) * p)))
            hor = np.maximum(hor, np.where(np.isfinite(ang), ang, 0))
        total += hor
    out = 90 - total / 8
    return ndi.zoom(out, (dem.shape[0] / d.shape[0], dem.shape[1] / d.shape[1]), order=1)[:dem.shape[0], :dem.shape[1]] \
        if f > 1 else out


def erosion_index(acc_km2, slope, ndvi, px):
    """Göreli erozyon duyarlılığı = LS (yamaç uzunluğu-dikliği, Moore & Burch) × C (bitki örtüsü, NDVI'dan)."""
    a_s = np.maximum(acc_km2 * 1e6 / px, px)
    ls = (np.minimum(a_s, 5000) / 22.13) ** 0.4 * (np.maximum(np.sin(np.radians(slope)), 1e-3) / 0.0896) ** 1.3
    nd = np.clip(np.nan_to_num(ndvi, nan=0.0), 0, 0.95)
    c = np.exp(-2 * nd / (1 - nd))
    spi = a_s * np.tan(np.radians(slope))
    return ls, c, ls * c, spi


def landing_sites(safe, px, bbox, n_sites=5):
    """Güvenli maskede en büyük engelsiz daireleri bulur (birbirine binmeyen ilk n)."""
    dist = ndi.distance_transform_edt(np.pad(safe, 1, constant_values=False))[1:-1, 1:-1] * px  # çerçeve kenarı da engel
    work = dist.copy()
    H, W = safe.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rows = []
    for i in range(n_sites):
        r, c = np.unravel_index(int(np.argmax(work)), work.shape)
        rad = float(work[r, c])
        if rad < 2 * px:
            break
        lon, lat = rc_to_lonlat(r, c, safe.shape, bbox)
        rows.append({"Enlem": round(lat, 4), "Boylam": round(lon, 4), "Engelsiz yarıçap (m)": int(rad),
                     "Alan (km²)": round(math.pi * rad * rad / 1e6, 2), "_r": int(r), "_c": int(c)})
        work[(yy - r) ** 2 + (xx - c) ** 2 <= (1.6 * rad / px) ** 2] = 0
    df = pd.DataFrame(rows)
    if len(df):
        df.index = [f"İ{i + 1}" for i in range(len(df))]
    return dist, df


@st.cache_data(show_spinner=False)
def least_cost_path(slope, blocked, px, src, dst):
    """Eğime dayalı maliyet yüzeyinde iki nokta arasında en düşük maliyetli güzergâh (Dijkstra, 8 komşu)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    f = max(1, int(math.ceil(max(slope.shape) / 420)))
    s = slope[::f, ::f]
    b = blocked[::f, ::f]
    H, W = s.shape
    cost = 1 + (s / 8.0) ** 2  # düzde 1, 8°'de 2, 16°'de 5, 25°'de ≈ 11
    cost = np.where(b | (s > 30), cost + 500, cost)
    idx = np.arange(H * W).reshape(H, W)
    ii, jj, ww = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dc >= 0:
            a, t = idx[:H - dr, :W - dc], idx[dr:, dc:]
        else:
            a, t = idx[:H - dr, 1:], idx[dr:, :W - 1]
        w = (cost.ravel()[a.ravel()] + cost.ravel()[t.ravel()]) / 2 * math.hypot(dr, dc) * px * f
        ii += [a.ravel(), t.ravel()]
        jj += [t.ravel(), a.ravel()]
        ww += [w, w]
    g = coo_matrix((np.concatenate(ww), (np.concatenate(ii), np.concatenate(jj))), shape=(H * W, H * W)).tocsr()
    s_i = (src[0] // f) * W + min(src[1] // f, W - 1)
    d_i = min(dst[0] // f, H - 1) * W + min(dst[1] // f, W - 1)
    dist, pred = dijkstra(g, directed=False, indices=s_i, return_predecessors=True)
    path = [d_i]
    while path[-1] != s_i and pred[path[-1]] >= 0:
        path.append(int(pred[path[-1]]))
    path = np.asarray(path[::-1])
    pr, pc = np.divmod(path, W)
    return pr * f, pc * f, float(dist[d_i])


def share_bar(names, values, colors, title, xlabel="Alan payı (%)", figsize=(5.2, 3.2)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.barh(names, values, color=colors, edgecolor="#555", lw=0.5)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, max(values) * 1.15 if len(values) and max(values) > 0 else 1)
    return fig


def lonlat(df):
    return df["Boylam"].to_numpy(), df["Enlem"].to_numpy()


# --------------------------------------------------------------------------------------
# BÖLÜM 4 (ek): don, buz, permafrost ve kaynak özeti
# --------------------------------------------------------------------------------------
def frost_and_resources(bbox, n, year, dem, px, wc):
    lat = (bbox[1] + bbox[3]) / 2
    st.subheader("Don, buz ve permafrost")
    ser = None
    try:
        ser, myear, _ = modis_best_year(bbox, year)
    except Exception as e:
        st.info(f"MODIS sıcaklık serisi alınamadı: {e}")
    hw, _, _ = illumination(dem, px, lat, 355)
    if ser is not None:
        mean_t = float(np.nanmean((ser["gunduz"] + ser["gece"]) / 2))
        frost_n = int((ser["gece"] < 0).sum())
        allday = int((ser["gunduz"] < 0).sum())
        # yükseklikle soğuma: ortalama yükseklikteki değeri standart 6,5 °C/km ile zirveye taşı
        top_t = mean_t - 6.5 * (float(dem.max()) - float(dem.mean())) / 1000
        m = st.columns(5)
        m[0].metric("Yıllık ort. yüzey sıcaklığı", f"{mean_t:.1f} °C")
        m[1].metric("Zirvede tahmini yıllık ort.", f"{top_t:.1f} °C", help="Standart 6,5 °C/km soğuma ile tahmin.")
        m[2].metric("Gece donu görülen dönem", f"{frost_n} × 8 gün")
        m[3].metric("Gündüz de donlu dönem", f"{allday} × 8 gün")
        m[4].metric("Kışın < 3 sa güneş alan", f"%{100 * (hw < 3).mean():.1f}")
        if top_t > 0:
            st.success("**Permafrost beklenmez.** Kalıcı donmuş zemin için yıllık ortalama zemin sıcaklığının 0 °C altında "
                       f"olması gerekir; burada alan ortalaması {mean_t:.1f} °C, en yüksek noktada bile yaklaşık {top_t:.1f} °C. "
                       "Buz yalnızca mevsimlik kar ve gece donu olarak görülür.")
        else:
            st.warning(f"Zirve kesiminde tahmini yıllık ortalama {top_t:.1f} °C; gölgeli kuzey yamaçlarda mevsimlik donmuş "
                       "zemin uzun sürebilir. Kalıcı permafrost için saha ölçümü gerekir.")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
            ax.axhspan(min(-15, float(ser["gece_min"].min()) - 2), 0, color="#cfe8ff", alpha=0.6, lw=0)
            ax.plot(ser["tarih"], ser["gece"], "-o", ms=2.5, color="#1f4e9c", lw=1.2, label="Gece (alan ort.)")
            ax.plot(ser["tarih"], ser["gece_min"], color="#1f4e9c", lw=0.8, ls=":", label="En soğuk piksel")
            ax.plot(ser["tarih"], ser["gunduz"], color="#d62728", lw=0.8, alpha=0.6, label="Gündüz")
            ax.axhline(0, color="#333", lw=0.7)
            ax.set_ylabel("Yüzey sıcaklığı (°C)")
            ax.set_title(f"Donma eşiğine göre yüzey sıcaklığı — {myear}", loc="left")
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
            ax.tick_params(axis="x", labelsize=7)
            show(fig)
        with c2:
            show(map_fig(hw, bbox, "Soğuk tuzak adayları: 21 Aralık güneş süresi", cmap="cividis", label="saat/gün",
                         vmin=0, vmax=float(hw.max()))[0])
        st.caption("Mavi bant donma bölgesidir. Kışın günde birkaç saatten az güneş alan krater içleri ve kuzey yamaçlar "
                   "karın ve donun en uzun kaldığı 'soğuk tuzak' adaylarıdır (Ay ve Mars kutup kraterlerinin küçük ölçekli benzeri).")

    st.subheader("Potansiyel kaynak özeti")
    rows = []
    try:
        occ, tr = get_jrc(bbox, n)
        rows.append(("Açık su yüzeyi (≥ %75 kalıcı)", float((occ >= 75).sum() * px * px / 1e6), "#2b6cb0"))
        rows.append(("Mevsimlik su yüzeyi", float(((occ > 5) & (occ < 75)).sum() * px * px / 1e6), "#9ecae1"))
    except Exception:
        pass
    try:
        s2 = get_s2(bbox, n, year)
        ndvi, rock, water, veg, bad, _ = spectral_products(s2)
        R = s2["refl"]
        alb = np.mean([R[b] for b in ("B02", "B03", "B04", "B8A", "B11", "B12")], axis=0)
        rows.append(("Koyu bazalt / cüruf yüzeyi (yapı agregası)", float((rock & (alb < 0.15)).sum() * px * px / 1e6), "#2b2b33"))
        rows.append(("Açık renkli tüf / pomza / çökel yüzeyi", float((rock & (alb > 0.30)).sum() * px * px / 1e6), "#d8c08a"))
        rows.append(("Bitki örtüsü (biyokütle)", float(veg.sum() * px * px / 1e6), "#4c9a4c"))
    except Exception:
        pass
    hs, _, _ = illumination(dem, px, lat, 172)
    c1, c2 = st.columns(2)
    with c1:
        if rows:
            rows.sort(key=lambda r: r[1])
            show(share_bar([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                           "Kaynak sınıflarının kapladığı alan", xlabel="Alan (km²)", figsize=(5.2, 3.4)))
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.hist(hs.ravel(), bins=40, color="#e9a23b", alpha=0.8, label="21 Haziran", weights=np.full(hs.size, 100 / hs.size))
        ax.hist(hw.ravel(), bins=40, color="#34558b", alpha=0.8, label="21 Aralık", weights=np.full(hw.size, 100 / hw.size))
        ax.set_xlabel("Doğrudan güneş süresi (saat/gün)")
        ax.set_ylabel("Alan payı (%)")
        ax.set_title("Güneş enerjisi kaynağı — yaz / kış", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    st.caption("Kaynak sınıfları uydu spektrumundan ayrılmıştır; bazalt ve cüruf, yerinde kaynak kullanımı (ISRU) "
               "çalışmalarında Ay/Mars regoliti benzeri yapı malzemesi olarak değerlendirilir. Rezerv değil, yüzey alanıdır.")


# --------------------------------------------------------------------------------------
# BÖLÜM 5: toz, rüzgâr, erozyon ve yüzey süreçleri
# --------------------------------------------------------------------------------------
def tab_processes(bbox, n, year, dem, px, wc):
    st.caption("Planetary Computer'da kırpılabilir bir rüzgâr ya da toz (aerosol) ürünü yoktur. Bu bölüm rüzgâra ve toza "
               "**maruziyeti** arazi biçiminden, toz ve erozyon **potansiyelini** yüzey örtüsü ve eğimden türetir; "
               "ölçülmüş rüzgâr hızı içermez.")
    slope, _ = slope_normals(dem, px)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    try:
        s2 = get_s2(bbox, n, year)
    except Exception as e:
        st.warning(f"Sentinel-2 verisi alınamadı: {e}")
        return
    R = s2["refl"]
    ndvi, rock, water, veg, bad, _ = spectral_products(s2)
    bsi = ((R["B11"] + R["B04"]) - (R["B08"] + R["B02"])) / ((R["B11"] + R["B04"]) + (R["B08"] + R["B02"]))
    ndmi = (R["B8A"] - R["B11"]) / (R["B8A"] + R["B11"])
    with st.spinner("Açıklık ve akış birikimi hesaplanıyor…"):
        opn = openness(dem, px)
        depth, lab, dep = find_depressions(dem, px, bbox)
        sinks = tuple(int(r["_r"]) * dem.shape[1] + int(r["_c"]) for _, r in dep.iterrows())
        fr = flow_routing(dem, px, max(0.3, round(150 * px * px / 1e6, 2)), sinks)
    ls, cfac, ero, spi = erosion_index(fr["acc_km2"], slope, ndvi, px)

    # toz kaynağı potansiyeli: çıplak + kuru + düz + rüzgâra açık
    s_bare = np.clip((0.25 - np.nan_to_num(ndvi, nan=1)) / 0.20, 0, 1)
    s_dry = np.clip((0.05 - np.nan_to_num(ndmi, nan=1)) / 0.20, 0, 1)
    s_flat = np.clip((6 - slope) / 5, 0, 1)
    s_open = np.clip((opn - 80) / 9, 0, 1)
    dust = (s_bare * s_dry) ** 0.5 * (0.5 + 0.5 * s_flat) * (0.5 + 0.5 * s_open)
    dust[water | bad] = 0
    dcls = np.digitize(dust, [0.25, 0.5, 0.75]).astype("uint8") + 1
    DUST = {1: ("Düşük", "#e8e4d8"), 2: ("Orta", "#e3c27a"), 3: ("Yüksek", "#c9813a"), 4: ("Çok yüksek", "#8c3b1a")}
    ecls = np.digitize(ero, [0.05, 0.5, 2.0, 6.0]).astype("uint8") + 1
    ERO = {1: ("Çok düşük", "#eef3e2"), 2: ("Düşük", "#cfe0a8"), 3: ("Orta", "#f1cf6b"),
           4: ("Yüksek", "#e08a3c"), 5: ("Çok yüksek", "#b2301e")}

    m = st.columns(5)
    m[0].metric("Çıplak / seyrek yüzey (NDVI < 0,15)", f"%{100 * (ndvi < 0.15).mean():.0f}")
    m[1].metric("Yüksek toz potansiyeli", f"%{100 * (dcls >= 3).mean():.1f}")
    m[2].metric("Yüksek erozyon duyarlılığı", f"%{100 * (ecls >= 4).mean():.1f}")
    m[3].metric("Ortalama açıklık", f"{opn.mean():.1f}°")
    m[4].metric("Korunaklı alan (açıklık < 80°)", f"%{100 * (opn < 80).mean():.1f}")

    st.subheader("Toz")
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(bsi, bbox, "Çıplak toprak indeksi (BSI)", cmap="BrBG_r", label="(B11+B04−B08−B02)/(toplam)",
                     vmin=-0.3, vmax=0.3)[0])
    with c2:
        show(class_fig(dcls, bbox, "Toz kaynağı potansiyeli", DUST))
    c1, c2 = st.columns(2)
    with c1:
        v, cnt = np.unique(dcls, return_counts=True)
        show(share_bar([DUST[int(x)][0] for x in v], list(100 * cnt / cnt.sum()), [DUST[int(x)][1] for x in v],
                       "Toz potansiyeli sınıflarının payı"))
    with c2:
        if wc is not None:
            rows = [(WORLDCOVER[k][0], WORLDCOVER[k][1], float(dust[wc == k].mean()))
                    for k in WORLDCOVER if (wc == k).mean() > 0.005]
            rows.sort(key=lambda r: r[2])
            show(share_bar([r[0] for r in rows], [r[2] for r in rows], [r[1] for r in rows],
                           "Yüzey örtüsüne göre ortalama toz puanı", xlabel="Puan (0–1)"))
    st.caption("Toz puanı = çıplaklık × kuruluk, düzlük ve rüzgâra açıklıkla ağırlıklandırılmıştır. Kuruyan göl tabanları, "
               "nadasa bırakılmış tarlalar ve ince taneli volkanik kül düzlükleri en yüksek puanı alır.")

    st.subheader("Rüzgâra maruziyet (arazi biçiminden)")
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(opn, bbox, "Topoğrafik açıklık (1,5 km yarıçap)", cmap="viridis", label="derece (90 = tam açık)",
                     vmin=float(np.percentile(opn, 1)), vmax=90)[0])
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        bins = np.linspace(dem.min(), dem.max(), 16)
        which = np.clip(np.digitize(dem.ravel(), bins) - 1, 0, 14)
        mid = (bins[:-1] + bins[1:]) / 2
        mo = [opn.ravel()[which == b].mean() if (which == b).any() else np.nan for b in range(15)]
        lo = [np.percentile(opn.ravel()[which == b], 10) if (which == b).any() else np.nan for b in range(15)]
        ax.fill_betweenx(mid, lo, mo, color="#5aa469", alpha=0.25, lw=0, label="%10 – ortalama")
        ax.plot(mo, mid, "-o", ms=3, color="#2f6b3c", lw=1.3, label="ortalama açıklık")
        ax.set_xlabel("Açıklık (°)")
        ax.set_ylabel("Yükseklik (m)")
        ax.set_title("Yüksekliğe göre açıklık", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    st.caption("Açıklık düşük olan krater içleri, vadi tabanları ve koni etekleri rüzgârdan korunaklıdır ve taşınan tozun "
               "çökeldiği yerlerdir; sırtlar ve zirveler en açık, aşındırmaya en çok maruz kesimlerdir.")

    st.subheader("Su erozyonu")
    c1, c2 = st.columns(2)
    with c1:
        show(class_fig(ecls, bbox, "Göreli erozyon duyarlılığı (LS × C)", ERO))
    with c2:
        show(map_fig(np.log10(np.maximum(spi, 1)), bbox, "Akarsu gücü indeksi (SPI)", cmap="magma_r", label="log₁₀(a · tan β)",
                     under=shade)[0])
    c1, c2 = st.columns(2)
    with c1:
        v, cnt = np.unique(ecls, return_counts=True)
        show(share_bar([ERO[int(x)][0] for x in v], list(100 * cnt / cnt.sum()), [ERO[int(x)][1] for x in v],
                       "Erozyon duyarlılığı sınıflarının payı"))
    with c2:
        try:
            nd, ny = ndvi_best_year(bbox, year)
            fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
            ax.fill_between(nd["tarih"], nd["p10"], nd["p90"], color="#7fbf7b", alpha=0.3, lw=0, label="%10–%90")
            ax.plot(nd["tarih"], nd["ndvi"], "-o", ms=2.5, color="#2f6b3c", lw=1.3, label="NDVI (alan ort.)")
            ax.axhline(0.15, color="#a1632a", lw=0.8, ls="--")
            ax.set_ylabel("NDVI")
            ax.set_title(f"Koruyucu bitki örtüsünün yıl içi seyri — {ny}", loc="left")
            ax.legend(fontsize=7, frameon=False)
            ax.tick_params(axis="x", labelsize=7)
            show(fig)
        except Exception as e:
            st.info(f"Bitki örtüsü serisi atlandı: {e}")
    st.caption("LS, yamaç uzunluğu ve dikliğinin (RUSLE) birleşik etkisidir; C, NDVI'dan türetilen örtü koruma katsayısıdır. "
               "Toprak aşınabilirliği ve yağış şiddeti bilinmediğinden sonuç ton/hektar değil **göreli** bir sıralamadır. "
               "Kesikli çizginin (NDVI 0,15) altındaki dönemlerde zemin fiilen korumasızdır.")


# --------------------------------------------------------------------------------------
# BÖLÜM 6: biyoloji, ekstremofil nişleri, yaşanabilirlik ve biyolojik imzalar
# --------------------------------------------------------------------------------------
def tab_biology(bbox, n, year, dem, px, wc, lst):
    st.caption("Uydu verisi mikrobiyal yaşamı doğrudan göremez. Burada görülen biyolojik imza **klorofildir** (bitki örtüsü); "
               "ekstremofil nişleri ise sıcaklık, nem, ışık ve su kimyasına dair dolaylı göstergelerden sınıflandırılmıştır.")
    lat = (bbox[1] + bbox[3]) / 2
    slope, _ = slope_normals(dem, px)
    try:
        s2 = get_s2(bbox, n, year)
    except Exception as e:
        st.warning(f"Sentinel-2 verisi alınamadı: {e}")
        return
    R = s2["refl"]
    ndvi, rock, water, veg, bad, _ = spectral_products(s2)
    ndmi = (R["B8A"] - R["B11"]) / (R["B8A"] + R["B11"])
    alb = np.mean([R[b] for b in ("B02", "B03", "B04", "B8A", "B11", "B12")], axis=0)
    occ = tr = None
    try:
        occ, tr = get_jrc(bbox, n)
    except Exception:
        pass
    hw, _, _ = illumination(dem, px, lat, 355)
    if hw.shape != dem.shape:
        hw = ndi.zoom(hw, (dem.shape[0] / hw.shape[0], dem.shape[1] / hw.shape[1]), order=1)[:dem.shape[0], :dem.shape[1]]
        if hw.shape != dem.shape:
            hw = np.pad(hw, ((0, dem.shape[0] - hw.shape[0]), (0, dem.shape[1] - hw.shape[1])), mode="edge")
    wet = water | ((occ >= 50) if occ is not None else False)

    # yaşanabilirlik puanı: nem, suya yakınlık, ısıl ılımanlık, mevcut bitki örtüsü
    s_moist = np.clip((np.nan_to_num(ndmi, nan=-0.3) + 0.25) / 0.5, 0, 1)
    d_water = ndi.distance_transform_edt(~wet) * px if wet.any() else np.full(dem.shape, 1e5)
    s_water = np.exp(-d_water / 2000)
    if lst is not None and np.isfinite(lst).any():
        l = np.where(np.isfinite(lst), lst, np.nanmedian(lst))
        s_therm = np.clip((np.nanpercentile(l, 98) - l) / max(np.nanpercentile(l, 98) - np.nanpercentile(l, 2), 1e-6), 0, 1)
    else:
        l, s_therm = None, np.full(dem.shape, 0.5)
    s_veg = np.clip(np.nan_to_num(ndvi, nan=0) / 0.6, 0, 1)
    hab = 0.35 * s_moist + 0.20 * s_water + 0.25 * s_therm + 0.20 * s_veg

    # ekstremofil benzeri nişler (kural tabanlı; öncelik sırasıyla)
    NICHE = {1: ("Ilıman / bitkili yüzey", "#8fc98a"), 2: ("Kuru çıplak toprak / kaya (kserofil)", "#d9c79a"),
             3: ("Sıcak koyu lav yüzeyi (termofil, endolitik)", "#5a2a27"),
             4: ("Parlak tuzlu / kuruyan göl tabanı (halofil)", "#f2f2f2"),
             5: ("Soğuk gölgeli yamaç / krater içi (psikrotolerant)", "#6a8fc7"),
             6: ("Açık su (sucul; göl kimyasına bağlı)", "#1f5fbf")}
    niche = np.full(dem.shape, 1, dtype="uint8")
    niche[rock] = 2
    hot = (l >= np.nanpercentile(l, 80)) if l is not None else np.zeros(dem.shape, bool)
    niche[rock & (alb < 0.15) & hot] = 3
    dried = np.isin(tr, [3, 6, 8, 10]) if tr is not None else np.zeros(dem.shape, bool)
    niche[(rock & (alb > 0.32) & (slope < 3)) | (dried & ~water)] = 4
    niche[(hw < 4) & (slope > 8)] = 5
    niche[wet] = 6

    m = st.columns(5)
    m[0].metric("Fotosentetik örtü (NDVI > 0,30)", f"%{100 * veg.mean():.0f}")
    m[1].metric("Ortalama NDVI", f"{np.nanmean(ndvi):.2f}")
    m[2].metric("Ort. yaşanabilirlik puanı", f"{hab.mean():.2f}")
    m[3].metric("Uç koşullu niş payı", f"%{100 * np.isin(niche, [3, 4, 5]).mean():.1f}")
    m[4].metric("Açık su", f"{wet.sum() * px * px / 1e6:.2f} km²")

    st.subheader("Biyolojik imza: klorofil ve üretkenlik")
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(ndvi, bbox, f"Bitki indeksi NDVI — {s2['tarih']}", cmap="RdYlGn", label="NDVI", vmin=-0.1, vmax=0.8)[0])
    with c2:
        try:
            nd, ny = ndvi_best_year(bbox, year)
            fig, ax = plt.subplots(figsize=(5.2, 4.0), layout="constrained")
            ax.fill_between(nd["tarih"], nd["p10"], nd["p90"], color="#7fbf7b", alpha=0.3, lw=0, label="NDVI %10–%90")
            ax.plot(nd["tarih"], nd["ndvi"], "-o", ms=2.5, color="#2f6b3c", lw=1.3, label="NDVI")
            ax.plot(nd["tarih"], nd["evi"], "-s", ms=2.5, color="#8a6d1f", lw=1.1, label="EVI")
            peak = nd.loc[nd["ndvi"].idxmax()]
            ax.annotate(f"en yeşil: {peak['tarih']:%d.%m}", (peak["tarih"], peak["ndvi"]), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=7)
            ax.set_ylabel("İndeks")
            ax.set_title(f"Fenoloji — {ny} (MODIS 16 günlük, 250 m)", loc="left")
            ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
            ax.tick_params(axis="x", labelsize=7)
            show(fig)
        except Exception as e:
            st.info(f"Fenoloji serisi atlandı: {e}")
    c1, c2 = st.columns(2)
    with c1:
        if wc is not None:
            ks = [k for k in WORLDCOVER if (wc == k).mean() > 0.005]
            data = [ndvi[(wc == k) & np.isfinite(ndvi)] for k in ks]
            fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
            bp = ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6,
                            medianprops={"color": "#222"})
            for patch, k in zip(bp["boxes"], ks):
                patch.set_facecolor(WORLDCOVER[k][1])
                patch.set_edgecolor("#555")
            ax.set_yticks(range(1, len(ks) + 1), [WORLDCOVER[k][0] for k in ks], fontsize=7)
            ax.set_xlabel("NDVI")
            ax.set_title("Yüzey örtüsüne göre NDVI dağılımı", loc="left")
            show(fig)
    with c2:
        try:
            npp = get_npp(bbox, year)
            fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
            ax.bar(npp["yil"].astype(int).astype(str), npp["npp"], color="#6aa84f", edgecolor="#555", lw=0.5)
            for i, v in enumerate(npp["npp"]):
                ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
            ax.set_ylabel("gC / m² / yıl")
            ax.set_title("Yıllık net birincil üretim (MODIS)", loc="left")
            show(fig)
            st.caption(f"Ürün yalnızca bitkili pikselleri kapsar (alanın %{100 * npp['kapsama'].iloc[-1]:.0f}'i). "
                       "Karşılaştırma: çöl < 100, bozkır 200–400, ılıman orman 600–1200 gC/m²/yıl.")
        except Exception as e:
            st.info(f"Net birincil üretim atlandı: {e}")

    st.subheader("Yaşanabilirlik")
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(hab, bbox, "Yaşanabilirlik puanı (nem · su · ısı · örtü)", cmap="YlGn", label="0 = elverişsiz, 1 = elverişli",
                     vmin=0, vmax=1)[0])
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        parts = {"Nem (NDMI)": s_moist, "Suya yakınlık": s_water, "Isıl ılımanlık": s_therm, "Bitki örtüsü": s_veg}
        w = [0.35, 0.20, 0.25, 0.20]
        ax.barh(list(parts), [float(v.mean()) for v in parts.values()],
                color=["#4f9bb5", "#2b6cb0", "#e08a3c", "#6aa84f"], edgecolor="#555", lw=0.5)
        for i, (v, wi) in enumerate(zip(parts.values(), w)):
            ax.text(float(v.mean()), i, f" {v.mean():.2f}  (ağırlık {wi:.2f})", va="center", fontsize=7)
        ax.set_xlim(0, 1.25)
        ax.set_xlabel("Alan ortalaması (0–1)")
        ax.set_title("Yaşanabilirliği sınırlayan etkenler", loc="left")
        show(fig)
    st.caption("En düşük puanlı bileşen, bu alanda yaşamı en çok sınırlayan etkendir. Puan, yüzeydeki makroskopik yaşam "
               "için göreli bir elverişlilik ölçüsüdür; ağırlıklar dosyada `tab_biology` içinde değiştirilebilir.")

    st.subheader("Ekstremofil benzeri nişler")
    c1, c2 = st.columns(2)
    with c1:
        show(class_fig(niche, bbox, "Niş sınıfları (kural tabanlı)", NICHE))
    with c2:
        v, cnt = np.unique(niche, return_counts=True)
        order = np.argsort(cnt)
        show(share_bar([NICHE[int(x)][0].split(" (")[0] for x in v[order]], list(100 * cnt[order] / cnt.sum()),
                       [NICHE[int(x)][1] for x in v[order]], "Niş sınıflarının alan payı", figsize=(5.2, 3.8)))
        if l is not None:
            fig, ax = plt.subplots(figsize=(5.2, 2.6), layout="constrained")
            ks = [int(x) for x in v if np.isfinite(l[niche == x]).any()]
            ax.bar([NICHE[k][0].split(" ")[0] for k in ks], [float(np.nanmean(l[niche == k])) for k in ks],
                   color=[NICHE[k][1] for k in ks], edgecolor="#555", lw=0.5)
            ax.set_ylabel("Yaz LST (°C)")
            ax.set_title("Nişlerin yaz yüzey sıcaklığı", loc="left")
            ax.tick_params(axis="x", labelsize=7)
            lo = min(float(np.nanmean(l[niche == k])) for k in ks)
            ax.set_ylim(lo - 4, None)
            show(fig)
    st.caption("Sınıflar yalnızca fiziksel ortamı tanımlar: koyu lav yüzeyleri yazın en sıcak ve kuru (kaya içi/endolitik "
               "topluluklar için analog), kuruyan göl tabanları tuzlu (halofil analoğu), kışın az güneş alan krater içleri "
               "ve kuzey yamaçlar en soğuk ve nemli mikro-ortamlardır. Canlı varlığı saha örneklemesiyle doğrulanır.")


# --------------------------------------------------------------------------------------
# BÖLÜM 7: mağaralar, lav tüpleri ve yeraltı ortamları
# --------------------------------------------------------------------------------------
def tab_subsurface(bbox, n, year, dem, px, wc, lst):
    st.caption(f"Lav tüpleri ve mağara girişleri çoğunlukla 30 m'den küçüktür; bu panelin {px:.0f} m'lik ızgarası onları "
               "**doğrudan göremez**. Aşağıdakiler tüp oluşumuna elverişli zemini ve çökme çukuru olabilecek kapalı "
               "çöküntüleri gösteren **aday** haritalarıdır; doğrulama için LiDAR, drone ya da yer radarı gerekir.")
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    lava = None
    try:
        s2 = get_s2(bbox, n, year)
        R = s2["refl"]
        ndvi, rock, water, veg, bad, _ = spectral_products(s2)
        alb = np.mean([R[b] for b in ("B02", "B03", "B04", "B8A", "B11", "B12")], axis=0)
        lava = ndi.binary_opening(rock & (alb < 0.16), iterations=1)
    except Exception as e:
        st.info(f"Sentinel-2 alınamadı; lav yüzeyi ayrımı yapılamadı: {e}")
    if lava is None:
        lava = np.zeros(dem.shape, bool)
    subsurface_body(bbox, dem, px, lava, lst)


def subsurface_body(bbox, dem, px, lava, lst, lava_name="Koyu lav / cüruf"):
    """Lav zemini maskesinden çukur adayları, tüp elverişliliği ve grafikler (Dünya ve Mars için ortak)."""
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    near_lava = ndi.binary_dilation(lava, iterations=max(1, int(round(200 / px))))

    # küçük kapalı çöküntüler: çökme çukuru / tavan penceresi (skylight) adayları
    depth = fill_depressions(dem) - dem
    lab, nlab = ndi.label(depth > 0.5)
    rows = []
    if nlab:
        idx = np.arange(1, nlab + 1)
        area = ndi.sum(np.ones_like(depth), lab, idx) * px * px
        dmax = ndi.maximum(depth, lab, idx)
        pos = ndi.maximum_position(depth, lab, idx)
        for k in range(nlab):
            if dmax[k] >= 2.0 and area[k] <= max(150_000, 40 * px * px):
                r, c = pos[k]
                lon, la = rc_to_lonlat(r, c, dem.shape, bbox)
                rows.append({"Enlem": round(la, 4), "Boylam": round(lon, 4), "Derinlik (m)": round(float(dmax[k]), 1),
                             "Eşdeğer çap (m)": int(round(2 * math.sqrt(area[k] / math.pi))),
                             "Lav içinde": bool(near_lava[r, c]), "Eğim (°)": round(float(slope[r, c]), 1),
                             "_r": int(r), "_c": int(c)})
    pits = pd.DataFrame(rows)
    if len(pits):
        pits = pits.sort_values(["Lav içinde", "Derinlik (m)"], ascending=False).head(40).reset_index(drop=True)
        pits.index = [f"P{i + 1}" for i in range(len(pits))]

    # tüp elverişlilik indeksi: lav yüzeyi × hafif eğim (0,3–6°) × düşük pürüzlülük
    s_slope = np.clip(np.minimum((slope - 0.2) / 0.8, (8 - slope) / 3), 0, 1)
    s_smooth = np.clip(1 - tri / max(float(np.percentile(tri, 95)), 1e-6), 0, 1)
    fav = np.where(lava, 1.0, np.where(near_lava, 0.5, 0.0)) * s_slope * (0.4 + 0.6 * s_smooth)

    n_in = int(pits["Lav içinde"].sum()) if len(pits) else 0
    m = st.columns(5)
    m[0].metric(f"{lava_name} yüzeyi", f"{lava.sum() * px * px / 1e6:.0f} km²")
    m[1].metric("Lav yüzeyi payı", f"%{100 * lava.mean():.1f}")
    m[2].metric("Çukur adayı (≥ 2 m)", f"{len(pits)}")
    m[3].metric("Lav içindeki / kenarındaki", f"{n_in}")
    m[4].metric("Tüpe elverişli alan (puan > 0,5)", f"{(fav > 0.5).sum() * px * px / 1e6:.1f} km²")

    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = _axes(bbox)
        ax.imshow(shade, extent=ext, cmap="gray", vmin=-0.6, vmax=1.2)
        ax.imshow(np.where(lava, 1.0, np.nan), extent=ext, cmap=ListedColormap(["#7a1f1f"]), alpha=0.55, interpolation="nearest")
        if len(pits):
            a = pits[pits["Lav içinde"]]
            b = pits[~pits["Lav içinde"]]
            ax.scatter(*lonlat(b), s=14, facecolor="none", edgecolor="#1f4e9c", lw=0.9, label="Çukur (lav dışı)")
            ax.scatter(*lonlat(a), s=22, facecolor="#ffd23f", edgecolor="#222", lw=0.6, label="Çukur (lav içi / kenarı)")
            ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
        ax.set_title(f"{lava_name} yüzeyi ve çökme çukuru adayları", loc="left")
        show(fig)
    with c2:
        fig, ax = map_fig(np.where(fav > 0.02, fav, np.nan), bbox, "Lav tüpü elverişlilik indeksi", cmap="inferno",
                          label="0–1", vmin=0, vmax=1, under=shade)
        if len(pits):
            ax.scatter(*lonlat(pits[pits["Lav içinde"]]), s=12, facecolor="none", edgecolor="#39d5ff", lw=0.8)
        show(fig)
    st.caption("Tüpler, akışkan bazaltın hafif eğimli (yaklaşık 0,3–6°) zeminde uzun süre aktığı yerlerde oluşur; indeks "
               "koyu lav yüzeyi, bu eğim aralığı ve düşük pürüzlülüğü birleştirir. Aynı hat üzerinde dizilen lav içi çukurlar "
               "çökmüş tüp tavanına işaret edebilir.")

    c1, c2 = st.columns(2)
    with c1:
        if lst is not None and np.isfinite(lst).any():
            l = np.where(np.isfinite(lst), lst, np.nanmedian(lst))
            k = max(3, int(round(600 / px)) | 1)
            resid = np.where(np.isfinite(lst), l - ndi.uniform_filter(l, size=k, mode="nearest"), np.nan)
            lim = float(max(1.0, np.nanpercentile(np.abs(resid), 98)))
            fig, ax = map_fig(resid, bbox, "Yerel sıcaklık farkı (Landsat, 600 m çevreye göre)", cmap="RdBu_r",
                              label="°C  (mavi = çevresinden serin)", vmin=-lim, vmax=lim)
            if len(pits):
                ax.scatter(*lonlat(pits), s=10, facecolor="none", edgecolor="#111", lw=0.6)
            show(fig)
        elif BODY.get("key") != "mars":
            st.info("Landsat yüzey sıcaklığı yok; ısıl fark haritası çizilemedi.")
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 4.0), layout="constrained")
        if len(pits):
            for flag, col, labl in ((True, "#e0a800", "Lav içi / kenarı"), (False, "#1f4e9c", "Lav dışı")):
                q = pits[pits["Lav içinde"] == flag]
                ax.scatter(q["Eşdeğer çap (m)"], q["Derinlik (m)"], s=30, c=col, edgecolor="white", lw=0.5, label=labl, zorder=3)
            x = np.array([pits["Eşdeğer çap (m)"].min(), pits["Eşdeğer çap (m)"].max()], dtype=float)
            ax.plot(x, 0.1 * x, color="#888", lw=0.7, ls="--")
            ax.annotate("d/D = 0,1", (x[-1], 0.1 * x[-1]), fontsize=6.5, color="#666", ha="right", va="bottom")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend(fontsize=7, frameon=False)
        else:
            ax.text(0.5, 0.5, "Çukur adayı bulunamadı", ha="center", va="center", transform=ax.transAxes, color="#777")
        ax.set_xlabel("Eşdeğer çap (m)")
        ax.set_ylabel("Derinlik (m)")
        ax.set_title("Çukur adaylarının boyutu", loc="left")
        show(fig)
    st.caption("Gündüz görüntüsünde mağara ağızları ve derin çukurlar çevresinden serin görünür; ancak 30 m'lik termal piksel "
               "küçük girişleri seyreltir, gölge ve bitki de aynı izi verir. Mavi leke ile çukur adayının çakışması en güçlü ipucudur.")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        bins = np.linspace(0, min(30, float(np.percentile(slope, 99.5))), 40)
        if lava.any():
            ax.hist(slope[lava], bins=bins, color="#7a1f1f", alpha=0.75, label="Lav yüzeyi", density=True)
        ax.hist(slope[~lava], bins=bins, color="#9aa7b5", alpha=0.6, label="Diğer yüzeyler", density=True)
        ax.axvspan(0.3, 6, color="#ffd23f", alpha=0.18, lw=0)
        ax.set_xlabel("Eğim (°)")
        ax.set_ylabel("Yoğunluk")
        ax.set_title("Eğim dağılımı (sarı: tüp oluşumuna elverişli aralık)", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    with c2:
        fig = plt.figure(figsize=(5.2, 3.8), layout="constrained")
        ax = fig.add_subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        q = pits[pits["Lav içinde"]] if len(pits) else pits
        if len(q) >= 4:
            lon, la = lonlat(q)
            x = (lon - lon.mean()) * BODY["m_lon"] / 1000 * math.cos(math.radians(la.mean()))
            y = (la - la.mean()) * BODY["m_lat"] / 1000
            az = []
            for i in range(len(q)):
                d = np.hypot(x - x[i], y - y[i])
                d[i] = np.inf
                j = int(np.argmin(d))
                if d[j] < max(3.0, 15 * px / 1000):
                    az.append(math.degrees(math.atan2(x[j] - x[i], y[j] - y[i])) % 180)
            h, _ = np.histogram(az, bins=np.arange(0, 181, 20))
            for off in (0, 180):
                ax.bar(np.radians(np.arange(10, 180, 20) + off), h, width=np.radians(18), color="#e0a800", edgecolor="#444", lw=0.5)
            ax.set_title("Lav içi çukurların dizilim yönü (en yakın komşu)", loc="left", pad=14, fontsize=9)
        else:
            ax.set_title("Dizilim yönü için yeterli lav içi çukur yok", loc="left", pad=14, fontsize=9)
        ax.tick_params(labelsize=7)
        show(fig)
    if len(pits):
        st.markdown("**Çukur adayları** (önce lav içindekiler, derinliğe göre)")
        st.dataframe(pits.drop(columns=["_r", "_c"]))
    st.caption("Belirgin tek bir dizilim yönü, çukurların ortak bir akış hattı (tüp ekseni ya da çatlak) boyunca sıralandığını "
               "düşündürür. Dağınık bir gül ise rastgele dağılıma ya da DEM gürültüsüne işaret eder.")


# --------------------------------------------------------------------------------------
# BÖLÜM 8: iniş, hareketlilik, navigasyon, robotik operasyon ve inşaat koşulları
# --------------------------------------------------------------------------------------
def tab_operations(bbox, n, year, dem, px, wc):
    lat = (bbox[1] + bbox[3]) / 2
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    depth, lab, dep = find_depressions(dem, px, bbox)
    tpi, peaks = find_peaks(dem, px, bbox)
    water = (wc == 80) | (wc == 90) if wc is not None else np.zeros(dem.shape, bool)
    obst = water | (np.isin(wc, [10, 50]) if wc is not None else False)  # su, orman, yapılaşma
    k = max(3, int(round(150 / px)) | 1)
    smax = ndi.maximum_filter(slope, size=k)
    safe = (slope <= 5) & (smax <= 8) & ~obst
    with st.spinner("İniş alanları ve güzergâh hesaplanıyor…"):
        dist, sites = landing_sites(safe, px, bbox)
        opn = openness(dem, px)
    MOB = {1: ("İnişe uygun (≤ 5°, engelsiz)", "#3a9a5c"), 2: ("Tekerlekli araç (≤ 15°)", "#b7d98a"),
           3: ("Paletli araç / yaya (≤ 25°)", "#f1cf6b"), 4: ("Geçilemez (> 25°)", "#b2301e"), 5: ("Su", "#2b6cb0")}
    mob = np.full(dem.shape, 2, dtype="uint8")
    mob[slope > 15] = 3
    mob[slope > 25] = 4
    mob[safe] = 1
    mob[water] = 5

    m = st.columns(5)
    m[0].metric("İnişe uygun alan", f"%{100 * safe.mean():.0f}")
    m[1].metric("En büyük engelsiz daire", f"{int(sites['Engelsiz yarıçap (m)'].iloc[0])} m yarıçap" if len(sites) else "—")
    m[2].metric("Tekerlekli araca açık", f"%{100 * (slope <= 15).mean():.0f}")
    m[3].metric("Geçilemez (> 25°)", f"%{100 * (slope > 25).mean():.1f}")
    m[4].metric("Ort. gökyüzü açıklığı", f"{opn.mean():.0f}°")

    st.subheader("İniş alanı seçimi")
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = _axes(bbox, figsize=(5.2, 5.4))
        vals = [v for v in MOB if (mob == v).any()]
        lut = np.zeros(8, dtype=int)
        for i, v in enumerate(vals):
            lut[v] = i
        ax.imshow(lut[mob], extent=ext, cmap=ListedColormap([MOB[v][1] for v in vals]), vmin=0,
                  vmax=max(len(vals) - 1, 1), interpolation="nearest")
        ax.imshow(shade, extent=ext, cmap="gray", vmin=0, vmax=1, alpha=0.25)
        cosl = math.cos(math.radians(lat))
        for name, row in sites.iterrows():
            rad_deg = row["Engelsiz yarıçap (m)"] / BODY["m_lat"]
            t = np.linspace(0, 2 * math.pi, 80)
            ax.plot(row["Boylam"] + rad_deg / cosl * np.cos(t), row["Enlem"] + rad_deg * np.sin(t), color="#111", lw=1.1)
            ax.annotate(name, (row["Boylam"], row["Enlem"]), ha="center", va="center", fontsize=7, weight="bold")
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
        ax.legend(handles=[Patch(facecolor=MOB[v][1], edgecolor="#555", label=MOB[v][0]) for v in vals],
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7, frameon=False)
        ax.set_title("Geçilebilirlik sınıfları ve iniş daireleri", loc="left")
        show(fig)
    with c2:
        show(map_fig(np.where(safe, dist, np.nan), bbox, "En yakın engele uzaklık (güvenli alan içinde)", cmap="YlGn",
                     label="m", vmin=0, vmax=float(max(dist.max(), 1)), under=shade)[0])
        if len(sites):
            st.dataframe(sites.drop(columns=["_r", "_c"]))
    c1, c2 = st.columns(2)
    with c1:
        v, cnt = np.unique(mob, return_counts=True)
        show(share_bar([MOB[int(x)][0].split(" (")[0] for x in v], list(100 * cnt / cnt.sum()), [MOB[int(x)][1] for x in v],
                       "Geçilebilirlik sınıflarının payı"))
    with c2:
        s = np.sort(slope.ravel())
        fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
        ax.plot(s, np.linspace(0, 100, s.size), color="#333", lw=1.4)
        for thr, col, txt in ((5, "#3a9a5c", "iniş 5°"), (15, "#8ab04a", "tekerlekli 15°"), (25, "#b2301e", "paletli 25°")):
            ax.axvline(thr, color=col, lw=1, ls="--")
            ax.annotate(f"{txt}\n%{100 * (slope <= thr).mean():.0f}", (thr, 8), fontsize=7, color=col, ha="left",
                        xytext=(3, 0), textcoords="offset points")
        ax.set_xlim(0, max(30, float(np.percentile(slope, 99.5))))
        ax.set_xlabel("Eğim (°)")
        ax.set_ylabel("Bu eğimin altındaki alan (%)")
        ax.set_title("Kümülatif eğim eğrisi", loc="left")
        show(fig)
    st.caption(f"İniş ölçütü: eğim ≤ 5°, 150 m çevrede en dik eğim ≤ 8°, su / orman / yapılaşma yok. Eğim {px:.0f} m'lik "
               "piksellerden hesaplandığı için kaya blokları, lav yüzeyi pürüzleri ve küçük çukurlar gibi **metre ölçeğindeki "
               "tehlikeler görünmez**; gerçek iniş alanı sertifikasyonu için yüksek çözünürlüklü veri gerekir.")

    st.subheader("Hareketlilik ve güzergâh")
    tgt = None
    if len(dep) and dep["Derinlik (m)"].iloc[0] >= 15:
        tgt, tname = (int(dep["_r"].iloc[0]), int(dep["_c"].iloc[0])), f"{dep.index[0]} çöküntüsü (krater tabanı)"
    elif len(peaks):
        tgt, tname = (int(peaks["_r"].iloc[0]), int(peaks["_c"].iloc[0])), f"{peaks.index[0]} tepesi (koni zirvesi)"
    if len(sites) and tgt is not None:
        src = (int(sites["_r"].iloc[0]), int(sites["_c"].iloc[0]))
        pr, pc, total = least_cost_path(slope, water, px, src, tgt)
        H, W = dem.shape
        lon = bbox[0] + (pc + 0.5) / W * (bbox[2] - bbox[0])
        la = bbox[3] - (pr + 0.5) / H * (bbox[3] - bbox[1])
        seg = np.hypot(np.diff(pr), np.diff(pc)) * px
        d = np.concatenate([[0], np.cumsum(seg)]) / 1000
        z = dem[pr, pc]
        ps = slope[pr, pc]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = map_fig(slope, bbox, "En düşük maliyetli güzergâh", cmap="Greys", label="eğim (°)", vmin=0,
                              vmax=float(max(15, np.percentile(slope, 99))))
            ax.plot(lon, la, color="#e4572e", lw=1.8)
            ax.plot(lon[0], la[0], "o", ms=7, mfc="#3a9a5c", mec="white")
            ax.plot(lon[-1], la[-1], "*", ms=11, mfc="#ffd23f", mec="#222")
            show(fig)
        with c2:
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True, layout="constrained")
            a1.fill_between(d, z, z.min() - 5, color="#c9b27c", alpha=0.7, lw=0)
            a1.plot(d, z, color="#5a4526", lw=1.2)
            a1.set_ylabel("Yükseklik (m)")
            a1.set_title(f"Güzergâh profili: İ1 → {tname.split(' (')[0]}", loc="left")
            a2.fill_between(d, ps, color="#d9534f", alpha=0.35, lw=0)
            a2.plot(d, ps, color="#a12a25", lw=1)
            a2.axhline(15, color="#8ab04a", lw=0.8, ls="--")
            a2.axhline(25, color="#b2301e", lw=0.8, ls="--")
            a2.set_ylabel("Eğim (°)")
            a2.set_xlabel("Güzergâh boyunca uzaklık (km)")
            show(fig)
        straight = math.hypot((src[0] - tgt[0]), (src[1] - tgt[1])) * px / 1000
        st.caption(f"Başlangıç: en büyük iniş dairesi (İ1) · hedef: {tname}. Güzergâh **{d[-1]:.1f} km** "
                   f"(kuş uçuşu {straight:.1f} km), toplam tırmanış **{float(np.clip(np.diff(z), 0, None).sum()):.0f} m**, "
                   f"en dik kesim **{ps.max():.0f}°**, 15°'den dik kesimlerin payı %{100 * (ps > 15).mean():.0f}. "
                   "Maliyet eğimin karesiyle artar; su yüzeyleri ve 30°'den dik yamaçlar fiilen kapalıdır.")
    else:
        st.info("Güzergâh için uygun bir iniş alanı ya da hedef bulunamadı.")

    st.subheader("Navigasyon, enerji ve inşaat")
    hw, _, _ = illumination(dem, px, lat, 355)
    if hw.shape != dem.shape:
        hw = ndi.zoom(hw, (dem.shape[0] / hw.shape[0], dem.shape[1] / hw.shape[1]), order=1)[:dem.shape[0], :dem.shape[1]]
        if hw.shape != dem.shape:
            hw = np.pad(hw, ((0, dem.shape[0] - hw.shape[0]), (0, dem.shape[1] - hw.shape[1])), mode="edge")
    firm = np.full(dem.shape, 0.7)
    if wc is not None:
        for cls, val in ((60, 1.0), (30, 0.8), (20, 0.7), (40, 0.5), (50, 0.6), (10, 0.3), (80, 0.0), (90, 0.0)):
            firm[wc == cls] = val
    s_slope = np.clip((10 - slope) / 8, 0, 1)
    s_dry = np.clip((depth < 0.5) * 1.0 - 0, 0, 1) * np.clip(1 - tri / max(float(np.percentile(tri, 99)), 1e-6), 0, 1)
    s_sun = np.clip(hw / max(float(hw.max()), 1e-6), 0, 1)
    s_sky = np.clip((opn - 75) / 15, 0, 1)
    build = (0.40 * s_slope + 0.20 * firm + 0.15 * s_dry + 0.15 * s_sun + 0.10 * s_sky) * (~water)
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(build, bbox, "Üs / inşaat uygunluk puanı", cmap="YlGnBu", label="0–1", vmin=0, vmax=1)[0])
    with c2:
        show(map_fig(opn, bbox, "Gökyüzü açıklığı (haberleşme, konum belirleme)", cmap="viridis",
                     label="derece (90 = ufuk tamamen açık)", vmin=float(np.percentile(opn, 1)), vmax=90)[0])
    c1, c2 = st.columns(2)
    with c1:
        parts = {"Düzlük (eğim)": s_slope, "Zemin sağlamlığı": firm, "Kuru ve pürüzsüz zemin": s_dry,
                 "Kış güneşi (enerji)": s_sun, "Gökyüzü açıklığı": s_sky}
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.barh(list(parts), [float(v.mean()) for v in parts.values()],
                color=["#3a9a5c", "#8d7b6c", "#4f9bb5", "#e9a23b", "#5b7c99"], edgecolor="#555", lw=0.5)
        for i, v in enumerate(parts.values()):
            ax.text(float(v.mean()), i, f" {v.mean():.2f}", va="center", fontsize=7)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Alan ortalaması (0–1)")
        ax.set_title("İnşaat uygunluğunun bileşenleri", loc="left")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        ax.hist(build[~water], bins=40, color="#3f7fb0", edgecolor="none",
                weights=np.full((~water).sum(), 100 / max((~water).sum(), 1)))
        ax.axvline(0.75, color="#222", lw=0.9, ls="--")
        ax.set_xlabel("Uygunluk puanı")
        ax.set_ylabel("Alan payı (%)")
        ax.set_title(f"Puan dağılımı · çok uygun (> 0,75): %{100 * (build > 0.75).mean():.0f}", loc="left")
        show(fig)
    st.caption("Uygunluk puanı düz, sağlam (çıplak/kayalık), su toplamayan, kışın güneş alan ve ufku açık yerleri öne çıkarır. "
               "Zemin sağlamlığı yüzey örtüsü sınıfından kabaca atanmıştır; taşıma gücü için jeoteknik sondaj gerekir.")


# --------------------------------------------------------------------------------------
# MARS: ortak yardımcılar
# --------------------------------------------------------------------------------------
MARS_SVC = {
    "sulu": "Hydrous_Mineral_Detections_OMEGA_CRISM_clon0/MapServer/0",
    "sulu_sinif": "Aquieous_Mineral_Distribution_Options_clon0/MapServer/0",
    "klorur": "Chloride_Salt_bearing_Material_THEMIS_clon0/MapServer/0",
    "jeoloji": "SIM3292_Global_Geology/MapServer/0",
    "kumul": "Dune_Field/MapServer/0",
    "vadi": "Hynek_Valley_Networks/MapServer/0",
    "buzul_vadi": "Mars_Fans_Glaciers_Gullies/MapServer/0",
    "buzul": "Mars_Fans_Glaciers_Gullies/MapServer/1",
    "yelpaze": "Mars_Fans_Glaciers_Gullies/MapServer/2",
    "sel_oyuntusu": "Mars_Fans_Glaciers_Gullies/MapServer/3",
    "rsl": "RSLs_Gullies_New_IcyCraters_clon0/MapServer/0",
    "kesif_bolgesi": "EZMars/MapServer/0",
    "inis_bolgesi": "EZMars/MapServer/2",
}
MINERAL_COLORS = {"phyllosilicate": "#2ca25f", "silica": "#fdae61", "sulfate": "#d7191c", "carbonate": "#2b83ba",
                  "chloride": "#f1e05a", "zeolite": "#9e6ab8"}
MINERAL_TR = {"phyllosilicate": "Fillosilikat (kil)", "silica": "Sulu silika", "sulfate": "Sülfat",
              "carbonate": "Karbonat", "chloride": "Klorür", "zeolite": "Zeolit"}


def mfeat(key, bbox, geometry=True):
    """Öznitelik katmanını güvenli biçimde çeker; hata olursa boş liste döner."""
    try:
        return mars_features(MARS_SVC[key], tuple(bbox), geometry)
    except Exception:
        return []


def mlayer(layer, bbox, shape, smooth=True):
    try:
        return trek_image(layer, tuple(bbox), int(shape[1]), int(shape[0]), smooth)
    except Exception:
        return None


def mvalue(layer, bbox, shape, kind="rainbow"):
    im = mlayer(layer, bbox, shape)
    if im is None:
        return None
    return rainbow_value(im) if kind == "rainbow" else gray_value(im)


def draw_lines(ax, feats, **kw):
    for f in feats:
        for path in (f.get("g") or {}).get("paths", []):
            p = np.asarray(path)
            ax.plot(p[:, 0], p[:, 1], **kw)
            kw.pop("label", None)


def draw_rings(ax, feats, **kw):
    for f in feats:
        for ring in (f.get("g") or {}).get("rings", []):
            p = np.asarray(ring)
            ax.plot(p[:, 0], p[:, 1], **kw)
            kw.pop("label", None)


def clip_axes(ax, bbox):
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])


def base_map(bbox, img, title, figsize=(5.2, 4.6), gray=False):
    """Viking / THEMIS altlığı üzerinde çizim için eksen."""
    fig, ax = _axes(bbox, figsize=figsize)
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]
    if img is not None:
        if gray:
            ax.imshow(img[..., :3].mean(axis=-1), extent=ext, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img[..., :3], extent=ext)
    ax.set_title(title, loc="left")
    return fig, ax


def insolation(lat_deg, ls_deg, body):
    """Atmosfer üstü günlük ortalama güneş ışınımı [W/m²]; ls = güneş boylamı (0° = kuzey ilkbahar ekinoksu)."""
    b = BODIES[body]
    phi = math.radians(lat_deg)
    dec = math.asin(math.sin(math.radians(b["obliq"])) * math.sin(math.radians(ls_deg)))
    r = (1 - b["ecc"] ** 2) / (1 + b["ecc"] * math.cos(math.radians(ls_deg - b["ls_peri"])))
    x = -math.tan(phi) * math.tan(dec)
    h0 = math.pi if x <= -1 else (0.0 if x >= 1 else math.acos(x))
    return b["S0"] / r ** 2 / math.pi * (h0 * math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.sin(h0))


def mars_pressure(z):
    """Ortalama yüzey basıncı [Pa]: referans yüzeyde 610 Pa, ölçek yüksekliği 10,8 km."""
    return 610.0 * np.exp(-np.asarray(z, dtype="float64") / 10_800.0)


def resize_to(a, shape):
    if a.shape == tuple(shape):
        return a
    out = ndi.zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)[:shape[0], :shape[1]]
    if out.shape != tuple(shape):
        out = np.pad(out, ((0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1])), mode="edge")
    return out


def rel_note():
    st.caption("TES haritaları Mars Trek'te renk ölçekli görüntü olarak sunulur; uygulama rengi 0–1 arası **göreli** "
               "değere çevirir (0 = ölçeğin en düşüğü, 1 = en yükseği). Fiziksel birim değildir ve TES pikseli yaklaşık "
               "20 km olduğundan küçük alanlarda harita bulanık görünür.")


# --------------------------------------------------------------------------------------
# MARS 1: jeoloji, mineraloji ve yüzey malzemeleri
# --------------------------------------------------------------------------------------
def mars_geology(bbox, n, dem, px):
    shape = dem.shape
    viking = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, shape)
    geo = mfeat("jeoloji", bbox)
    aq = mfeat("sulu_sinif", bbox)
    hyd = mfeat("sulu", bbox)
    chl = mfeat("klorur", bbox)
    grid, names = rasterize_polygons(geo, bbox, shape, "Unit") if geo else (np.zeros(shape, "int16"), [])
    desc = {str(f["a"].get("Unit")): str(f["a"].get("UnitDesc")) for f in geo}
    cls_count = pd.Series([str(f["a"].get("Class")) for f in aq]).value_counts() if aq else pd.Series(dtype=int)
    area = np.bincount(grid.ravel(), minlength=len(names) + 1)[1:] * px * px / 1e6

    m = st.columns(5)
    m[0].metric("Jeolojik birim sayısı", f"{len(names)}")
    m[1].metric("Baskın birim", names[int(np.argmax(area))] if len(names) else "—")
    m[2].metric("Sulu mineral tespiti (sınıflı)", f"{len(aq)}")
    m[3].metric("OMEGA / CRISM tespiti", f"{len(hyd)}")
    m[4].metric("Klorür tuzu alanı", f"{len(chl)}")

    st.subheader("Yüzey görünümü ve jeolojik birimler")
    c1, c2 = st.columns(2)
    with c1:
        if viking is not None:
            show(base_map(bbox, viking, "Viking renk mozaiği (232 m)")[0])
    with c2:
        if len(names):
            pal = plt.cm.tab20(np.linspace(0, 1, 20))
            cls = {i + 1: (f"{nm} — {desc.get(nm, '')[:38]}", matplotlib.colors.to_hex(pal[i % 20])) for i, nm in enumerate(names)}
            show(class_fig(grid.astype("uint8"), bbox, "Jeoloji haritası (USGS SIM 3292)", cls))
        else:
            st.info("Jeoloji haritası alınamadı.")
    if len(names):
        order = np.argsort(area)
        fig, ax = plt.subplots(figsize=(10, 0.42 * len(names) + 1.6), layout="constrained")
        pal = plt.cm.tab20(np.linspace(0, 1, 20))
        ax.barh([f"{names[i]} — {desc.get(names[i], '')[:60]}" for i in order], 100 * area[order] / max(area.sum(), 1e-9),
                color=[pal[i % 20] for i in order], edgecolor="#555", lw=0.5)
        ax.set_xlabel("Alan payı (%)")
        ax.set_title("Jeolojik birimlerin alan payı", loc="left")
        ax.tick_params(axis="y", labelsize=7.5)
        show(fig)
        st.caption("Birim kodlarında ilk harfler yaşı verir: N = Noakiyen (≈ 4,1–3,7 milyar yıl), H = Hesperiyen, "
                   "A = Amazoniyen (en genç); e/m/l = erken/orta/geç. Sonraki harfler türü: h = yayla, v = volkanik, "
                   "t = geçiş, i = çarpma, a = önlük.")

    st.subheader("Mineraloji — TES (ASU)")
    tes = {"Yüksek-Ca piroksen": mvalue("TES_Clinopyroxene", bbox, shape),
           "Plajiyoklaz": mvalue("TES_Plagioclase", bbox, shape),
           "Tabaka silikat / yüksek-Si cam": mvalue("TES_Glass_Clay", bbox, shape)}
    tes = {k: v for k, v in tes.items() if v is not None and np.isfinite(v).any()}
    cols = st.columns(3)
    for col, (name, arr) in zip(cols, tes.items()):
        with col:
            show(map_fig(arr, bbox, name, cmap="turbo", label="göreli bolluk (0–1)", vmin=0, vmax=1)[0])
    alb = mvalue("Mars_MGS_TES_Albedo_mosaic_global_7410m", bbox, shape, "gray")
    c1, c2 = st.columns(2)
    with c1:
        if tes:
            fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
            data = [v[np.isfinite(v)] for v in tes.values()]
            bp = ax.boxplot(data, orientation="horizontal", showfliers=False, patch_artist=True, widths=0.6, medianprops={"color": "#222"}) \
                if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 10) else \
                ax.boxplot(data, vert=False, showfliers=False, patch_artist=True, widths=0.6, medianprops={"color": "#222"})
            for patch, c in zip(bp["boxes"], ["#3b8f6b", "#c9a227", "#7a6ab0"]):
                patch.set_facecolor(c)
            ax.set_yticks(range(1, len(tes) + 1), list(tes), fontsize=7.5)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Göreli bolluk (0–1)")
            ax.set_title("TES mineral bolluklarının dağılımı", loc="left")
            show(fig)
    with c2:
        if alb is not None and np.isfinite(alb).any():
            show(map_fig(alb, bbox, "TES yüzey albedosu", cmap="gray", label="göreli parlaklık (0–1)")[0])
    rel_note()
    st.caption("Piroksen ve plajiyoklaz bazaltik kabuğun ana mineralleridir; koyu, tozsuz bölgelerde yüksektir. "
               "'Tabaka silikat / yüksek-Si cam' sınıfı kil ya da ayrışmış volkanik cam içeren yüzeyleri gösterir.")

    st.subheader("Sulu (hidratlı) mineraller — OMEGA + CRISM")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_map(bbox, viking, "Sulu mineral tespitleri", gray=True)
        seen = set()
        for f in aq:
            c = str(f["a"].get("Class"))
            g = f.get("g") or {}
            if "x" in g:
                ax.plot(g["x"], g["y"], "o", ms=5, mfc=MINERAL_COLORS.get(c, "#cccccc"), mec="#111", mew=0.5,
                        label=None if c in seen else MINERAL_TR.get(c, c))
                seen.add(c)
        hx, hy = feature_points(hyd)
        if hx.size:
            ax.plot(hx, hy, "s", ms=6, mfc="none", mec="#00e5ff", mew=1.1, label="OMEGA/CRISM küresel tarama")
        draw_rings(ax, chl, color="#f1e05a", lw=1.2, label="Klorür tuzu" if chl else None)
        clip_axes(ax, bbox)
        if seen or hx.size or chl:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
        show(fig)
    with c2:
        if len(cls_count) or len(hyd):
            rows = [(MINERAL_TR.get(k, k), int(v), MINERAL_COLORS.get(k, "#bbbbbb")) for k, v in cls_count.items()]
            if chl:
                rows.append(("Klorür tuzu (THEMIS)", len(chl), MINERAL_COLORS["chloride"]))
            inst = pd.Series([str(f["a"].get("Instrument")) for f in hyd]).value_counts()
            rows += [(f"Küresel tarama — {k}", int(v), "#00b8d4") for k, v in inst.items()]
            rows.sort(key=lambda r: r[1])
            show(share_bar([r[0] for r in rows], [float(r[1]) for r in rows], [r[2] for r in rows],
                           "Tespit sayısı (mineral sınıfı ve alet)", xlabel="Tespit sayısı", figsize=(5.2, 3.8)))
        else:
            st.info("Bu alanda kayıtlı sulu mineral tespiti yok. Genç volkanik ve tozla örtülü bölgelerde bu beklenen bir sonuçtur.")
    st.caption("Noktalar Carter vd. (2013) küresel OMEGA + CRISM taramasından ve sınıflandırılmış sulu mineral "
               "dağılımından gelir; ESA'nın 2022'de yayımladığı sulu mineral haritası aynı ekibin bu çalışmayı genişletmiş "
               "sürümüdür. Kil mineralleri eski, uzun süreli ıslak dönemi; sülfat ve klorür tuzları suyun buharlaşarak "
               "çekildiği daha geç dönemi gösterir.")
    with st.expander("ESA küresel sulu mineral haritası (2022, referans görsel)"):
        st.image(ESA_HYDRATED_PNG, caption="Global map of hydrated minerals on Mars — ESA / Mars Express (OMEGA) ve NASA / MRO (CRISM)")
        st.markdown(f"Kaynak: [esa.int]({ESA_HYDRATED_PAGE}). ESA bu haritayı yalnızca görsel olarak yayımlıyor; "
                    "sayısal karşılığı için yukarıdaki nokta verisi kullanıldı.")


# --------------------------------------------------------------------------------------
# MARS 2: topoğrafya
# --------------------------------------------------------------------------------------
def mars_topo(bbox, n, dem, px):
    tab_topo(bbox, dem, px, None, None)
    st.subheader("MOLA yüzey pürüzlülüğü")
    rough = mlayer("mola_roughness", bbox, dem.shape)
    c1, c2 = st.columns(2)
    with c1:
        if rough is not None:
            show(base_map(bbox, rough, "MOLA pürüzlülük bileşimi (0,6 / 2,4 / 9,2 km)")[0])
    with c2:
        if rough is not None:
            fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
            for ch, col, lab in ((2, "#2b6cb0", "0,6 km (mavi kanal)"), (1, "#3a9a5c", "2,4 km (yeşil)"), (0, "#c0392b", "9,2 km (kırmızı)")):
                ax.hist(rough[..., ch].ravel() / 255, bins=40, histtype="step", color=col, lw=1.3, label=lab, density=True)
            ax.set_xlabel("Göreli pürüzlülük (0–1)")
            ax.set_ylabel("Yoğunluk")
            ax.set_title("Ölçeğe göre pürüzlülük dağılımı", loc="left")
            ax.legend(fontsize=7, frameon=False)
            show(fig)
    st.caption("Bileşimde her renk kanalı farklı bir taban uzunluğundaki pürüzlülüğü verir (Kreslavsky & Head): parlak ve "
               "beyaza yakın yerler her ölçekte engebeli, koyu yerler pürüzsüzdür. Yükseklik modeli MOLA lazer altimetresi "
               f"ile HRSC stereo verisinin 200 m'lik harmanıdır; bu panelde piksel ≈ {px:.0f} m.")


# --------------------------------------------------------------------------------------
# MARS 3: sıcaklık, aydınlatma ve çevresel aşırılıklar
# --------------------------------------------------------------------------------------
def mars_thermal(bbox, n, dem, px):
    lat = (bbox[1] + bbox[3]) / 2
    shape = dem.shape
    day = mvalue("THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018", bbox, shape, "gray")
    night = mvalue("THEMIS_NightIR_ControlledMosaics_100m_v2_oct2018", bbox, shape, "gray")
    ti = mvalue("TES_Thermal_Inertia", bbox, shape)
    ls_axis = np.arange(0, 361, 5)
    q_mars = np.array([insolation(lat, l, "mars") for l in ls_axis])
    q_earth = np.array([insolation(lat, l, "earth") for l in ls_axis])
    teq = lambda q: (max(q, 1e-6) * (1 - 0.25) / (0.95 * 5.670e-8)) ** 0.25 - 273.15

    m = st.columns(5)
    m[0].metric("Yıllık ort. güneş ışınımı", f"{q_mars.mean():.0f} W/m²", help="Atmosfer üstü, günlük ortalama.")
    m[1].metric("En yüksek günlük ışınım", f"{q_mars.max():.0f} W/m²")
    m[2].metric("En düşük günlük ışınım", f"{q_mars.min():.0f} W/m²")
    m[3].metric("Tahmini günlük ort. sıcaklık (en sıcak)", f"{teq(q_mars.max()):.0f} °C")
    m[4].metric("Tahmini günlük ort. sıcaklık (en soğuk)", f"{teq(q_mars.min()):.0f} °C")
    st.caption("Bu dört kaynakta kırpılabilir bir **ölçülmüş sıcaklık** ürünü yoktur. Sıcaklık değerleri, güneş ışınımından "
               "hesaplanan kaba bir ışınım dengesi tahminidir (albedo 0,25; yayıcılık 0,95; atmosfer ve ısı depolama yok). "
               "Gerçekte ince atmosfer nedeniyle gündüz–gece farkı 60–100 °C'yi bulur.")

    st.subheader("Mevsimlik güneş ışınımı")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        ax.plot(ls_axis, q_mars, color="#c1440e", lw=1.8, label=f"Mars, {lat:.1f}° enlem")
        ax.plot(ls_axis, q_earth, color="#2b6cb0", lw=1.2, ls="--", label="Dünya, aynı enlem")
        for x, t in ((0, "ilkbahar"), (90, "yaz"), (180, "sonbahar"), (270, "kış")):
            ax.axvline(x, color="#bbb", lw=0.6)
            ax.annotate(t, (x + 4, ax.get_ylim()[0]), fontsize=6.5, color="#777", va="bottom")
        ax.set_xlabel("Güneş boylamı Ls (°) — kuzey yarımküre mevsimleri")
        ax.set_ylabel("Günlük ort. ışınım (W/m²)")
        ax.set_title("Yıl boyunca atmosfer üstü güneş ışınımı", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        t_m = [teq(q) for q in q_mars]
        ax.fill_between(ls_axis, t_m, min(t_m) - 5, color="#c1440e", alpha=0.2, lw=0)
        ax.plot(ls_axis, t_m, color="#c1440e", lw=1.6)
        ax.axhline(0, color="#333", lw=0.7)
        ax.set_xlabel("Güneş boylamı Ls (°)")
        ax.set_ylabel("Tahmini günlük ort. sıcaklık (°C)")
        ax.set_title("Işınım dengesi sıcaklığı (kaba tahmin)", loc="left")
        show(fig)
    st.caption("Mars yörüngesi belirgin biçimde basıktır: günberi güney yazına (Ls ≈ 251°) denk gelir, bu yüzden güney "
               "yazları kısa ve sıcak, kuzey yazları uzun ve ılıktır. Mars'a ulaşan güneş enerjisi Dünya'nın yaklaşık %43'üdür.")

    st.subheader("Isıl davranış — THEMIS (ASU) ve TES")
    cols = st.columns(3)
    with cols[0]:
        if day is not None:
            show(map_fig(day, bbox, "THEMIS gündüz kızılötesi", cmap="gray", label="göreli parlaklık")[0])
    with cols[1]:
        if night is not None:
            show(map_fig(night, bbox, "THEMIS gece kızılötesi", cmap="magma", label="göreli parlaklık")[0])
    with cols[2]:
        if ti is not None and np.isfinite(ti).any():
            show(map_fig(ti, bbox, "TES ısıl atalet", cmap="turbo", label="göreli (0–1)", vmin=0, vmax=1)[0])
    c1, c2 = st.columns(2)
    with c1:
        if day is not None and night is not None:
            ok = np.isfinite(day) & np.isfinite(night)
            if ok.sum() > 500:
                fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
                hb = ax.hexbin(day[ok], night[ok], gridsize=45, cmap="inferno_r", mincnt=1, bins="log")
                fig.colorbar(hb, ax=ax, label="piksel sayısı (log)")
                ax.set_xlabel("Gündüz parlaklığı (→ gündüz sıcak)")
                ax.set_ylabel("Gece parlaklığı (→ gece sıcak)")
                ax.set_title("Gündüz – gece ısıl davranışı", loc="left")
                show(fig)
    with c2:
        if night is not None and np.isfinite(night).any():
            slope, _ = slope_normals(dem, px)
            fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
            bins = np.linspace(dem.min(), dem.max(), 16)
            which = np.clip(np.digitize(dem.ravel(), bins) - 1, 0, 14)
            mid = (bins[:-1] + bins[1:]) / 2
            nn = night.ravel()
            mn = [np.nanmean(nn[which == b]) if (which == b).any() and np.isfinite(nn[which == b]).any() else np.nan for b in range(15)]
            ax.plot(mn, mid, "-o", ms=3, color="#7b2d8e", lw=1.3)
            ax.set_xlabel("Gece kızılötesi parlaklığı (göreli)")
            ax.set_ylabel("Yükseklik (m)")
            ax.set_title("Yüksekliğe göre gece ısıl parlaklığı", loc="left")
            show(fig)
    st.caption("Gece sıcak (parlak) görünen yüzeyler ısıyı tutan kaya ve sıkı malzemedir; gece soğuk (koyu) olanlar gevşek toz "
               "ve kumdur. THEMIS mozaikleri görsel kontrast için normalleştirilmiştir, kelvin değeri taşımaz.")

    st.subheader("Aydınlanma ve gölge")
    with st.spinner("Güneş geometrisi ve arazi gölgeleri hesaplanıyor…"):
        hs, es, ts = illumination(dem, px, lat, 172)
        hw, ew, tw = illumination(dem, px, lat, 355)
    m = st.columns(4)
    m[0].metric("Öğle güneş yüksekliği · kuzey yazı", f"{sun_position(lat, 172, 12)[0]:.1f}°")
    m[1].metric("Öğle güneş yüksekliği · kuzey kışı", f"{sun_position(lat, 355, 12)[0]:.1f}°")
    m[2].metric("Ort. doğrudan güneş · Ls 90°", f"{hs.mean():.1f} sa")
    m[3].metric("Ort. doğrudan güneş · Ls 270°", f"{hw.mean():.1f} sa")
    cols = st.columns(3)
    with cols[0]:
        show(map_fig(hw, bbox, "Doğrudan güneş süresi — Ls 270°", cmap="YlOrBr_r", label="saat / sol", vmin=0, vmax=float(hw.max()))[0])
    with cols[1]:
        show(map_fig(ew, bbox, "Göreli ışınım — Ls 270°", cmap="RdYlBu_r", label="düz zemine oran", vmin=0, vmax=2)[0])
    with cols[2]:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
        ax.plot(ts["saat"], ts["golge_%"], color="#e9a23b", lw=1.6, label="Ls 90° (kuzey yazı)")
        ax.plot(tw["saat"], tw["golge_%"], color="#34558b", lw=1.6, label="Ls 270° (kuzey kışı)")
        ax.set_xlabel("Yerel güneş saati")
        ax.set_ylabel("Gölgedeki alan (%)")
        ax.set_title("Gün içinde gölgede kalan alan", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    st.caption("Aydınlanma Mars'ın 25,2°'lik eksen eğikliğiyle, yükseklik modeli üzerinde hesaplanmıştır. Saatler Mars günü "
               "(sol ≈ 24 sa 39 dk) 24 eşit parçaya bölünerek verilir; toz fırtınalarının ışığı kesmesi hesaba katılmaz.")


# --------------------------------------------------------------------------------------
# MARS 4: su, buz, permafrost ve kaynaklar
# --------------------------------------------------------------------------------------
def mars_water(bbox, n, dem, px):
    lat = (bbox[1] + bbox[3]) / 2
    shape = dem.shape
    hydro_dem_section(bbox, dem, px, None)

    st.subheader("Eski ve güncel su izleri (haritalanmış yapılar)")
    viking = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, shape)
    layers = [("Vadi ağları", "vadi", "#2b83ba", "line"), ("Alüvyon yelpazeleri", "yelpaze", "#fdae61", "pt"),
              ("Sel oyuntuları", "sel_oyuntusu", "#d7191c", "pt"), ("Buzul benzeri yapılar", "buzul", "#a6d9f7", "pt"),
              ("Buzul vadileri", "buzul_vadi", "#5e9bd1", "pt"), ("RSL / yeni buzlu kraterler", "rsl", "#f1e05a", "pt"),
              ("Sulu mineral tespitleri", "sulu_sinif", "#2ca25f", "pt")]
    data = {lab: mfeat(key, bbox) for lab, key, _, _ in layers}
    valley_km = sum(float(f["a"].get("tot_length") or 0) for f in data["Vadi ağları"]) / 1000
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_map(bbox, viking, "Su ve buzla ilişkili yapılar", gray=True)
        for lab, key, col, kind in layers:
            feats = data[lab]
            if not feats:
                continue
            if kind == "line":
                draw_lines(ax, feats, color=col, lw=1.2, label=lab)
            else:
                x, y = feature_points(feats)
                if x.size:
                    ax.plot(x, y, "o", ms=5, mfc=col, mec="#111", mew=0.5, ls="none", label=lab)
        clip_axes(ax, bbox)
        if any(data.values()):
            ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
        show(fig)
    with c2:
        rows = [(lab, float(len(data[lab])), col) for lab, _, col, _ in layers]
        rows.sort(key=lambda r: r[1])
        show(share_bar([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                       "Alandaki kayıt sayısı", xlabel="Kayıt sayısı", figsize=(5.2, 3.6)))
        st.caption(f"Haritalanmış vadi ağı uzunluğu (alanla kesişen ağların toplamı): **{valley_km:.0f} km**.")
    st.caption("Vadi ağları (Hynek vd.) ve alüvyon yelpazeleri eski akarsu etkinliğini; buzul benzeri yapılar orta "
               "enlemlerde toz altında korunmuş su buzunu; sel oyuntuları ve yinelenen yamaç çizgileri (RSL) görece güncel, "
               "kökeni tartışmalı akış izlerini gösterir. Kayıt yoksa bu, katalogda yapı bulunmadığı anlamına gelir.")

    st.subheader("Sıvı su ve buz için koşullar")
    P = mars_pressure(dem)
    hw, _, _ = illumination(dem, px, lat, 355 if lat >= 0 else 172)
    m = st.columns(4)
    m[0].metric("Ortalama yüzey basıncı", f"{P.mean():.0f} Pa")
    m[1].metric("Üçlü nokta üstü alan (> 611 Pa)", f"%{100 * (P > 611).mean():.0f}")
    m[2].metric("Kış gündönümünde < 4 sa güneş", f"%{100 * (hw < 4).mean():.1f}")
    m[3].metric("Enlem", f"{abs(lat):.1f}° {'K' if lat >= 0 else 'G'}")
    if abs(lat) >= 40:
        st.success("**Yüzeye yakın yer buzu beklenir.** Yaklaşık 40–45° enlemin kutba doğru tarafında, toz örtüsünün "
                   "birkaç santimetre–metre altında su buzu yıl boyu kararlıdır (nötron spektrometresi ve radar bulguları).")
    else:
        st.info("**Yüzeye yakın kalıcı buz beklenmez.** Bu enlemde yer buzu günümüz ikliminde kararlı değildir; su "
                "büyük oranda hidratlı minerallerde bağlı ya da derinde bulunur. Buz ancak yerel soğuk tuzaklarda kalabilir.")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        ax.hist(dem.ravel(), bins=50, color="#b5651d", edgecolor="none", weights=np.full(dem.size, 100 / dem.size))
        ztp = -10_800 * math.log(611 / 610)
        ax.axvline(ztp, color="#2b6cb0", lw=1.2, ls="--")
        ax.annotate("sıvı su için basınç\nyeterli  ←", (ztp, ax.get_ylim()[1] * 0.85), fontsize=7, color="#2b6cb0", ha="right",
                    xytext=(-4, 0), textcoords="offset points")
        ax.set_xlabel("Yükseklik (m, MOLA)")
        ax.set_ylabel("Alan payı (%)")
        sec = ax.secondary_xaxis("top", functions=(lambda z: 610 * np.exp(-z / 10_800), lambda p: -10_800 * np.log(np.maximum(p, 1e-6) / 610)))
        sec.set_xlabel("Ortalama basınç (Pa)", fontsize=8)
        sec.tick_params(labelsize=7)
        ax.set_title("Yükseklik ve basınç dağılımı", loc="left", pad=28)
        show(fig)
    with c2:
        show(map_fig(hw, bbox, "Soğuk tuzak adayları: kış gündönümü güneş süresi", cmap="cividis", label="saat / sol",
                     vmin=0, vmax=float(hw.max()))[0])
    st.caption("Suyun sıvı kalabilmesi için basıncın üçlü noktanın (611 Pa) üzerinde olması gerekir; bu yaklaşık MOLA sıfır "
               "seviyesinin altındaki alanlarda sağlanır, ama sıcaklık çoğu zaman yine de donma noktasının altındadır.")

    st.subheader("Potansiyel kaynak özeti")
    aq = data["Sulu mineral tespitleri"]
    cls = pd.Series([MINERAL_TR.get(str(f["a"].get("Class")), str(f["a"].get("Class"))) for f in aq]).value_counts()
    c1, c2 = st.columns(2)
    with c1:
        if len(cls):
            show(share_bar(list(cls.index[::-1]), [float(v) for v in cls.values[::-1]], ["#2ca25f"] * len(cls),
                           "Su bağlayan mineral yüzlekleri", xlabel="Tespit sayısı"))
        else:
            st.info("Kayıtlı sulu mineral yüzleği yok; su kaynağı olarak yer buzu ya da atmosfer değerlendirilmelidir.")
    with c2:
        hs, _, _ = illumination(dem, px, lat, 172)
        hw2, _, _ = illumination(dem, px, lat, 355)
        fig, ax = plt.subplots(figsize=(5.2, 3.2), layout="constrained")
        ax.hist(hs.ravel(), bins=40, color="#e9a23b", alpha=0.8, label="Ls 90°", weights=np.full(hs.size, 100 / hs.size))
        ax.hist(hw2.ravel(), bins=40, color="#34558b", alpha=0.8, label="Ls 270°", weights=np.full(hw2.size, 100 / hw2.size))
        ax.set_xlabel("Doğrudan güneş süresi (saat / sol)")
        ax.set_ylabel("Alan payı (%)")
        ax.set_title("Güneş enerjisi kaynağı", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    st.caption("Killer ağırlıkça yaklaşık %5–15, çok sulu sülfatlar %20'ye varan oranda su içerebilir ve ısıtılarak "
               "çıkarılabilir; bu nedenle sulu mineral yüzlekleri yerinde kaynak kullanımı (ISRU) için hedeftir.")


# --------------------------------------------------------------------------------------
# MARS 5: toz, rüzgâr, erozyon
# --------------------------------------------------------------------------------------
def mars_processes(bbox, n, dem, px):
    shape = dem.shape
    slope, _ = slope_normals(dem, px)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
    viking = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, shape)
    dci = mvalue("tes_ruffdust", bbox, shape)
    dust = mvalue("TES_Dust", bbox, shape)
    ti = mvalue("TES_Thermal_Inertia", bbox, shape)
    alb = mvalue("Mars_MGS_TES_Albedo_mosaic_global_7410m", bbox, shape, "gray")
    dunes = mfeat("kumul", bbox)
    with st.spinner("Açıklık ve akış birikimi hesaplanıyor…"):
        opn = openness(dem, px, radius_m=max(1500, 4 * px))
        depth, lab, dep = find_depressions(dem, px, bbox)
        sinks = tuple(int(r["_r"]) * dem.shape[1] + int(r["_c"]) for _, r in dep.iterrows())
        fr = flow_routing(dem, px, max(0.3, round(150 * px * px / 1e6, 2)), sinks)
    ls, _, ero, spi = erosion_index(fr["acc_km2"], slope, np.zeros(shape), px)
    fin = lambda a: a is not None and np.isfinite(a).any()

    m = st.columns(5)
    m[0].metric("Toz örtüsü (TES, göreli)", f"{np.nanmean(dust):.2f}" if fin(dust) else "—")
    m[1].metric("Isıl atalet (göreli)", f"{np.nanmean(ti):.2f}" if fin(ti) else "—",
                help="Düşük değer gevşek toz, yüksek değer kaya ya da sıkı zemin.")
    m[2].metric("Kumul alanı", f"{len(dunes)}")
    m[3].metric("Ortalama açıklık", f"{opn.mean():.1f}°")
    m[4].metric("Korunaklı alan (açıklık < 80°)", f"%{100 * (opn < 80).mean():.1f}")

    st.subheader("Toz")
    named = {}
    cols = st.columns(3)
    for col, (name, arr, cmap, labl) in zip(cols, (("TES toz örtüsü", dust, "turbo", "göreli (0–1)"),
                                                 ("TES toz indeksi (Ruff & Christensen)", dci, "turbo", "göreli (1 = tozlu uç)"),
                                                 ("TES ısıl atalet", ti, "turbo", "göreli (0–1)"))):
        with col:
            if fin(arr):
                named[name] = arr
                show(map_fig(arr, bbox, name, cmap=cmap, label=labl, vmin=0, vmax=1)[0])
    if fin(alb):
        named["TES albedo"] = alb
    if named:
        show(hist_grid(named, np.ones(shape, bool), color="#c1440e", ncols=4))
    rel_note()
    st.caption("Tozla kaplı yüzeyler parlaktır (yüksek albedo) ve ısıyı tutmaz (düşük ısıl atalet); tozsuz, kayalık ya da "
               "kumlu yüzeyler koyudur ve ısıl ataleti yüksektir. İki toz haritasında da kırmızı uç tozlu yüzeydir; bu yönü, "
               "küresel haritada bilinen tozlu bölgelerin (Tharsis, Arabia) kırmızı görünmesinden çıkardım.")

    st.subheader("Rüzgâr: kumullar ve açıklık")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_map(bbox, viking, "Kumul alanları (Mars Küresel Kumul Veritabanı)", gray=True)
        draw_rings(ax, dunes, color="#ffd23f", lw=1.3)
        clip_axes(ax, bbox)
        show(fig)
    with c2:
        az, wt = [], []
        for f in dunes:
            for k_az, k_n in (("SF_1_Az", "SF_1_Count"), ("SF_2_Azimu", "SF_2_Count"), ("SF_3_Azimu", "SF_3_Count"), ("SF_4_Azimu", "SF_4_Count")):
                a_, n_ = f["a"].get(k_az), f["a"].get(k_n)
                if a_ and n_:
                    az.append(float(a_))
                    wt.append(float(n_))
        fig = plt.figure(figsize=(5.2, 4.2), layout="constrained")
        ax = fig.add_subplot(projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        if az:
            h, _ = np.histogram(np.asarray(az) % 360, bins=np.arange(0, 361, 22.5), weights=wt)
            ax.bar(np.radians(np.arange(11.25, 360, 22.5)), h, width=np.radians(20), color="#e0a800", edgecolor="#444", lw=0.5)
            ax.set_title("Kumul kayma yüzü yönleri (rüzgârın estiği yön)", loc="left", pad=14, fontsize=9)
        else:
            ax.set_title("Bu alanda ölçülmüş kumul kayma yüzü yok", loc="left", pad=14, fontsize=9)
        ax.tick_params(labelsize=7)
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(opn, bbox, "Topoğrafik açıklık", cmap="viridis", label="derece (90 = tam açık)",
                     vmin=float(np.percentile(opn, 1)), vmax=90)[0])
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        bins = np.linspace(dem.min(), dem.max(), 16)
        which = np.clip(np.digitize(dem.ravel(), bins) - 1, 0, 14)
        mid = (bins[:-1] + bins[1:]) / 2
        mo = [opn.ravel()[which == b].mean() if (which == b).any() else np.nan for b in range(15)]
        ax.plot(mo, mid, "-o", ms=3, color="#2f6b3c", lw=1.3)
        ax.set_xlabel("Ortalama açıklık (°)")
        ax.set_ylabel("Yükseklik (m)")
        ax.set_title("Yüksekliğe göre açıklık", loc="left")
        show(fig)
    st.caption("Kumulların kayma yüzleri hâkim rüzgârın estiği yöne bakar; gül, katalogdaki ölçümlerin sayısıyla "
               "ağırlıklandırılmıştır. Ölçülmüş rüzgâr hızı bu kaynaklarda yoktur. Açıklığı düşük krater ve kanyon tabanları "
               "kumun ve tozun biriktiği tuzaklardır.")

    st.subheader("Yamaç süreçleri ve erozyon potansiyeli")
    ecls = np.digitize(ero, [1, 5, 15, 40]).astype("uint8") + 1
    ERO = {1: ("Çok düşük", "#eef3e2"), 2: ("Düşük", "#cfe0a8"), 3: ("Orta", "#f1cf6b"), 4: ("Yüksek", "#e08a3c"), 5: ("Çok yüksek", "#b2301e")}
    c1, c2 = st.columns(2)
    with c1:
        show(class_fig(ecls, bbox, "Yamaç uzunluğu–dikliği etkeni (LS)", ERO))
    with c2:
        v, cnt = np.unique(ecls, return_counts=True)
        show(share_bar([ERO[int(x)][0] for x in v], list(100 * cnt / cnt.sum()), [ERO[int(x)][1] for x in v],
                       "LS sınıflarının alan payı"))
    st.caption("Günümüz Mars'ında yağış olmadığından bu harita su erozyonunu değil, **kütle hareketine ve eski akışa en yatkın "
               "yamaçları** gösterir: dik ve uzun yamaçlar heyelan, kaya düşmesi ve toz çığı için en elverişli yerlerdir.")


# --------------------------------------------------------------------------------------
# MARS 6: biyoloji, yaşanabilirlik, biyolojik imzalar
# --------------------------------------------------------------------------------------
def mars_biology(bbox, n, dem, px):
    st.caption("Mars'ta bugüne dek doğrulanmış bir biyolojik imza yoktur ve bu dört kaynakta biyolojik veri bulunmaz. "
               "Bu bölüm, **geçmişte yaşanabilir olmuş** ve biyolojik imzayı **koruyabilecek** ortamların dolaylı "
               "göstergelerini bir araya getirir.")
    lat = (bbox[1] + bbox[3]) / 2
    shape = dem.shape
    slope, _ = slope_normals(dem, px)
    viking = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, shape)
    aq = mfeat("sulu_sinif", bbox)
    hyd = mfeat("sulu", bbox)
    chl = mfeat("klorur", bbox)
    val = mfeat("vadi", bbox)
    rsl = mfeat("rsl", bbox)
    dust = mvalue("TES_Dust", bbox, shape)
    hw = resize_to(illumination(dem, px, lat, 355 if lat >= 0 else 172)[0], shape)
    P = mars_pressure(dem)

    # suyla ilişkili noktalara uzaklık
    mask = np.zeros(shape, bool)
    for feats in (aq, hyd, rsl):
        x, y = feature_points(feats)
        if x.size:
            c = np.clip(((x - bbox[0]) / (bbox[2] - bbox[0]) * shape[1]).astype(int), 0, shape[1] - 1)
            r = np.clip(((bbox[3] - y) / (bbox[3] - bbox[1]) * shape[0]).astype(int), 0, shape[0] - 1)
            mask[r, c] = True
    for f in val:
        for path in (f.get("g") or {}).get("paths", []):
            p = np.asarray(path)
            c = ((p[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * shape[1]).astype(int)
            r = ((bbox[3] - p[:, 1]) / (bbox[3] - bbox[1]) * shape[0]).astype(int)
            ok = (c >= 0) & (c < shape[1]) & (r >= 0) & (r < shape[0])
            mask[r[ok], c[ok]] = True
    d_water = ndi.distance_transform_edt(~mask) * px / 1000 if mask.any() else np.full(shape, 1e4)
    s_water = np.exp(-d_water / 25)                       # sulu mineral / vadi izine yakınlık
    s_press = np.clip((P - 400) / 400, 0, 1)             # yüksek basınç (alçak arazi)
    s_light = np.clip(hw / max(float(hw.max()), 1e-6), 0, 1)
    s_dust = 1 - np.nan_to_num(dust, nan=0.5) if dust is not None else np.full(shape, 0.5)
    hab = 0.45 * s_water + 0.25 * s_press + 0.15 * s_light + 0.15 * s_dust

    NICHE = {1: ("Diğer yüzeyler", "#d9c79a"), 2: ("Sulu mineral / vadi çevresi (≤ 15 km)", "#2ca25f"),
             3: ("Alçak, yüksek basınçlı taban (> 611 Pa)", "#2b6cb0"), 4: ("Soğuk gölgeli yamaç", "#6a8fc7"),
             5: ("Kalın toz örtüsü", "#e8b89a")}
    niche = np.full(shape, 1, dtype="uint8")
    if dust is not None:
        niche[np.nan_to_num(dust, nan=0) > 0.6] = 5
    niche[P > 611] = 3
    niche[(hw < 4) & (slope > 8)] = 4
    niche[d_water <= 15] = 2

    cls = pd.Series([str(f["a"].get("Class")) for f in aq]).value_counts() if aq else pd.Series(dtype=int)
    m = st.columns(5)
    m[0].metric("Kil (fillosilikat) tespiti", f"{int(cls.get('phyllosilicate', 0))}")
    m[1].metric("Karbonat / silika tespiti", f"{int(cls.get('carbonate', 0) + cls.get('silica', 0))}")
    m[2].metric("Tuz (sülfat + klorür)", f"{int(cls.get('sulfate', 0)) + len(chl)}")
    m[3].metric("Vadi ağı kaydı", f"{len(val)}")
    m[4].metric("Ort. yaşanabilirlik puanı", f"{hab.mean():.2f}")

    st.subheader("Geçmiş yaşanabilirlik ve korunma potansiyeli")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_map(bbox, viking, "Biyolojik imza arama hedefleri", gray=True)
        draw_lines(ax, val, color="#2b83ba", lw=1.1, label="Vadi ağı" if val else None)
        seen = set()
        for f in aq:
            c = str(f["a"].get("Class"))
            g = f.get("g") or {}
            if "x" in g:
                ax.plot(g["x"], g["y"], "o", ms=5, mfc=MINERAL_COLORS.get(c, "#ccc"), mec="#111", mew=0.5,
                        label=None if c in seen else MINERAL_TR.get(c, c))
                seen.add(c)
        draw_rings(ax, chl, color="#f1e05a", lw=1.2, label="Klorür tuzu" if chl else None)
        x, y = feature_points(rsl)
        if x.size:
            ax.plot(x, y, "^", ms=6, mfc="#ff5fd2", mec="#111", mew=0.5, ls="none", label="RSL / yeni krater")
        clip_axes(ax, bbox)
        if seen or val or chl or x.size:
            ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
        show(fig)
    with c2:
        rows = [("Kil (organik maddeyi korur)", float(cls.get("phyllosilicate", 0)), MINERAL_COLORS["phyllosilicate"]),
                ("Karbonat (nötr pH'lı su)", float(cls.get("carbonate", 0)), MINERAL_COLORS["carbonate"]),
                ("Sulu silika (mikrofosil korur)", float(cls.get("silica", 0)), MINERAL_COLORS["silica"]),
                ("Sülfat (asidik / tuzlu su)", float(cls.get("sulfate", 0)), MINERAL_COLORS["sulfate"]),
                ("Klorür tuzu (buharlaşma havzası)", float(len(chl)), MINERAL_COLORS["chloride"])]
        show(share_bar([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], [r[2] for r in rows][::-1],
                       "Korunma ortamı göstergeleri", xlabel="Tespit sayısı", figsize=(5.2, 3.8)))
    st.caption("Astrobiyolojide en değerli hedefler, uzun süre sıvı suyla temas etmiş ve organik maddeyi hapsedebilen "
               "ince taneli killer, karbonatlar ve silika çökelleridir; Jezero ve Mawrth Vallis bu nedenle seçilmiştir.")

    st.subheader("Yaşanabilirlik puanı ve nişler")
    c1, c2 = st.columns(2)
    with c1:
        show(map_fig(hab, bbox, "Göreli yaşanabilirlik puanı", cmap="YlGn", label="0 = elverişsiz, 1 = elverişli", vmin=0, vmax=1)[0])
    with c2:
        parts = {"Su izine yakınlık": s_water, "Basınç (alçak arazi)": s_press, "Kış ışığı": s_light, "Az toz": s_dust}
        w = [0.45, 0.25, 0.15, 0.15]
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        ax.barh(list(parts), [float(np.mean(v)) for v in parts.values()], color=["#2ca25f", "#2b6cb0", "#e9a23b", "#b98b6a"],
                edgecolor="#555", lw=0.5)
        for i, (v, wi) in enumerate(zip(parts.values(), w)):
            ax.text(float(np.mean(v)), i, f" {np.mean(v):.2f}  (ağırlık {wi:.2f})", va="center", fontsize=7)
        ax.set_xlim(0, 1.3)
        ax.set_xlabel("Alan ortalaması (0–1)")
        ax.set_title("Puanı sınırlayan etkenler", loc="left")
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        show(class_fig(niche, bbox, "Ortam sınıfları (kural tabanlı)", NICHE))
    with c2:
        v, cnt = np.unique(niche, return_counts=True)
        order = np.argsort(cnt)
        show(share_bar([NICHE[int(x)][0].split(" (")[0] for x in v[order]], list(100 * cnt[order] / cnt.sum()),
                       [NICHE[int(x)][1] for x in v[order]], "Ortam sınıflarının alan payı", figsize=(5.2, 3.6)))
    st.caption("Yüzey, morötesi ışınım, oksitleyici toprak ve düşük basınç nedeniyle bugün bilinen yaşam için elverişsizdir; "
               "güncel yaşanabilirlik adayları yeraltındadır (mağaralar, buz, tuzlu sızıntılar). Puan ağırlıklarını ben "
               "belirledim; göreli sıralama verir.")


# --------------------------------------------------------------------------------------
# MARS 7: mağaralar, lav tüpleri, yeraltı
# --------------------------------------------------------------------------------------
def mars_subsurface(bbox, n, dem, px):
    shape = dem.shape
    geo = mfeat("jeoloji", bbox)
    volc = [f for f in geo if "volcan" in str(f["a"].get("UnitDesc", "")).lower()]
    lava = rasterize_polygons(volc, bbox, shape)[0] > 0 if volc else np.zeros(shape, bool)
    units = sorted({f"{f['a'].get('Unit')} ({f['a'].get('UnitDesc')})" for f in volc})
    st.caption(f"Mars'taki bilinen lav tüpü tavan pencereleri 100–250 m çapındadır; bu panelin {px:.0f} m'lik ızgarası "
               "çoğunu **çözemez**. Volkanik zemin jeoloji haritasındaki volkanik birimlerden alınmıştır"
               + (f": {', '.join(units)}." if units else "; bu alanda volkanik birim yok."))
    subsurface_body(bbox, dem, px, lava, None, lava_name="Volkanik birim")
    st.caption("Mars'ta bu piksel boyunda bulunan kapalı çöküntülerin büyük çoğunluğu **çarpma krateridir**; çökme çukuru "
               "sayılabilmeleri için volkanik birim içinde, bir akış hattı boyunca dizili ve kenar yükseltisiz olmaları gerekir.")
    night = mvalue("THEMIS_NightIR_ControlledMosaics_100m_v2_oct2018", bbox, shape, "gray")
    if night is not None and np.isfinite(night).any():
        st.subheader("Gece ısıl izi — THEMIS")
        k = max(3, int(round(4000 / px)) | 1)
        base = np.where(np.isfinite(night), night, np.nanmedian(night))
        resid = np.where(np.isfinite(night), base - ndi.uniform_filter(base, size=k, mode="nearest"), np.nan)
        lim = float(max(0.02, np.nanpercentile(np.abs(resid), 98)))
        c1, c2 = st.columns(2)
        with c1:
            show(map_fig(night, bbox, "THEMIS gece kızılötesi", cmap="magma", label="göreli parlaklık")[0])
        with c2:
            show(map_fig(resid, bbox, "Yerel gece ısıl farkı (4 km çevreye göre)", cmap="RdBu_r",
                         label="kırmızı = çevresinden sıcak", vmin=-lim, vmax=lim)[0])
        st.caption("Mağara ağızları ve derin çukurlar gece çevresinden sıcak kalır; Arsia Mons'taki tavan pencereleri ilk "
                   "kez bu yolla, THEMIS gece görüntülerinde fark edilmiştir. Burada görülen sıcak lekelerin çoğu ise "
                   "kayalık yüzeylerdir; ayırt etmek için yüksek çözünürlüklü görüntü gerekir.")


# --------------------------------------------------------------------------------------
# MARS 8: iniş, hareketlilik, navigasyon, inşaat
# --------------------------------------------------------------------------------------
def mars_operations(bbox, n, dem, px):
    lat = (bbox[1] + bbox[3]) / 2
    shape = dem.shape
    tab_operations(bbox, n, 0, dem, px, None)
    st.subheader("Mars'a özgü iniş kısıtları")
    slope, _ = slope_normals(dem, px)
    ti = mvalue("TES_Thermal_Inertia", bbox, shape)
    viking = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, shape)
    ez = mfeat("kesif_bolgesi", bbox)
    lz = mfeat("inis_bolgesi", bbox)
    ok_elev = dem < -1000
    ok_slope = slope <= 5
    ok_ti = (np.nan_to_num(ti, nan=0.5) > 0.2) if ti is not None else np.ones(shape, bool)
    ok_lat = abs(lat) <= 30
    allok = ok_elev & ok_slope & ok_ti & ok_lat
    m = st.columns(5)
    m[0].metric("Yükseklik < −1 km", f"%{100 * ok_elev.mean():.0f}", help="Paraşütle yavaşlama için yeterli atmosfer.")
    m[1].metric("Eğim ≤ 5°", f"%{100 * ok_slope.mean():.0f}")
    m[2].metric("Sıkı zemin (ısıl atalet)", f"%{100 * ok_ti.mean():.0f}", help="Çok düşük ısıl atalet kalın, gevşek toz demektir.")
    m[3].metric("Enlem ±30° içinde", "Evet" if ok_lat else "Hayır", help="Güneş enerjisi ve ısıl koşullar için.")
    m[4].metric("Tüm ölçütleri sağlayan", f"%{100 * allok.mean():.0f}")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_map(bbox, viking, "Tüm ölçütleri sağlayan alan ve önerilen bölgeler", gray=True)
        ax.imshow(np.where(allok, 1.0, np.nan), extent=[bbox[0], bbox[2], bbox[1], bbox[3]], cmap=ListedColormap(["#3a9a5c"]),
                  alpha=0.5, interpolation="nearest")
        draw_rings(ax, ez, color="#ffd23f", lw=1.3, label="Keşif bölgesi (NASA EZ)" if ez else None)
        x, y = feature_points(lz)
        if x.size:
            ax.plot(x, y, "*", ms=11, mfc="#ff5fd2", mec="#111", ls="none", label="Önerilen iniş noktası")
        clip_axes(ax, bbox)
        if ez or x.size:
            ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
        show(fig)
    with c2:
        names = ["Yükseklik < −1 km", "Eğim ≤ 5°", "Sıkı zemin", "Hepsi birlikte"]
        vals = [100 * ok_elev.mean(), 100 * ok_slope.mean(), 100 * ok_ti.mean(), 100 * allok.mean()]
        show(share_bar(names[::-1], [float(v) for v in vals[::-1]], ["#3a9a5c", "#8d7b6c", "#c9813a", "#2b6cb0"],
                       "İniş ölçütlerini sağlayan alan", figsize=(5.2, 3.2)))
        if lz:
            st.dataframe(pd.DataFrame([{"Ad": f["a"].get("name"), "Enlem": f["a"].get("latitude"),
                                        "Boylam": f["a"].get("longitude"), "Yükseklik (m)": f["a"].get("elevation")} for f in lz]))
    st.caption("Bugüne kadarki bütün başarılı inişler MOLA sıfır seviyesinin belirgin biçimde altına yapılmıştır; ince "
               "atmosfer yüksek arazide aracı yeterince yavaşlatamaz. Keşif bölgeleri NASA'nın 2015 insanlı iniş alanı "
               "çalıştayında önerilen 100 km yarıçaplı alanlardır.")


# --------------------------------------------------------------------------------------
# KARŞILAŞTIRMA
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def dem_summary(dem, px, bbox: tuple, body: str):
    """Yalnızca yükseklik modelinden türetilen, iki gezegende de aynı yöntemle hesaplanan ölçütler."""
    set_body(body)
    lat = (bbox[1] + bbox[3]) / 2
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    depth, lab, dep = find_depressions(dem, px, bbox)
    nl = int(lab.max())
    if nl:  # tablo ilk 15 çöküntüyle sınırlı olduğundan sayım burada ayrıca yapılır
        ii = np.arange(1, nl + 1)
        n_dep = int(((ndi.maximum(depth, lab, ii) >= 5.0) & (ndi.sum(np.ones_like(depth), lab, ii) * px * px / 1e6 >= 0.01)).sum())
    else:
        n_dep = 0
    sinks = tuple(int(r["_r"]) * dem.shape[1] + int(r["_c"]) for _, r in dep.iterrows())
    fr = flow_routing(dem, px, max(0.3, round(150 * px * px / 1e6, 2)), sinks)
    opn = openness(dem, px, radius_m=max(1500, 4 * px))
    hs, _, _ = illumination(dem, px, lat, 172)
    hw, _, _ = illumination(dem, px, lat, 355)
    k = max(3, int(round(150 / px)) | 1)
    safe = (slope <= 5) & (ndi.maximum_filter(slope, size=k) <= 8)
    dist, sites = landing_sites(safe, px, bbox)
    area = dem.size * px * px / 1e6
    q = [insolation(lat, l, body) for l in range(0, 360, 5)]
    return {
        "Alan (km²)": area, "Piksel boyu (m)": px,
        "En alçak nokta (m)": float(dem.min()), "En yüksek nokta (m)": float(dem.max()),
        "Rölyef (m)": float(dem.max() - dem.min()), "Ortalama eğim (°)": float(slope.mean()),
        "Eğimin %90'lık değeri (°)": float(np.percentile(slope, 90)), "Dik alan, eğim > 15° (%)": float(100 * (slope > 15).mean()),
        "Hipsometrik integral": float((dem.mean() - dem.min()) / max(dem.max() - dem.min(), 1e-6)),
        "Kapalı çöküntü sayısı": float(n_dep),
        "En derin çöküntü (m)": float(dep["Derinlik (m)"].max()) if len(dep) else 0.0,
        "En geniş çöküntü çapı (m)": float(dep["Eşdeğer çap (m)"].max()) if len(dep) else 0.0,
        "Drenaj yoğunluğu (km/km²)": float(sum(fr["length_km"].values()) / area),
        "Kapalı havza payı (%)": float(100 * (fr["closed"] > 0).mean()),
        "Ortalama açıklık (°)": float(opn.mean()), "Korunaklı alan, açıklık < 80° (%)": float(100 * (opn < 80).mean()),
        "Öğle güneş yüksekliği, kuzey yaz gündönümü (°)": sun_position(lat, 172, 12)[0], "Öğle güneş yüksekliği, kuzey kış gündönümü (°)": sun_position(lat, 355, 12)[0],
        "Doğrudan güneş, kuzey yaz gündönümü (sa)": float(hs.mean()), "Doğrudan güneş, kuzey kış gündönümü (sa)": float(hw.mean()),
        "Kuzey kış gündönümünde < 4 sa güneş alan (%)": float(100 * (hw < 4).mean()),
        "Yıllık ort. güneş ışınımı (W/m²)": float(np.mean(q)), "En düşük günlük ışınım (W/m²)": float(np.min(q)),
        "İnişe uygun eğim, ≤ 5° (%)": float(100 * safe.mean()), "Tekerlekli araca açık, ≤ 15° (%)": float(100 * (slope <= 15).mean()),
        "Geçilemez, > 25° (%)": float(100 * (slope > 25).mean()),
        "En büyük engelsiz daire yarıçapı (m)": float(sites["Engelsiz yarıçap (m)"].iloc[0]) if len(sites) else 0.0,
        "Lav tüpüne elverişli eğim, 0,3–6° (%)": float(100 * ((slope >= 0.3) & (slope <= 6)).mean()),
        "_slope": slope, "_lat": lat,
    }


COMPARE_GROUPS = {
    "Topoğrafya": ["Alan (km²)", "Piksel boyu (m)", "En alçak nokta (m)", "En yüksek nokta (m)", "Rölyef (m)", "Ortalama eğim (°)",
                   "Eğimin %90'lık değeri (°)", "Dik alan, eğim > 15° (%)", "Hipsometrik integral", "Kapalı çöküntü sayısı",
                   "En derin çöküntü (m)", "En geniş çöküntü çapı (m)"],
    "Sıcaklık ve aydınlanma": ["Öğle güneş yüksekliği, kuzey yaz gündönümü (°)", "Öğle güneş yüksekliği, kuzey kış gündönümü (°)", "Doğrudan güneş, kuzey yaz gündönümü (sa)",
                               "Doğrudan güneş, kuzey kış gündönümü (sa)", "Kuzey kış gündönümünde < 4 sa güneş alan (%)", "Yıllık ort. güneş ışınımı (W/m²)",
                               "En düşük günlük ışınım (W/m²)"],
    "Su ve drenaj": ["Drenaj yoğunluğu (km/km²)", "Kapalı havza payı (%)"],
    "Toz ve rüzgâr": ["Ortalama açıklık (°)", "Korunaklı alan, açıklık < 80° (%)"],
    "Yeraltı": ["Lav tüpüne elverişli eğim, 0,3–6° (%)"],
    "İniş ve hareketlilik": ["İnişe uygun eğim, ≤ 5° (%)", "Tekerlekli araca açık, ≤ 15° (%)", "Geçilemez, > 25° (%)",
                             "En büyük engelsiz daire yarıçapı (m)"],
}


def site_extras(place, bbox, n, year, dem, px, body):
    """Yalnızca o gezegenin veri setlerinden gelen ek ölçütler: {başlık: [(ad, değer metni)]}"""
    out = {}
    add = lambda grp, name, val: out.setdefault(grp, []).append((name, val))
    if body == "earth":
        try:
            ser, y, _ = modis_best_year(bbox, year)
            add("Sıcaklık", f"En sıcak gündüz yüzey sıcaklığı ({y})", f"{ser['gunduz'].max():.1f} °C")
            add("Sıcaklık", "En soğuk gece yüzey sıcaklığı", f"{ser['gece'].min():.1f} °C")
            add("Sıcaklık", "Yıllık ort. yüzey sıcaklığı", f"{np.nanmean((ser['gunduz'] + ser['gece']) / 2):.1f} °C")
            add("Buz", "Gece donu görülen 8 günlük dönem", f"{int((ser['gece'] < 0).sum())}")
        except Exception:
            pass
        try:
            s2 = get_s2(bbox, n, year)
            ndvi, rock, water, veg, bad, _ = spectral_products(s2)
            add("Biyoloji", "Ortalama NDVI", f"{np.nanmean(ndvi):.2f}")
            add("Biyoloji", "Fotosentetik örtü (NDVI > 0,30)", f"%{100 * veg.mean():.0f}")
            add("Yüzey", "Çıplak yüzey payı", f"%{100 * rock.mean():.0f}")
        except Exception:
            pass
        try:
            occ, tr = get_jrc(bbox, n)
            add("Su", "Açık su yüzeyi (≥ %75 kalıcı)", f"{(occ >= 75).sum() * px * px / 1e6:.2f} km²")
            add("Su", "Kaybolan su yüzeyi (1984–2020)", f"{np.isin(tr, [3, 6]).sum() * px * px / 1e6:.2f} km²")
        except Exception:
            pass
        add("Atmosfer", "Ortalama yüzey basıncı (yaklaşık)", f"{1013.25 * math.exp(-float(dem.mean()) / 8434):.0f} hPa")
    else:
        P = mars_pressure(dem)
        add("Atmosfer", "Ortalama yüzey basıncı (yaklaşık)", f"{P.mean() / 100:.1f} hPa")
        add("Su", "Üçlü nokta üstü alan (> 611 Pa)", f"%{100 * (P > 611).mean():.0f}")
        aq = mfeat("sulu_sinif", bbox, False)
        add("Mineraloji", "Sulu mineral tespiti", f"{len(aq)}")
        cls = pd.Series([str(f["a"].get("Class")) for f in aq]).value_counts() if aq else pd.Series(dtype=int)
        if len(cls):
            add("Mineraloji", "Baskın sulu mineral sınıfı", MINERAL_TR.get(cls.index[0], cls.index[0]))
        val = mfeat("vadi", bbox, False)
        add("Su", "Haritalanmış vadi ağı uzunluğu", f"{sum(float(f['a'].get('tot_length') or 0) for f in val) / 1000:.0f} km")
        add("Buz", "Buzul benzeri yapı kaydı", f"{len(mfeat('buzul', bbox, False))}")
        add("Toz", "Kumul alanı kaydı", f"{len(mfeat('kumul', bbox, False))}")
        for name, layer in (("TES toz örtüsü (göreli 0–1)", "TES_Dust"), ("TES ısıl atalet (göreli 0–1)", "TES_Thermal_Inertia")):
            v = mvalue(layer, bbox, dem.shape)
            if v is not None and np.isfinite(v).any():
                add("Toz", name, f"{np.nanmean(v):.2f}")
        geo = mfeat("jeoloji", bbox, False)
        add("Jeoloji", "Jeolojik birim sayısı", f"{len({f['a'].get('Unit') for f in geo})}")
        add("Sıcaklık", "Ölçülmüş yüzey sıcaklığı", "bu kaynaklarda yok")
        add("Biyoloji", "Bitki örtüsü / biyolojik imza", "yok")
    return out


def load_place(place, mars_region, n, year):
    """Karşılaştırma için bir mekânın yükseklik modelini ve tanımını yükler."""
    if place == "Mars":
        set_body("mars")
        reg = MARS_REGIONS[mars_region]
        bbox = tuple(reg["bbox"])
        dem, px = get_mars_dem(bbox, n)
        return {"ad": f"Mars — {mars_region}", "body": "mars", "bbox": bbox, "dem": dem, "px": px, "reg": reg}
    set_body("earth")
    reg = REGIONS[place]
    bbox = tuple(reg["bbox"])
    dem, px = get_dem(bbox, n)
    return {"ad": PLACE_LABEL.get(place, place), "body": "earth", "bbox": bbox, "dem": dem, "px": px, "reg": reg}


def fmt_num(v):
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}"


def compare_view(place_a, place_b, mars_region, n, year):
    CA, CB = "#1f77b4", "#c1440e"
    with st.spinner("İki mekânın verileri yükleniyor…"):
        A = load_place(place_a, mars_region, n, year)
        B = load_place(place_b, mars_region, n, year)
        A["s"] = dem_summary(A["dem"], A["px"], A["bbox"], A["body"])
        B["s"] = dem_summary(B["dem"], B["px"], B["bbox"], B["body"])
    st.title("Karşılaştırma")
    st.markdown(f"<span style='color:{CA}'>■</span> **{A['ad']}** &nbsp;↔&nbsp; <span style='color:{CB}'>■</span> **{B['ad']}**",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, S in ((c1, A), (c2, B)):
        with col:
            st.markdown(f"**{S['reg']['tip']}** — {S['reg']['ozet']}")
    st.caption(f"Piksel boyu: {A['ad']} ≈ {A['px']:.0f} m, {B['ad']} ≈ {B['px']:.0f} m. Eğim, pürüzlülük ve çöküntü sayısı "
               "piksel boyuna bağlıdır: kaba ızgara araziyi daha düz ve daha az çukurlu gösterir. Büyük farklarda sıralamaya "
               "güvenin, küçük farkları yorumlamayın.")

    st.subheader("Görünüm")
    for title, draw in (("Yükseklik", "elev"), ("Eğim", "slope"), ("Geçilebilirlik", "mob")):
        cols = st.columns(2)
        for col, S in zip(cols, (A, B)):
            with col:
                set_body(S["body"])
                dem, px, bbox, slope = S["dem"], S["px"], S["bbox"], S["s"]["_slope"]
                if draw == "elev":
                    shade = LightSource(315, 45).hillshade(dem, vert_exag=2, dx=px, dy=px)
                    show(map_fig(dem, bbox, f"{title} — {S['ad']}", cmap=ELEV_CMAP, label="m", vmin=float(dem.min()),
                                 vmax=float(dem.max()), shade=shade)[0])
                elif draw == "slope":
                    show(map_fig(slope, bbox, f"{title} — {S['ad']}", cmap="magma_r", label="derece", vmin=0, vmax=30)[0])
                else:
                    MOB = {1: ("≤ 5° iniş", "#3a9a5c"), 2: ("≤ 15° tekerlekli", "#b7d98a"), 3: ("≤ 25° paletli", "#f1cf6b"),
                           4: ("> 25° geçilemez", "#b2301e")}
                    show(class_fig((np.digitize(slope, [5, 15, 25]) + 1).astype("uint8"), bbox, f"{title} — {S['ad']}", MOB))

    st.subheader("Ortak ölçütler")
    rows = []
    for grp, keys in COMPARE_GROUPS.items():
        for k in keys:
            rows.append({"Başlık": grp, "Özellik": k, A["ad"]: fmt_num(A["s"][k]), B["ad"]: fmt_num(B["s"][k])})
    st.dataframe(pd.DataFrame(rows).set_index(["Başlık", "Özellik"]), height=min(38 * len(rows) + 40, 900))
    st.caption("Bu tablodaki her değer iki mekânda da aynı yöntemle, yalnızca yükseklik modelinden ve gezegenin yörünge "
               "sabitlerinden hesaplanmıştır; bu yüzden doğrudan karşılaştırılabilir.")

    c1, c2 = st.columns(2)
    with c1:
        radar = [("Eğim", "Ortalama eğim (°)", 20), ("Rölyef", "Rölyef (m)", 6000), ("İnişe uygunluk", "İnişe uygun eğim, ≤ 5° (%)", 100),
                 ("Kapalı havza", "Kapalı havza payı (%)", 50), ("Aralık güneşi", "Doğrudan güneş, kuzey kış gündönümü (sa)", 12),
                 ("Güneş ışınımı", "Yıllık ort. güneş ışınımı (W/m²)", 450), ("Açıklık", "Ortalama açıklık (°)", 90),
                 ("Drenaj", "Drenaj yoğunluğu (km/km²)", 3)]
        ang = np.linspace(0, 2 * np.pi, len(radar), endpoint=False)
        fig = plt.figure(figsize=(5.2, 4.8), layout="constrained")
        ax = fig.add_subplot(projection="polar")
        for S, col in ((A, CA), (B, CB)):
            v = [min(1.0, max(0.0, S["s"][k] / ref)) for _, k, ref in radar]
            ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=col, lw=1.8, label=S["ad"])
            ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=col, alpha=0.15)
        ax.set_xticks(ang, [r[0] for r in radar], fontsize=7.5)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""])
        ax.set_ylim(0, 1)
        ax.set_title("Genel profil (sabit ölçeklere göre 0–1)", loc="left", pad=16)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=False)
        show(fig)
    with c2:
        keys = ["İnişe uygun eğim, ≤ 5° (%)", "Tekerlekli araca açık, ≤ 15° (%)", "Dik alan, eğim > 15° (%)", "Kapalı havza payı (%)",
                "Korunaklı alan, açıklık < 80° (%)", "Kuzey kış gündönümünde < 4 sa güneş alan (%)", "Lav tüpüne elverişli eğim, 0,3–6° (%)"]
        y = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(5.2, 4.8), layout="constrained")
        ax.barh(y + 0.2, [A["s"][k] for k in keys], height=0.38, color=CA, label=A["ad"])
        ax.barh(y - 0.2, [B["s"][k] for k in keys], height=0.38, color=CB, label=B["ad"])
        for i, k in enumerate(keys):
            ax.text(A["s"][k], i + 0.2, f" {A['s'][k]:.0f}", va="center", fontsize=6.5)
            ax.text(B["s"][k], i - 0.2, f" {B['s'][k]:.0f}", va="center", fontsize=6.5)
        ax.set_yticks(y, [k.replace(" (%)", "") for k in keys], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 112)
        ax.set_xlabel("Alan payı (%)")
        ax.set_title("Alan paylarıyla karşılaştırma", loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1))
        show(fig)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        for S, col in ((A, CA), (B, CB)):
            zs = np.sort(S["dem"].ravel())[::-1]
            ax.plot(np.linspace(0, 1, zs.size), (zs - zs[-1]) / max(zs[0] - zs[-1], 1e-6), color=col, lw=1.6,
                    label=f"{S['ad']} (HI = {S['s']['Hipsometrik integral']:.2f})")
        ax.plot([0, 1], [1, 0], color="#999", lw=0.7, ls=":")
        ax.set_xlabel("Göreli alan")
        ax.set_ylabel("Göreli yükseklik")
        ax.set_title("Hipsometrik eğriler", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        for S, col in ((A, CA), (B, CB)):
            s = np.sort(S["s"]["_slope"].ravel())
            ax.plot(s, np.linspace(0, 100, s.size), color=col, lw=1.6, label=S["ad"])
        for thr in (5, 15, 25):
            ax.axvline(thr, color="#999", lw=0.7, ls="--")
        ax.set_xlim(0, 35)
        ax.set_xlabel("Eğim (°)")
        ax.set_ylabel("Bu eğimin altındaki alan (%)")
        ax.set_title("Kümülatif eğim eğrileri", loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        show(fig)
    c1, c2 = st.columns(2)
    with c1:
        ls_axis = np.arange(0, 361, 5)
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        for S, col in ((A, CA), (B, CB)):
            ax.plot(ls_axis, [insolation(S["s"]["_lat"], l, S["body"]) for l in ls_axis], color=col, lw=1.6,
                    label=f"{S['ad']} ({S['s']['_lat']:.1f}°)")
        ax.set_xlabel("Güneş boylamı Ls (°) — 90° kuzey yazı, 270° kuzey kışı")
        ax.set_ylabel("Günlük ort. ışınım (W/m²)")
        ax.set_title("Yıl boyunca atmosfer üstü güneş ışınımı", loc="left")
        ax.legend(fontsize=7, frameon=False)
        show(fig)
    with c2:
        keys = ["Rölyef (m)", "En derin çöküntü (m)", "En geniş çöküntü çapı (m)", "En büyük engelsiz daire yarıçapı (m)"]
        y = np.arange(len(keys))
        fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
        ax.barh(y + 0.2, [max(A["s"][k], 1) for k in keys], height=0.38, color=CA, label=A["ad"])
        ax.barh(y - 0.2, [max(B["s"][k], 1) for k in keys], height=0.38, color=CB, label=B["ad"])
        ax.set_xscale("log")
        ax.set_yticks(y, [k.replace(" (m)", "") for k in keys], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("metre (logaritmik)")
        ax.set_title("Boyut ölçütleri", loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.16))
        show(fig)

    st.subheader("Yalnızca o mekânın verisinde bulunan özellikler")
    with st.spinner("Mekâna özgü veriler yükleniyor…"):
        ex = []
        for S, place in ((A, place_a), (B, place_b)):
            set_body(S["body"])
            ex.append(site_extras(place, S["bbox"], n, year, S["dem"], S["px"], S["body"]))
    cols = st.columns(2)
    for col, S, e in zip(cols, (A, B), ex):
        with col:
            st.markdown(f"**{S['ad']}**")
            df = pd.DataFrame([{"Başlık": g, "Özellik": nme, "Değer": v} for g, items in e.items() for nme, v in items])
            if len(df):
                st.dataframe(df.set_index(["Başlık", "Özellik"]))
    st.caption("Bu ölçütler iki tarafta farklı aletlerden gelir (Dünya: Sentinel-2, MODIS, JRC; Mars: TES, OMEGA/CRISM, "
               "haritalanmış yapılar) ve birimleri aynı değildir; yan yana okunmak içindir, doğrudan oranlanmamalıdır.")


# --------------------------------------------------------------------------------------
# 2D SİMÜLASYON — Mars, Elysium – Cerberus Fossae
# --------------------------------------------------------------------------------------
SIM_REGION = "Elysium – Cerberus Fossae"
SIM_GEO_TR = {"lAvf": "Geç Amazoniyen volkanik alan birimi", "lAv": "Geç Amazoniyen volkanik birim",
              "AHv": "Amazoniyen–Hesperiyen volkanik birim", "HNt": "Hesperiyen–Noakiyen geçiş birimi",
              "AHi": "Amazoniyen–Hesperiyen çarpma birimi", "Htu": "Hesperiyen geçiş birimi (çökel)"}
SIM_LF = {1: "Vadi / çukur tabanı", 2: "Alt yamaç", 3: "Düzlük", 4: "Orta yamaç", 5: "Üst yamaç", 6: "Sırt / zirve"}


def _b64(arr, dtype):
    import base64
    return base64.b64encode(np.ascontiguousarray(arr.astype(dtype)).tobytes()).decode("ascii")


def _jpeg_b64(rgb, quality=86):
    import base64
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb.astype("uint8")).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def sim_prepare(n: int = 450):
    """Simülasyonun tüm verisini hazırlar: altlık görüntüler, yükseklik türevleri, iniş noktası, hedefler, rotalar."""
    set_body("mars")
    bbox = tuple(MARS_REGIONS[SIM_REGION]["bbox"])
    dem, px = get_mars_dem(bbox, n)
    H, W = dem.shape
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    lf = landforms(dem, px, slope, radius_m=max(300, 3 * px))
    rough_img = mlayer("mola_roughness", bbox, dem.shape)
    rough = rough_img[..., :3].mean(axis=-1) if rough_img is not None else np.zeros(dem.shape)
    geo = mfeat("jeoloji", bbox)
    grid, names = rasterize_polygons(geo, bbox, dem.shape, "Unit") if geo else (np.zeros(dem.shape, "int16"), [])
    desc = {str(f["a"].get("Unit")): str(f["a"].get("UnitDesc")) for f in geo}

    # altlık görüntüler (NASA Mars Trek)
    S = 1100
    imgs = {}
    vk = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, (S, S))
    th = mlayer("THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018", bbox, (S, S))
    if th is not None:
        imgs["themis"] = _jpeg_b64(th[..., :3])
    if vk is not None:
        imgs["viking"] = _jpeg_b64(vk[..., :3])
    shaded = LightSource(315, 45).shade(dem, cmap=ELEV_CMAP, vert_exag=3, dx=px, dy=px, blend_mode="soft")
    from PIL import Image
    imgs["elev"] = _jpeg_b64(np.asarray(Image.fromarray((shaded[..., :3] * 255).astype("uint8")).resize((S, S), Image.BILINEAR)))
    if not ("themis" in imgs or "viking" in imgs):
        raise RuntimeError("NASA Mars Trek görüntüleri alınamadı.")

    def to_rc(lon, lat):
        return (int(np.clip((bbox[3] - lat) / (bbox[3] - bbox[1]) * H, 0, H - 1)),
                int(np.clip((lon - bbox[0]) / (bbox[2] - bbox[0]) * W, 0, W - 1)))

    # başlangıç: NASA'nın önerdiği iniş noktası; yoksa en büyük engelsiz düz alan
    lz = mfeat("inis_bolgesi", bbox)
    px_, py_ = feature_points(lz)
    if px_.size:
        start_rc = to_rc(float(px_[0]), float(py_[0]))
        start_name = f"{lz[0]['a'].get('name', 'Cerberus')} iniş noktası (NASA önerisi)"
    else:
        safe = (slope <= 5) & (ndi.maximum_filter(slope, size=3) <= 8)
        _, sites = landing_sites(safe, px, bbox)
        start_rc = (int(sites["_r"].iloc[0]), int(sites["_c"].iloc[0]))
        start_name = "En geniş engelsiz düz alan"

    # hedefler
    depth, lab, dep = find_depressions(dem, px, bbox)
    _, peaks = find_peaks(dem, px, bbox)
    targets = []
    r0, c0 = to_rc(162.78, 9.06)  # Cerberus Fossae çatlağının orta kesimi: en yakın çukur tabanına oturtulur
    k = max(3, int(round(6000 / px)))
    tpi = dem - ndi.uniform_filter(dem, size=2 * k + 1, mode="nearest")
    win = tpi[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1]
    rr, cc = np.unravel_index(int(np.argmin(win)), win.shape)
    targets.append(("Cerberus Fossae çatlağı", "Genç, sismik olarak etkin çatlak sisteminin kenarı ve tabanı",
                    (max(0, r0 - k) + int(rr), max(0, c0 - k) + int(cc))))
    if len(dep):
        targets.append((f"Derin krater ({dep.index[0]})", f"Alanın en derin kapalı çöküntüsü, {dep['Derinlik (m)'].iloc[0]:.0f} m",
                        (int(dep["_r"].iloc[0]), int(dep["_c"].iloc[0]))))
    if len(peaks):
        hi = peaks.sort_values("Zirve (m)", ascending=False).iloc[0]
        targets.append(("Volkanik yükselti zirvesi", f"Alanın en yüksek belirgin tepesi, {hi['Zirve (m)']:.0f} m",
                        (int(hi["_r"]), int(hi["_c"]))))
        rest = peaks[(peaks["Boylam"] - hi["Boylam"]).abs() > 1.0]
        if len(rest):
            e = rest.iloc[0]
            targets.append(("Doğu tepeleri", f"Ovadan {e['Çevreden yükseklik (m)']:.0f} m yükselen mesa / tepe grubu",
                            (int(e["_r"]), int(e["_c"]))))

    colors = ["#ffd23f", "#39d5ff", "#ff6b6b", "#7dff8a", "#d59bff"]
    routes = []
    blocked = np.zeros(dem.shape, bool)
    for i, (name, info, rc) in enumerate(targets):
        pr, pc, _ = least_cost_path(slope, blocked, px, start_rc, rc)
        pr, pc = pr.astype(float), pc.astype(float)
        if len(pr) >= 5:  # köşeleri yumuşat
            ker = np.ones(5) / 5
            pr[2:-2] = np.convolve(pr, ker, mode="valid")
            pc[2:-2] = np.convolve(pc, ker, mode="valid")
        d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(pr), np.diff(pc)) * px)]) / 1000
        ri, ci = np.clip(pr.round().astype(int), 0, H - 1), np.clip(pc.round().astype(int), 0, W - 1)
        z, sl = dem[ri, ci], slope[ri, ci]
        routes.append({"name": name, "info": info, "color": colors[i % len(colors)],
                       "pts": [[round(float(c), 2), round(float(r), 2)] for r, c in zip(pr, pc)],
                       "d": [round(float(x), 3) for x in d], "z": [int(v) for v in z], "sl": [round(float(v), 1) for v in sl],
                       "len": round(float(d[-1]), 1), "maxSlope": round(float(sl.max()), 1),
                       "climb": int(np.clip(np.diff(z), 0, None).sum()), "descent": int(np.clip(-np.diff(z), 0, None).sum()),
                       "steep": round(float(100 * (sl > 15).mean()), 1)})
    return {
        "W": W, "H": H, "px": float(px), "bbox": list(bbox), "img": imgs,
        "dem": _b64(dem, "<i2"), "slope": _b64(np.clip(slope * 4, 0, 255), "u1"), "tri": _b64(np.clip(tri / 2, 0, 255), "u1"),
        "rough": _b64(np.clip(rough, 0, 255), "u1"), "lf": _b64(lf, "u1"), "geo": _b64(np.clip(grid, 0, 255), "u1"),
        "geoNames": [{"unit": nm, "desc": SIM_GEO_TR.get(nm, desc.get(nm, ""))} for nm in names],
        "lfNames": SIM_LF, "start": {"name": start_name, "r": start_rc[0], "c": start_rc[1]}, "routes": routes,
    }


SIM_HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#0b0e14;color:#e8e6e3;font-family:"Source Sans Pro","Segoe UI",system-ui,sans-serif;font-size:14px}
#wrap{display:flex;gap:10px;height:760px;padding:8px}
#main{flex:1;min-width:0;display:flex;flex-direction:column;gap:6px}
#top{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
#top .ttl{font-weight:700;letter-spacing:.04em;color:#ffb38a;margin-right:8px}
button{font:inherit;color:#e8e6e3;background:#1b2130;border:1px solid #39435c;border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{background:#263049}
button.on{background:#c1440e;border-color:#ff8a57;color:#fff}
button.big{font-size:15px;font-weight:700;padding:9px 12px;width:100%}
button.primary{background:#c1440e;border-color:#ff8a57;color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
#cvbox{position:relative;flex:1;min-height:0;border:1px solid #2a3145;border-radius:8px;overflow:hidden;background:#000}
#cv{width:100%;height:100%;display:block;outline:none;cursor:crosshair}
#hint{position:absolute;left:10px;top:10px;background:rgba(11,14,20,.78);border:1px solid #39435c;border-radius:6px;padding:6px 9px;font-size:13px;max-width:70%}
#hint b{color:#ffd23f}
#busy{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(11,14,20,.85);font-size:18px;font-weight:700;color:#ffd23f}
#side{width:318px;flex:none;display:flex;flex-direction:column;gap:8px;overflow-y:auto}
.card{background:#141926;border:1px solid #2a3145;border-radius:8px;padding:9px 10px}
.card h3{margin:0 0 6px 0;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#9aa6c4}
.route{border:1px solid #2a3145;border-radius:7px;padding:7px 8px;margin-bottom:6px;cursor:pointer;background:#10141f}
.route:hover{background:#1a2032}
.route.sel{border-color:#ffd23f;background:#221f12}
.route .nm{font-weight:700;display:flex;align-items:center;gap:6px}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block;flex:none}
.route .st{color:#b9c2d8;font-size:12.5px;margin-top:3px;line-height:1.35}
table{width:100%;border-collapse:collapse}
td{padding:3px 0;border-bottom:1px solid #20263a;vertical-align:top}
td.k{color:#9aa6c4;width:48%}
td.v{font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;font-weight:700;color:#0b0e14}
.small{font-size:12px;color:#9aa6c4;line-height:1.35}
#prof{width:100%;height:96px;display:block}
.row{display:flex;gap:6px}
.row button{flex:1}
kbd{background:#263049;border:1px solid #4a5677;border-bottom-width:2px;border-radius:4px;padding:0 5px;font-family:inherit;font-weight:700;color:#ffd23f}
</style></head><body>
<div id="wrap">
  <div id="main">
    <div id="top">
      <span class="ttl">MARS · ELYSIUM – CERBERUS FOSSAE</span>
      <span class="small">Altlık:</span>
      <span id="bases"></span>
      <span style="flex:1"></span>
      <button id="zout" title="Uzaklaş">−</button><button id="zin" title="Yakınlaş">+</button>
    </div>
    <div id="cvbox"><canvas id="cv" tabindex="0"></canvas><div id="hint"></div><div id="busy">Detaylı analiz hazırlanıyor…</div></div>
  </div>
  <div id="side"></div>
</div>
<script>
const D = __DATA__;
const W = D.W, H = D.H, PXKM = D.px / 1000;
function dec(b64, T){ const s = atob(b64), u = new Uint8Array(s.length); for(let i=0;i<s.length;i++) u[i]=s.charCodeAt(i); return T===Int16Array ? new Int16Array(u.buffer) : u; }
const dem = dec(D.dem, Int16Array), slope = dec(D.slope), tri = dec(D.tri), rough = dec(D.rough), lf = dec(D.lf), geo = dec(D.geo);
const imgs = {}; const baseNames = {themis:"THEMIS kızılötesi", viking:"Viking renk", elev:"Yükseklik"};
let base = D.img.themis ? "themis" : Object.keys(D.img)[0];
for(const k in D.img){ imgs[k] = new Image(); imgs[k].src = D.img[k]; }
const LFCOL = {1:"#6ea8ff",2:"#a9c8f0",3:"#eee8c8",4:"#d8c08a",5:"#e09a63",6:"#ff8a6b"};
const fmt = (v,d=0)=> Number(v).toLocaleString("tr-TR",{minimumFractionDigits:d,maximumFractionDigits:d,useGrouping:false});

// ---- Streamlit bileşen protokolü (derleme gerektirmeyen düz sürüm)
function post(type, data){ window.parent.postMessage(Object.assign({isStreamlitMessage:true, type:type}, data||{}), "*"); }
let restored = false;
window.addEventListener("message", e=>{
  if(!e.data || e.data.type !== "streamlit:render") return;
  const a = e.data.args || {};
  if(!restored){ restored = true;
    if(a.route !== null && a.route !== undefined && D.routes[a.route]){ sel = a.route; s = Math.min(Math.max(a.s||0,0), D.routes[sel].len); stage = "drive"; zoom = 4; }
    buildSide(); }
});
post("streamlit:componentReady", {apiVersion:1});
post("streamlit:setFrameHeight", {height:776});

// ---- durum
let stage = "select", sel = null, s = 0, zoom = 1, heading = 0, moving = false, lastMove = 0;
const keys = {};
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const side = document.getElementById("side"), hint = document.getElementById("hint");

function at(arr, c, r){ const ci = Math.min(W-1, Math.max(0, Math.round(c))), ri = Math.min(H-1, Math.max(0, Math.round(r))); return arr[ri*W+ci]; }
function elevAt(c, r){ // çift doğrusal
  const c0 = Math.min(W-2, Math.max(0, Math.floor(c))), r0 = Math.min(H-2, Math.max(0, Math.floor(r))), fc = c-c0, fr = r-r0;
  const i = r0*W+c0; return dem[i]*(1-fc)*(1-fr) + dem[i+1]*fc*(1-fr) + dem[i+W]*(1-fc)*fr + dem[i+W+1]*fc*fr; }
function posAt(rt, dist){
  const d = rt.d; let lo = 0, hi = d.length-1;
  while(hi-lo > 1){ const m = (lo+hi)>>1; if(d[m] <= dist) lo = m; else hi = m; }
  const t = d[hi] > d[lo] ? (dist-d[lo])/(d[hi]-d[lo]) : 0, a = rt.pts[lo], b = rt.pts[hi];
  return {c:a[0]+(b[0]-a[0])*t, r:a[1]+(b[1]-a[1])*t, ang:Math.atan2(b[1]-a[1], b[0]-a[0]), i:lo};
}
function lonlat(c, r){ return [D.bbox[0] + c/W*(D.bbox[2]-D.bbox[0]), D.bbox[3] - r/H*(D.bbox[3]-D.bbox[1])]; }
function mobility(sl){ return sl<=5 ? ["Kolay (≤ 5°)","#7dff8a"] : sl<=15 ? ["Tekerlekli araca uygun (≤ 15°)","#d7f06a"] : sl<=25 ? ["Zor — paletli / yaya (≤ 25°)","#ffcf5a"] : ["Geçilemez (> 25°)","#ff6b6b"]; }

// ---- yan panel
function buildSide(){
  document.getElementById("bases").innerHTML = Object.keys(D.img).map(k=>`<button data-b="${k}" class="${k===base?'on':''}">${baseNames[k]}</button>`).join(" ");
  document.querySelectorAll("#bases button").forEach(b=> b.onclick = ()=>{ base = b.dataset.b; buildSide(); });
  if(stage === "select"){
    hint.innerHTML = "Sağdaki listeden ya da haritadan bir <b>rota</b> seçin, sonra <b>Sürüşü başlat</b>'a basın.";
    side.innerHTML = `<div class="card"><h3>İniş noktası</h3><div><span class="dot" style="background:#fff;border:2px solid #7dff8a"></span> ${D.start.name}</div>
      <div class="small" style="margin-top:4px">Rotalar, eğimin karesiyle artan bir maliyet yüzeyinde en düşük maliyetli yol olarak hesaplanmıştır.</div></div>
      <div class="card"><h3>Rotalar — birini seçin</h3>` + D.routes.map((r,i)=>`<div class="route ${sel===i?'sel':''}" data-i="${i}">
        <div class="nm"><span class="dot" style="background:${r.color}"></span>${i+1}. ${r.name}</div>
        <div class="st">${r.info}<br>Uzunluk <b>${fmt(r.len,1)} km</b> · en dik <b>${fmt(r.maxSlope,0)}°</b> · tırmanış <b>${fmt(r.climb)} m</b><br>15°'den dik kesim %${fmt(r.steep,1)} · tahmini <b>${fmt(r.len*1000/300)} sol</b></div></div>`).join("") +
      `<button id="go" class="big primary" ${sel===null?'disabled':''}>▶ Sürüşü başlat</button>
       <div class="small" style="margin-top:6px">Tahmini süre, günde 300 m yol alan bir gezici araç varsayımıyla verilmiştir.</div></div>`;
    side.querySelectorAll(".route").forEach(el=> el.onclick = ()=>{ sel = +el.dataset.i; buildSide(); });
    const go = document.getElementById("go"); if(go) go.onclick = ()=>{ stage = "drive"; s = 0; zoom = 4; buildSide(); cv.focus(); };
  } else {
    hint.innerHTML = "İlerlemek için <kbd>W</kbd>, geri gitmek için <kbd>S</kbd> tuşunu basılı tutun. Tuşlar çalışmazsa önce haritaya tıklayın.";
    const r = D.routes[sel];
    side.innerHTML = `<div class="card"><h3>Seçili rota</h3><div class="nm"><span class="dot" style="background:${r.color}"></span> <b>${r.name}</b></div>
        <canvas id="prof"></canvas><div class="small">Rota boyunca yükseklik; kırmızı kesimler 15°'den dik.</div></div>
      <div class="card"><h3>Aracın anlık konumu</h3><table id="live"></table></div>
      <div class="card"><div class="row" style="margin-bottom:6px"><button id="bw">▲ İleri (W)</button><button id="bs">▼ Geri (S)</button></div>
        <button id="an" class="big primary">🔬 Detaylı analiz</button>
        <div class="small" id="anh" style="margin:5px 0 7px 0"></div>
        <button id="back" style="width:100%">↩ Rota seçimine dön</button></div>`;
    const hold = (id, k)=>{ const b = document.getElementById(id); const dn = e=>{ e.preventDefault(); keys[k] = true; }, up = ()=>{ keys[k] = false; };
      b.addEventListener("pointerdown", dn); b.addEventListener("pointerup", up); b.addEventListener("pointerleave", up); b.addEventListener("pointercancel", up); };
    hold("bw","w"); hold("bs","s");
    document.getElementById("back").onclick = ()=>{ stage = "select"; zoom = 1; buildSide(); };
    document.getElementById("an").onclick = ()=>{ if(moving) return; const p = posAt(r, s), ll = lonlat(p.c, p.r);
      document.getElementById("busy").style.display = "flex";
      post("streamlit:setComponentValue", {dataType:"json", value:{evt:"analyze", nonce:Date.now(), route:sel, s:s, lon:ll[0], lat:ll[1],
        elev:elevAt(p.c,p.r), slope:at(slope,p.c,p.r)/4, tri:at(tri,p.c,p.r)*2, rough:at(rough,p.c,p.r)/255, lf:at(lf,p.c,p.r), name:r.name}}); };
    drawProfile();
  }
}
function live(){
  if(stage !== "drive") return; const r = D.routes[sel], p = posAt(r, s), ll = lonlat(p.c, p.r);
  const sl = at(slope,p.c,p.r)/4, mob = mobility(sl), L = at(lf,p.c,p.r), g = at(geo,p.c,p.r), gn = g>0 ? D.geoNames[g-1] : null;
  const rows = [["X (batı kenarından)", fmt(p.c*PXKM,1)+" km"], ["Y (güney kenarından)", fmt((H-p.r)*PXKM,1)+" km"],
    ["Enlem / boylam", fmt(ll[1],3)+"° K, "+fmt(ll[0],3)+"° D"], ["Yükselti (MOLA)", fmt(elevAt(p.c,p.r))+" m"], ["Eğim", fmt(sl,1)+"°"],
    ["Engebelilik (TRI)", fmt(at(tri,p.c,p.r)*2)+" m"], ["Yüzey pürüzlülüğü (MOLA)", fmt(at(rough,p.c,p.r)/255,2)+" (0–1)"],
    ["Topoğrafik yapı", `<span class="chip" style="background:${LFCOL[L]||'#ccc'}">${D.lfNames[L]||'—'}</span>`],
    ["Geçilebilirlik", `<span class="chip" style="background:${mob[1]}">${mob[0]}</span>`],
    ["Jeolojik birim", gn ? gn.unit+" — "+gn.desc : "—"],
    ["Kat edilen / kalan yol", fmt(s,1)+" / "+fmt(r.len-s,1)+" km"]];
  document.getElementById("live").innerHTML = rows.map(x=>`<tr><td class="k">${x[0]}</td><td class="v">${x[1]}</td></tr>`).join("");
  const an = document.getElementById("an"), anh = document.getElementById("anh");
  an.disabled = moving;
  anh.textContent = moving ? "Araç hareket ediyor; analiz için durun." : (s >= r.len-0.01 ? "Hedefe ulaştınız. Bulunduğunuz yerin analizini açabilirsiniz." : "Araç duruyor. Bulunduğu yerin 8 başlıklı analizini açabilirsiniz.");
  drawProfile();
}
function drawProfile(){
  const c = document.getElementById("prof"); if(!c) return; const r = D.routes[sel], dpr = window.devicePixelRatio||1;
  const w = c.clientWidth, h = c.clientHeight; c.width = w*dpr; c.height = h*dpr; const g = c.getContext("2d"); g.scale(dpr,dpr);
  const zmin = Math.min(...r.z), zmax = Math.max(...r.z), X = d=> 4 + d/r.len*(w-8), Y = z=> h-14 - (z-zmin)/Math.max(zmax-zmin,1)*(h-26);
  g.fillStyle = "#1b2130"; g.beginPath(); g.moveTo(X(0), h-12); r.d.forEach((d,i)=> g.lineTo(X(d), Y(r.z[i]))); g.lineTo(X(r.len), h-12); g.fill();
  for(let i=1;i<r.d.length;i++){ g.strokeStyle = r.sl[i] > 15 ? "#ff6b6b" : r.color; g.lineWidth = 1.6; g.beginPath(); g.moveTo(X(r.d[i-1]), Y(r.z[i-1])); g.lineTo(X(r.d[i]), Y(r.z[i])); g.stroke(); }
  const p = posAt(r, s), zz = elevAt(p.c, p.r); g.strokeStyle = "#fff"; g.lineWidth = 1; g.beginPath(); g.moveTo(X(s), 4); g.lineTo(X(s), h-12); g.stroke();
  g.fillStyle = "#fff"; g.beginPath(); g.arc(X(s), Y(zz), 3.5, 0, 7); g.fill();
  g.fillStyle = "#9aa6c4"; g.font = "10px sans-serif"; g.fillText(fmt(zmax)+" m", 4, 10); g.fillText(fmt(zmin)+" m", 4, h-15); g.textAlign = "right"; g.fillText(fmt(r.len,0)+" km", w-4, h-2); g.textAlign = "left"; g.fillText("0", 4, h-2);
}

// ---- çizim
function resize(){ const dpr = window.devicePixelRatio||1, b = cv.getBoundingClientRect(); cv.width = Math.round(b.width*dpr); cv.height = Math.round(b.height*dpr); }
function view(){
  const dpr = window.devicePixelRatio||1, cw = cv.width/dpr, ch = cv.height/dpr, fit = Math.min(cw/W, ch/H), sc = fit*zoom;
  let cx = W/2, cy = H/2;
  if(stage === "drive"){ const p = posAt(D.routes[sel], s); cx = p.c; cy = p.r; }
  const hw = cw/sc/2, hh = ch/sc/2;
  cx = hw >= W/2 ? W/2 : Math.min(W-hw, Math.max(hw, cx)); cy = hh >= H/2 ? H/2 : Math.min(H-hh, Math.max(hh, cy));
  return {dpr, cw, ch, sc, cx, cy};
}
function drawRoute(r, lw, alpha){ ctx.globalAlpha = alpha; ctx.strokeStyle = r.color; ctx.lineWidth = lw; ctx.lineJoin = "round"; ctx.lineCap = "round"; ctx.beginPath();
  r.pts.forEach((p,i)=> i ? ctx.lineTo(p[0],p[1]) : ctx.moveTo(p[0],p[1])); ctx.stroke(); ctx.globalAlpha = 1; }
function draw(){
  const v = view(); ctx.setTransform(v.dpr,0,0,v.dpr,0,0); ctx.fillStyle = "#000"; ctx.fillRect(0,0,v.cw,v.ch);
  ctx.setTransform(v.dpr*v.sc,0,0,v.dpr*v.sc, v.dpr*(v.cw/2 - v.cx*v.sc), v.dpr*(v.ch/2 - v.cy*v.sc));
  const im = imgs[base]; if(im && im.complete) ctx.drawImage(im, 0, 0, W, H);
  const u = 1/v.sc;  // 1 ekran pikseli
  if(stage === "select"){ D.routes.forEach((r,i)=>{ if(i!==sel){ drawRoute({pts:r.pts,color:"#000"}, 5*u, .45); drawRoute(r, 2.6*u, .9); } });
    if(sel!==null){ drawRoute({pts:D.routes[sel].pts,color:"#000"}, 8*u, .6); drawRoute(D.routes[sel], 4.6*u, 1); } }
  else { const r = D.routes[sel]; drawRoute({pts:r.pts,color:"#000"}, 6*u, .5); drawRoute(r, 3*u, .55);
    const p = posAt(r, s), done = r.pts.slice(0, p.i+1).concat([[p.c,p.r]]); drawRoute({pts:done,color:"#ffffff"}, 3.4*u, 1); }
  // işaretler
  const mark = (c, r, col, label, star)=>{ ctx.fillStyle = col; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5*u; ctx.beginPath();
    if(star){ for(let i=0;i<10;i++){ const a = -Math.PI/2 + i*Math.PI/5, rad = (i%2?4:9)*u; ctx.lineTo(c+Math.cos(a)*rad, r+Math.sin(a)*rad); } ctx.closePath(); }
    else ctx.arc(c, r, 6*u, 0, 7);
    ctx.fill(); ctx.stroke(); ctx.font = `600 ${12*u}px sans-serif`; ctx.lineWidth = 3*u; ctx.strokeText(label, c+10*u, r+4*u); ctx.fillStyle = "#fff"; ctx.fillText(label, c+10*u, r+4*u); };
  D.routes.forEach((r,i)=>{ if(stage==="select" || i===sel){ const e = r.pts[r.pts.length-1]; mark(e[0], e[1], r.color, (i+1)+". "+r.name, true); } });
  mark(D.start.c, D.start.r, "#7dff8a", "İniş noktası", false);
  if(stage === "drive"){ const p = posAt(D.routes[sel], s); if(moving) heading = p.ang + (keys.s && !keys.w ? Math.PI : 0); else if(!lastMove) heading = p.ang;
    ctx.save(); ctx.translate(p.c, p.r); ctx.rotate(heading); const k = u*1.25;
    ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.beginPath(); ctx.ellipse(1.5*k, 2*k, 13*k, 9*k, 0, 0, 7); ctx.fill();
    ctx.fillStyle = "#1a1a1a"; [[-8,-9],[0,-9.5],[8,-9],[-8,6],[0,6.5],[8,6]].forEach(q=> ctx.fillRect(q[0]*k-2.5*k, q[1]*k, 5*k, 3*k));
    ctx.fillStyle = "#f2f2f2"; ctx.strokeStyle = "#111"; ctx.lineWidth = 1*k; ctx.beginPath(); ctx.roundRect(-10*k, -6*k, 20*k, 12*k, 2.5*k); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#c1440e"; ctx.fillRect(-9*k, -5*k, 7*k, 10*k); ctx.fillStyle = "#ffd23f"; ctx.beginPath(); ctx.moveTo(13*k,0); ctx.lineTo(7*k,-4*k); ctx.lineTo(7*k,4*k); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore(); }
  // sabit katman: ölçek çubuğu, kuzey oku, küçük harita
  ctx.setTransform(v.dpr,0,0,v.dpr,0,0);
  const kmPerPx = PXKM/v.sc; let bar = [1,2,5,10,20,50,100,200].find(b=> b/kmPerPx >= 70) || 200, bw = bar/kmPerPx;
  ctx.fillStyle = "rgba(11,14,20,.75)"; ctx.fillRect(v.cw-bw-26, v.ch-34, bw+16, 24); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath();
  ctx.moveTo(v.cw-bw-18, v.ch-16); ctx.lineTo(v.cw-18, v.ch-16); ctx.moveTo(v.cw-bw-18, v.ch-21); ctx.lineTo(v.cw-bw-18, v.ch-11); ctx.moveTo(v.cw-18, v.ch-21); ctx.lineTo(v.cw-18, v.ch-11); ctx.stroke();
  ctx.fillStyle = "#fff"; ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.fillText(bar+" km", v.cw-bw/2-18, v.ch-21); ctx.textAlign = "left";
  ctx.fillStyle = "rgba(11,14,20,.75)"; ctx.beginPath(); ctx.arc(v.cw-28, 30, 18, 0, 7); ctx.fill(); ctx.fillStyle = "#ffd23f"; ctx.beginPath(); ctx.moveTo(v.cw-28, 16); ctx.lineTo(v.cw-34, 34); ctx.lineTo(v.cw-28, 30); ctx.lineTo(v.cw-22, 34); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.font = "700 10px sans-serif"; ctx.textAlign = "center"; ctx.fillText("K", v.cw-28, 45); ctx.textAlign = "left";
  if(stage === "drive" && zoom > 1.2){ const m = 150, mh = m*H/W, mx = 10, my = v.ch-mh-10; ctx.fillStyle = "#000"; ctx.fillRect(mx-2, my-2, m+4, mh+4);
    if(im && im.complete) ctx.drawImage(im, mx, my, m, mh); const r = D.routes[sel], f = m/W; ctx.strokeStyle = r.color; ctx.lineWidth = 1.5; ctx.beginPath();
    r.pts.forEach((p,i)=> i ? ctx.lineTo(mx+p[0]*f, my+p[1]*f) : ctx.moveTo(mx+p[0]*f, my+p[1]*f)); ctx.stroke();
    const hw = v.cw/v.sc/2, hh = v.ch/v.sc/2; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.strokeRect(mx+(v.cx-hw)*f, my+(v.cy-hh)*f, 2*hw*f, 2*hh*f);
    const p = posAt(r, s); ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(mx+p.c*f, my+p.r*f, 3, 0, 7); ctx.fill(); }
}

// ---- döngü ve girişler
let tPrev = performance.now(), tLive = 0;
function tick(t){
  const dt = Math.min(0.1, (t-tPrev)/1000); tPrev = t; moving = false;
  if(stage === "drive"){ const r = D.routes[sel], dir = (keys.w?1:0) - (keys.s?1:0);
    if(dir){ const p = posAt(r, s), sl = at(slope,p.c,p.r)/4, vkm = 4.0/(1 + Math.pow(sl/8, 2));  // dik yamaçta yavaşlar
      const ns = Math.min(r.len, Math.max(0, s + dir*vkm*dt)); moving = ns !== s; s = ns; if(moving) lastMove = t; }
    if(t - tLive > 90){ tLive = t; live(); } }
  draw(); requestAnimationFrame(tick);
}
function key(e, down){ const k = e.key.toLowerCase(); if(k === "w" || k === "arrowup"){ keys.w = down; e.preventDefault(); } if(k === "s" || k === "arrowdown"){ keys.s = down; e.preventDefault(); }
  if(down && (k === "+" || k === "=")) zoom = Math.min(12, zoom*1.3); if(down && k === "-") zoom = Math.max(1, zoom/1.3); }
window.addEventListener("keydown", e=> key(e, true)); window.addEventListener("keyup", e=> key(e, false));
window.addEventListener("blur", ()=>{ keys.w = keys.s = false; });
document.getElementById("zin").onclick = ()=>{ zoom = Math.min(12, zoom*1.4); cv.focus(); };
document.getElementById("zout").onclick = ()=>{ zoom = Math.max(1, zoom/1.4); cv.focus(); };
cv.addEventListener("wheel", e=>{ e.preventDefault(); zoom = Math.min(12, Math.max(1, zoom*(e.deltaY < 0 ? 1.15 : 1/1.15))); }, {passive:false});
cv.addEventListener("click", e=>{ cv.focus(); if(stage !== "select") return; const v = view(), b = cv.getBoundingClientRect();
  const c = (e.clientX-b.left - v.cw/2)/v.sc + v.cx, r = (e.clientY-b.top - v.ch/2)/v.sc + v.cy; let best = null, bd = 12/v.sc;
  D.routes.forEach((rt,i)=> rt.pts.forEach(p=>{ const d = Math.hypot(p[0]-c, p[1]-r); if(d < bd){ bd = d; best = i; } }));
  if(best !== null){ sel = best; buildSide(); } });
window.addEventListener("resize", resize);
resize(); buildSide(); requestAnimationFrame(tick);
setTimeout(()=>{ if(!restored){ restored = true; buildSide(); } }, 1500);
</script></body></html>
"""


@st.cache_resource(show_spinner=False)
def sim_component(n: int = 450):
    """Simülasyon sayfasını geçici bir klasöre yazar ve çift yönlü Streamlit bileşeni olarak tanımlar."""
    import hashlib, json, os, tempfile
    import streamlit.components.v1 as components
    data = sim_prepare(n)
    html = SIM_HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    d = os.path.join(tempfile.gettempdir(), "volkanik_sim_" + hashlib.md5(html.encode("utf-8")).hexdigest()[:12])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return components.declare_component("mars_sim_2d", path=d)


def sim_analysis():
    """Aracın durduğu noktanın çevresi için sekiz başlıklı ayrıntılı analiz ekranı."""
    ss = st.session_state
    p = ss["sim_point"]
    set_body("mars")
    half = 0.35  # derece ≈ 21 km; analiz kutusu yaklaşık 41 × 41 km
    bbox = (round(p["lon"] - half, 4), round(p["lat"] - half, 4), round(p["lon"] + half, 4), round(p["lat"] + half, 4))
    c1, c2 = st.columns([3, 1])
    with c1:
        st.title("Detaylı analiz — aracın bulunduğu yer")
        st.markdown(f"**Rota:** {p['name']} · kat edilen yol **{p['s']:.1f} km** · konum **{p['lat']:.3f}° K, {p['lon']:.3f}° D**")
    with c2:
        st.write("")
        if st.button("◀ Simülasyona dön", type="primary"):
            ss["sim_view"] = "sim"
            st.rerun()
    m = st.columns(5)
    m[0].metric("Yükselti", f"{p['elev']:.0f} m")
    m[1].metric("Eğim", f"{p['slope']:.1f}°")
    m[2].metric("Engebelilik (TRI)", f"{p['tri']:.0f} m")
    m[3].metric("Yüzey pürüzlülüğü", f"{p['rough']:.2f}")
    m[4].metric("Topoğrafik yapı", SIM_LF.get(int(p["lf"]), "—"))
    st.caption(f"Analiz, aracın çevresindeki yaklaşık {2 * half * BODY['m_lat'] / 1000:.0f} × {2 * half * BODY['m_lat'] / 1000:.0f} km'lik "
               "kutu için yapılır; araç kutunun tam ortasındadır. TES haritalarının pikseli yaklaşık 20 km olduğundan bu ölçekte "
               "neredeyse tek renk görünür.")
    try:
        with st.spinner("Konumun yükseklik modeli indiriliyor…"):
            dem, px = get_mars_dem(bbox, 300)
    except Exception as e:
        st.error(f"Mars yükseklik verisi alınamadı.\n\n{e}")
        return
    funcs = [mars_geology, mars_topo, mars_thermal, mars_water, mars_processes, mars_biology, mars_subsurface, mars_operations]
    themes = dict(zip(THEME_TITLES, funcs))
    theme = st.radio("İnceleme başlığı", list(themes), horizontal=True, label_visibility="collapsed", key="sim_theme")
    st.header(THEME_TITLES[theme])
    themes[theme](bbox, 300, dem, px)
    st.divider()
    if st.button("◀ Simülasyona dön", type="primary", key="sim_back_bottom"):
        ss["sim_view"] = "sim"
        st.rerun()


def sim_page():
    ss = st.session_state
    if ss.get("sim_view") == "analysis" and ss.get("sim_point"):
        sim_analysis()
        return
    set_body("mars")
    st.title("🎮 2D Simulation — Mars gezici aracı")
    st.caption("Altlık görüntüler NASA Mars Trek'ten canlı alınır (THEMIS gündüz kızılötesi 100 m, Viking renk 232 m); yükseklik, "
               "eğim ve engebelilik MOLA + HRSC yükseklik modelinden, pürüzlülük MOLA pürüzlülük haritasından gelir.")
    try:
        with st.spinner("Simülasyon verisi hazırlanıyor (ilk açılışta 10–20 sn)…"):
            comp = sim_component()
    except Exception as e:
        st.error(f"Simülasyon verisi hazırlanamadı (trek.nasa.gov). İnternet bağlantınızı kontrol edin.\n\n{e}")
        return
    val = comp(route=ss.get("sim_route"), s=float(ss.get("sim_s", 0.0)), key="mars_sim_2d", default=None)
    if isinstance(val, dict) and val.get("evt") == "analyze" and val.get("nonce") != ss.get("sim_nonce"):
        ss["sim_nonce"] = val["nonce"]
        ss["sim_route"], ss["sim_s"] = int(val["route"]), float(val["s"])
        ss["sim_point"] = val
        ss["sim_view"] = "analysis"
        st.rerun()


# --------------------------------------------------------------------------------------
# 3D SİMÜLASYON — A.R.E.S. çift-analog gezici rota simülasyonu (Mars, Elysium – Cerberus Fossae)
# --------------------------------------------------------------------------------------
# Yer gerçeği analogları (puanlar ve saha tanımları kullanıcının verdiği değerlerdir):
SIM3_ANALOG = {
    "karapinar": {"ad": "Karapınar Volkanik Alanı (Konya — Meke, Acıgöl, Yılan maarları)", "skor": "64,5 / 80",
                  "surec": "Islak / hidrovolkanik süreçler: freatomagmatik base surge katmanları, krater içi cüruf konisi, "
                           "düşük sıcaklıklı nötr-alkali sulu alterasyon (zeolit, kalsit, sülfat, ikincil silika), evaporitik göl çökelleri",
                  "sensor": "Raman + LIBS", "ek": "Ryan vd. (2024): 11 / 14"},
    "divlit": {"ad": "Divlit Volkanik Parkı (Kula, Manisa)", "skor": "61,0 / 80",
               "surec": "Kuru efüzif–stromboliyen bazaltik süreçler: Holosen (< 11–4,7 ka) skorya konisi, 1 m'ye varan keskin bloklu "
                        "'a'a lav akıntıları, alkali bazanit–fonotefrit, 3D haritalanmış lav tünelleri",
               "sensor": "LiDAR + GPR, tekerlek koruma", "ek": ""},
}
SIM3_L0 = (161.98, 9.64)   # iniş noktası (boylam, enlem)
SIM3_B1 = (161.896, 8.344)  # Cerberus Fossae güney kolu (engebeli tepelik alanı kesen çatlak): en yakın çatlak tabanına oturtulur


def _sim3d_cache_file():
    import os, tempfile
    return os.path.join(tempfile.gettempdir(), "volkanik_sim3d_cache.pkl")


def _synthetic_layers(n):
    """NASA servisine ulaşılamazsa ve önbellek de yoksa kullanılan sentetik arazi (gerçek veri DEĞİLDİR)."""
    bbox = tuple(MARS_REGIONS[SIM_REGION]["bbox"])
    W, H, px = grid_shape(bbox, n)
    rng = np.random.default_rng(42)
    z = sum(ndi.gaussian_filter(rng.standard_normal((H, W)), s) * a for s, a in ((40, 9000), (12, 1500), (4, 250), (1.2, 40)))
    yy, xx = np.mgrid[0:H, 0:W]
    z = z - 2800 + (H - yy) * 1.2
    d_f = np.abs((yy - 0.62 * H) - 0.28 * (xx - 0.3 * W)) / math.hypot(1, 0.28)   # çatlak
    z -= 320 * np.exp(-(d_f / 1.6) ** 2) * (xx > 0.25 * W)
    for cy, cx, rad, dep in ((0.35, 0.62, 9, 600), (0.5, 0.2, 14, 500), (0.75, 0.45, 6, 300), (0.2, 0.3, 5, 250)):
        rr = np.hypot(yy - cy * H, xx - cx * W) / rad
        z += -dep * np.exp(-rr ** 2 * 2.2) + 0.25 * dep * np.exp(-((rr - 1.05) / 0.22) ** 2)
    dem = z.astype("float32")
    shade = LightSource(315, 45).hillshade(dem, vert_exag=3, dx=px, dy=px)
    g = (np.clip(shade, 0, 1) * 255).astype("uint8")
    rough = np.clip(ndi.gaussian_filter(rng.random((H, W)), 6) * 3 - 1, 0, 1)
    tes = {k: np.clip(ndi.gaussian_filter(rng.random((H, W)), 30) * 2.2 - 0.6, 0.02, 1).astype("float32")
           for k in ("glass", "pyrox", "plag", "ti", "dust", "albedo")}
    return {"dem": dem, "px": px, "bbox": bbox, "rough06": rough.astype("float32"), "geo": np.ones((H, W), "int16"),
            "geoNames": ["lAv"], "geoDesc": {"lAv": "Late Amazonian volcanic unit"}, "tes": tes,
            "themis": np.dstack([g, g, g]), "viking": np.dstack([np.clip(g * 0.95 + 30, 0, 255), g * 0.62 + 20, g * 0.48 + 15]).astype("uint8"),
            "source": "synthetic"}


def _fetch_layers(n):
    """NASA Mars Trek'ten gerçek veriler."""
    import os
    if os.environ.get("VOLKANIK_SIM3D_OFFLINE"):  # bağlantı kesintisini denemek için
        raise RuntimeError("çevrimdışı deneme")
    bbox = tuple(MARS_REGIONS[SIM_REGION]["bbox"])
    dem, px = get_mars_dem(bbox, n)
    S = 1800  # ≈ 165 m / piksel: THEMIS'in (100 m) ayrıntısına yakın, Viking'in (232 m) özgün çözünürlüğünün üstünde
    th = trek_image("THEMIS_DayIR_ControlledMosaics_100m_v2_oct2018", bbox, S, S)
    vk = trek_image("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, S, S)
    r = mlayer("mola_roughness", bbox, dem.shape)
    rough = r[..., 2].astype("float32") / 255 if r is not None else np.zeros(dem.shape, "float32")  # mavi kanal = 0,6 km taban
    geo = mfeat("jeoloji", bbox)
    grid, names = rasterize_polygons(geo, bbox, dem.shape, "Unit") if geo else (np.ones(dem.shape, "int16"), ["?"])
    tes = {}
    for key, layer, kind in (("glass", "TES_Glass_Clay", "rainbow"), ("pyrox", "TES_Clinopyroxene", "rainbow"),
                             ("plag", "TES_Plagioclase", "rainbow"), ("ti", "TES_Thermal_Inertia", "rainbow"),
                             ("dust", "TES_Dust", "rainbow"), ("albedo", "Mars_MGS_TES_Albedo_mosaic_global_7410m", "gray")):
        v = mvalue(layer, bbox, dem.shape, kind)
        v = np.full(dem.shape, 0.3, "float32") if v is None or not np.isfinite(v).any() else v
        tes[key] = np.where(np.isfinite(v), v, np.nanmedian(v)).astype("float32")
    return {"dem": dem, "px": px, "bbox": bbox, "rough06": rough, "geo": grid, "geoNames": names,
            "geoDesc": {str(f["a"].get("Unit")): str(f["a"].get("UnitDesc")) for f in geo}, "tes": tes,
            "themis": th[..., :3], "viking": vk[..., :3], "source": "trek"}


@st.cache_resource(show_spinner=False)
def sim3d_layers(n: int = 450):
    """Bölge katmanları: gerçek veri -> (başarısızsa) yerel önbellek -> (o da yoksa) sentetik arazi."""
    import pickle
    set_body("mars")
    try:
        L = _fetch_layers(n)
        try:
            with open(_sim3d_cache_file(), "wb") as f:
                pickle.dump(L, f)
        except Exception:
            pass
    except Exception:
        try:
            with open(_sim3d_cache_file(), "rb") as f:
                L = pickle.load(f)
            L["source"] = "cache"
        except Exception:
            L = _synthetic_layers(n)
    dem, px, bbox = L["dem"], L["px"], L["bbox"]
    H, W = dem.shape
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    lf = landforms(dem, px, slope, radius_m=max(300, 3 * px))
    depth, lab, dep = find_depressions(dem, px, bbox, min_depth=30, min_area_km2=0.5)
    sinks = tuple(int(r["_r"]) * W + int(r["_c"]) for _, r in dep.iterrows())
    fr = flow_routing(dem, px, max(0.3, round(150 * px * px / 1e6, 2)), sinks)
    acc = fr["acc_km2"]

    def to_rc(lon, lat):
        return (int(np.clip((bbox[3] - lat) / (bbox[3] - bbox[1]) * H, 0, H - 1)),
                int(np.clip((lon - bbox[0]) / (bbox[2] - bbox[0]) * W, 0, W - 1)))

    yy, xx = np.mgrid[0:H, 0:W]
    km = lambda rc: np.hypot(yy - rc[0], xx - rc[1]) * px / 1000
    L0 = to_rc(*SIM3_L0)
    # B1: Cerberus Fossae çatlağı — verilen noktanın 5 km çevresindeki en çukur (dar pencereye göre) hücre
    r0, c0 = to_rc(*SIM3_B1)
    k = max(3, int(round(5000 / px)))
    tpi = dem - ndi.uniform_filter(dem, size=max(5, int(round(4000 / px)) | 1), mode="nearest")
    win = tpi[max(0, r0 - k):r0 + k + 1, max(0, c0 - k):c0 + k + 1]
    rr, cc = np.unravel_index(int(np.argmin(win)), win.shape)
    B1 = (max(0, r0 - k) + int(rr), max(0, c0 - k) + int(cc))
    d0 = km(L0)
    # Duraklar L0 → B1 hattı boyunca sıralı aranır: t = hat üzerindeki konum (0–1), xt = hatta dik uzaklık (km)
    vr, vc = B1[0] - L0[0], B1[1] - L0[1]
    vl = max(math.hypot(vr, vc), 1e-6)
    t_al = ((yy - L0[0]) * vr + (xx - L0[1]) * vc) / vl ** 2
    xt = np.abs((yy - L0[0]) * vc - (xx - L0[1]) * vr) / vl * px / 1000
    band = lambda t0, t1, w: (t_al >= t0) & (t_al <= t1) & (xt <= w)
    # S1: maar / base surge analoğu — hattın ilk yarısına en yakın, 10 km'den küçük belirgin krater; kenarına (L0 tarafı) konur
    S1 = None
    cand = dep[(dep["Eşdeğer çap (m)"] <= 10_000)] if len(dep) else dep
    if len(cand):
        rc_ = [(int(r["_r"]), int(r["_c"])) for _, r in cand.iterrows()]
        ok = np.array([band(0.10, 0.50, 45)[q] and d0[q] >= 15 and km(B1)[q] >= 15 for q in rc_])
        if ok.any():
            row = cand.iloc[int(np.argmin(np.where(ok, [xt[q] for q in rc_], 1e9)))]
            cr, cc_ = int(row["_r"]), int(row["_c"])
            rad = row["Eşdeğer çap (m)"] / 2 / px * 1.35
            v = np.array([L0[0] - cr, L0[1] - cc_], dtype=float)
            v /= max(np.hypot(*v), 1e-6)
            S1 = (int(np.clip(cr + v[0] * rad, 0, H - 1)), int(np.clip(cc_ + v[1] * rad, 0, W - 1)))
    if S1 is None:
        S1 = (int(L0[0] + 0.3 * vr - 0.08 * vc), int(L0[1] + 0.3 * vc + 0.08 * vr))
    # S2: zeolit–sülfat alterasyon analoğu — hattın orta kesiminde TES tabaka silikat / yüksek-Si cam haritasının en yüksek olduğu yer
    g = ndi.gaussian_filter(L["tes"]["glass"], 3)
    m2 = band(0.40, 0.72, 40) & (slope < 6) & (km(S1) >= 18)
    S2 = np.unravel_index(int(np.argmax(np.where(m2, g, -1))), dem.shape) if m2.any() else (int(L0[0] + 0.55 * vr), int(L0[1] + 0.55 * vc))
    S2 = (int(S2[0]), int(S2[1]))
    # S3: evaporit kanalı analoğu — hattın son kesiminde, düz zeminde en büyük akış birikimi (kanal / havza tabanı)
    m3 = band(0.68, 0.95, 40) & (slope < 3) & (km(S1) >= 18) & (km(S2) >= 18) & (km(S2) <= 55) & (km(B1) >= 12)
    S3 = np.unravel_index(int(np.argmax(np.where(m3, acc, -1))), dem.shape) if m3.any() else (int(L0[0] + 0.8 * vr), int(L0[1] + 0.8 * vc))
    S3 = (int(S3[0]), int(S3[1]))

    # Katman 1 — Karapınar bilim puanı (0–100)
    norm = lambda a: np.clip((a - np.percentile(a, 5)) / max(np.percentile(a, 99) - np.percentile(a, 5), 1e-6), 0, 1)
    twi = np.log(np.maximum(acc * 1e6 / px, px) / np.maximum(np.tan(np.radians(slope)), 1e-3))
    sci = 22 * norm(g) + 14 * norm(twi)
    for p in (S1, S2, S3):
        sci = np.maximum(sci, 100 * np.exp(-(km(p) / 5.0) ** 2))
    # Katman 2 — Divlit tekerlek aşınması (0–100): düz (< 6°) ama 0,6 km tabanında pürüzlü volkanik zemin
    rs = ndi.gaussian_filter(L["rough06"], 1.2)
    flat = slope < 6
    p50, p95 = (np.percentile(rs[flat], [40, 88]) if flat.any() else (0.4, 0.8))
    wear = 100 * np.clip((rs - p50) / max(p95 - p50, 1e-6), 0, 1) * np.clip((9 - slope) / 3, 0, 1)
    volc = np.isin(L["geo"], [i + 1 for i, nm in enumerate(L["geoNames"]) if "v" in nm.lower()]) | (L["source"] == "synthetic")
    wear = np.where(volc, wear, 0.5 * wear)
    L.update({"slope": slope, "tri": tri, "lf": lf, "acc": acc, "twi": twi, "sci": sci.astype("float32"),
              "wear": wear.astype("float32"), "pois": {"L0": L0, "S1": S1, "S2": S2, "S3": S3, "B1": B1}, "dep": dep})
    return L


@st.cache_resource(show_spinner=False)
def three_js_source():
    """three.js (r128) kaynağını indirir ve diske önbelleğe alır; sayfaya gömülür, tarayıcıdan ayrıca yüklenmez."""
    import os, tarfile, tempfile
    path = os.path.join(tempfile.gettempdir(), "volkanik_three_r128.min.js")
    if os.path.exists(path) and os.path.getsize(path) > 300_000:
        return open(path, encoding="utf-8").read()
    src = None
    for url in ("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
                "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js",
                "https://unpkg.com/three@0.128.0/build/three.min.js"):
        try:
            r = _session.get(url, timeout=25)
            if r.ok and len(r.text) > 300_000:
                src = r.text
                break
        except Exception:
            continue
    if src is None:
        r = _session.get("https://registry.npmjs.org/three/-/three-0.128.0.tgz", timeout=60)
        r.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as t:
            src = t.extractfile("package/build/three.min.js").read().decode("utf-8")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    return src


def sim3d_payload(n: int = 450):
    """Tarayıcıya gönderilen veri paketi: dokular, yükseklik türevleri, analog katmanları, duraklar."""
    from PIL import Image
    L = sim3d_layers(n)
    dem, px, bbox = L["dem"], L["px"], L["bbox"]
    H, W = dem.shape
    S = 1024
    up = lambda rgb: np.asarray(Image.fromarray(rgb.astype("uint8")).resize((S, S), Image.BILINEAR))
    shade = np.clip(LightSource(315, 45).hillshade(dem, vert_exag=3, dx=px, dy=px), 0, 1)
    elev = LightSource(315, 45).shade(dem, cmap=ELEV_CMAP, vert_exag=3, dx=px, dy=px, blend_mode="soft")[..., :3] * 255
    elev = np.where((L["slope"] > 15)[..., None], 0.45 * elev + 0.55 * np.array([214, 48, 49]), elev)
    pal = (plt.cm.Set2(np.arange(8))[:, :3] * 255)
    geo_rgb = pal[(L["geo"] - 1) % 8] * (0.45 + 0.55 * shade[..., None])
    base = np.dstack([shade * 150 + 40] * 3)
    w, s_ = (L["wear"] / 100)[..., None], (L["sci"] / 100)[..., None]
    cost = base * (1 - 0.75 * np.maximum(w, s_)) + 0.75 * (w * np.array([235, 64, 52]) + s_ * np.array([40, 220, 170]))
    cost = np.where((L["slope"] > 24)[..., None], np.array([150, 60, 200]), np.clip(cost, 0, 255))
    # Viking renk mozaiği 232 m'dir; aynı alanın 100 m'lik THEMIS görüntüsündeki ince ayrıntı renge bindirilerek keskinleştirilir
    vk = L["viking"].astype("float32")
    tg = L["themis"].astype("float32").mean(axis=-1)
    if tg.shape == vk.shape[:2] and L["source"] != "synthetic":
        vk = np.clip(vk * np.clip(tg / (ndi.gaussian_filter(tg, 7) + 1.0), 0.6, 1.55)[..., None], 0, 255)
    imgs = {"themis": _jpeg_b64(L["themis"], 88), "viking": _jpeg_b64(vk, 88), "elev": _jpeg_b64(up(elev)),
            "geo": _jpeg_b64(up(geo_rgb)), "cost": _jpeg_b64(up(cost))}
    names = L["geoNames"]
    return {
        "W": W, "H": H, "px": float(px), "bbox": list(bbox), "img": imgs, "source": L["source"],
        "dem": _b64(dem, "<i2"), "slope": _b64(np.clip(L["slope"] * 4, 0, 255), "u1"), "tri": _b64(np.clip(L["tri"] / 2, 0, 255), "u1"),
        "rough": _b64(np.clip(L["rough06"] * 255, 0, 255), "u1"), "lf": _b64(L["lf"], "u1"), "geo": _b64(np.clip(L["geo"], 0, 255), "u1"),
        "sci": _b64(np.clip(L["sci"] * 2.55, 0, 255), "u1"), "wear": _b64(np.clip(L["wear"] * 2.55, 0, 255), "u1"),
        "geoNames": [{"unit": nm, "desc": SIM_GEO_TR.get(nm, L["geoDesc"].get(nm, "")),
                      "color": matplotlib.colors.to_hex(pal[i % 8] / 255)} for i, nm in enumerate(names)],
        "lfNames": SIM_LF, "pois": {k: [int(v[1]), int(v[0])] for k, v in L["pois"].items()},
        "analog": SIM3_ANALOG,
    }


SIM3D_HTML = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<style>
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#0b0e14;color:#e8e6e3;font-family:"Source Sans Pro","Segoe UI",system-ui,sans-serif;font-size:13.5px}
#wrap{display:flex;gap:8px;height:940px;padding:8px}
#left{width:252px;flex:none;display:flex;flex-direction:column;gap:7px;overflow-y:auto}
#mid{flex:1;min-width:0;display:flex;flex-direction:column;gap:7px}
#right{width:306px;flex:none;display:flex;flex-direction:column;gap:7px;overflow-y:auto}
.card{background:#141926;border:1px solid #2a3145;border-radius:8px;padding:8px 9px}
.card h3{margin:0 0 6px 0;font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:#9aa6c4}
button{font:inherit;color:#e8e6e3;background:#1b2130;border:1px solid #39435c;border-radius:6px;padding:5px 8px;cursor:pointer}
button:hover{background:#263049}
button.on{background:#c1440e;border-color:#ff8a57;color:#fff}
button.big{font-size:15px;font-weight:700;padding:9px 10px;width:100%;background:#c1440e;border-color:#ff8a57;color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
.route{border:1px solid #2a3145;border-radius:7px;padding:6px 7px;margin-bottom:5px;cursor:pointer;background:#10141f}
.route.sel{border-color:#ffd23f;background:#221f12}
.route .nm{font-weight:700;display:flex;gap:6px;align-items:center}
.route .st{color:#b9c2d8;font-size:12px;margin-top:3px;line-height:1.35}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block;flex:none}
.lay{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.lay button{font-size:12px;padding:4px 3px}
.small{font-size:11.5px;color:#9aa6c4;line-height:1.35}
input[type=range]{width:100%;accent-color:#ff8a57}
#view{position:relative;flex:1;min-height:0;border:1px solid #2a3145;border-radius:8px;overflow:hidden;background:#05070b}
#gl{width:100%;height:100%;display:block;outline:none}
#hint{position:absolute;left:8px;top:8px;background:rgba(11,14,20,.8);border:1px solid #39435c;border-radius:6px;padding:5px 8px;font-size:12.5px;max-width:74%}
#src{position:absolute;right:8px;top:8px;border-radius:6px;padding:4px 8px;font-size:12px;font-weight:700}
#busy{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:rgba(11,14,20,.86);font-size:18px;font-weight:700;color:#ffd23f}
#charts{display:flex;gap:7px;height:196px;flex:none}
#charts .card{flex:1;min-width:0;padding:6px 8px;display:flex;flex-direction:column}
#charts canvas{width:100%;flex:1;min-height:0;display:block}
table{width:100%;border-collapse:collapse}
td{padding:2.5px 0;border-bottom:1px solid #20263a;vertical-align:top}
td.k{color:#9aa6c4;width:46%}
td.v{font-weight:600;text-align:right;font-variant-numeric:tabular-nums}
.chip{display:block;padding:5px 8px;border-radius:7px;font-weight:700;color:#0b0e14;line-height:1.3}
#analog{border-radius:7px;padding:7px 8px;border:1px solid #39435c;background:#10141f;line-height:1.38}
#analog b.t{display:block;font-size:13.5px;margin-bottom:3px}
kbd{background:#263049;border:1px solid #4a5677;border-bottom-width:2px;border-radius:4px;padding:0 5px;font-weight:700;color:#ffd23f}
.row{display:flex;gap:5px}.row>*{flex:1}
.lg{display:flex;align-items:center;gap:5px;font-size:11.5px;color:#b9c2d8;margin-top:2px}
.sw{width:12px;height:12px;border-radius:2px;flex:none;border:1px solid #0006}
</style></head><body>
<div id="wrap">
  <div id="left">
    <div class="card"><h3>Rota seçimi</h3><div id="routes"></div></div>
    <div class="card"><h3>Harita katmanı</h3><div class="lay" id="layers"></div><div id="legend" style="margin-top:6px"></div></div>
    <div class="card"><h3>Divlit 'a'a tekerlek aşınma cezası</h3>
      <input type="range" id="pen" min="0" max="10" step="0.5" value="5">
      <div class="small"><b id="penv" style="color:#ffd23f">5</b> / 10 — Rota B bu değere göre anında yeniden hesaplanır.</div></div>
    <div class="card"><h3>Hareket</h3><div class="row"><button id="bw">▲ İleri [W]</button><button id="bs">▼ Geri [S]</button></div>
      <div class="small" style="margin-top:5px"><kbd>W</kbd> ileri, <kbd>S</kbd> geri. Tuşlar çalışmazsa önce 3D görünüme tıklayın.</div></div>
    <div class="card"><h3>Kamera ve arazi</h3><div class="row"><button id="camF" class="on">Takip</button><button id="camO">Genel bakış</button></div>
      <div class="small" style="margin-top:6px">Arazi gerçek ölçektedir (düşey abartı yok). Sürükleyerek döndürün, tekerlekle yakınlaşın.</div></div>
  </div>
  <div id="mid">
    <div id="view"><canvas id="gl" tabindex="0"></canvas><div id="hint"></div><div id="src"></div><div id="busy">Detaylı analiz hazırlanıyor…</div></div>
    <div id="charts">
      <div class="card"><h3>Yükselti ve eğim profili (seçili rota)</h3><canvas id="c1"></canvas></div>
      <div class="card"><h3>Kümülatif bilim getirisi ve tekerlek aşınması (kesikli: Rota A · düz: Rota B)</h3><canvas id="c2"></canvas></div>
    </div>
  </div>
  <div id="right">
    <div class="card"><h3>Anlık telemetri</h3><table id="tele"></table></div>
    <div class="card"><h3>Geçilebilirlik durumu</h3><div id="pass" class="chip"></div></div>
    <div class="card"><h3>Aktif Türkiye analoğu eşleşmesi</h3><div id="analog"></div></div>
    <div class="card"><button id="an" class="big">🔬 Detaylı Analiz</button><div class="small" id="anh" style="margin-top:5px"></div></div>
  </div>
</div>
<script>__THREE__</script>
<script>
const D = __DATA__;
const W = D.W, H = D.H, PXKM = D.px/1000;
function dec(b64, T){ const s = atob(b64), u = new Uint8Array(s.length); for(let i=0;i<s.length;i++) u[i]=s.charCodeAt(i); return T===Int16Array ? new Int16Array(u.buffer) : u; }
const dem = dec(D.dem, Int16Array), slopeA = dec(D.slope), triA = dec(D.tri), roughA = dec(D.rough), lfA = dec(D.lf), geoA = dec(D.geo), sciA = dec(D.sci), wearA = dec(D.wear);
let zmin = 1e9, zmax = -1e9; for(let i=0;i<dem.length;i++){ if(dem[i]<zmin) zmin = dem[i]; if(dem[i]>zmax) zmax = dem[i]; }
const fmt = (v,d=0)=> Number(v).toLocaleString("tr-TR",{minimumFractionDigits:d,maximumFractionDigits:d,useGrouping:false});
const at = (arr,c,r)=> arr[Math.min(H-1,Math.max(0,Math.round(r)))*W + Math.min(W-1,Math.max(0,Math.round(c)))];
function elevAt(c,r){ const c0 = Math.min(W-2,Math.max(0,Math.floor(c))), r0 = Math.min(H-2,Math.max(0,Math.floor(r))), fc = Math.min(1,Math.max(0,c-c0)), fr = Math.min(1,Math.max(0,r-r0)), i = r0*W+c0;
  return dem[i]*(1-fc)*(1-fr) + dem[i+1]*fc*(1-fr) + dem[i+W]*(1-fc)*fr + dem[i+W+1]*fc*fr; }
const lonlat = (c,r)=> [D.bbox[0] + c/W*(D.bbox[2]-D.bbox[0]), D.bbox[3] - r/H*(D.bbox[3]-D.bbox[1])];

// ---------------- Streamlit bileşen protokolü
function post(type, data){ window.parent.postMessage(Object.assign({isStreamlitMessage:true, type:type}, data||{}), "*"); }
let restored = false;
window.addEventListener("message", e=>{ if(!e.data || e.data.type !== "streamlit:render" || restored) return; restored = true; const a = e.data.args || {};
  if(a.pen !== null && a.pen !== undefined){ pen = +a.pen; document.getElementById("pen").value = pen; }
  if(a.layer && D.img[a.layer]) layer = a.layer;
  if(a.route === "A" || a.route === "B") sel = a.route;
  planAll(); if(a.s) s = Math.min(Math.max(+a.s,0), R[sel].len); refreshAll(); });
post("streamlit:componentReady", {apiVersion:1}); post("streamlit:setFrameHeight", {height:956});

// ---------------- A* rota motoru (yarı çözünürlüklü ızgara)
const ST = 2, GW = Math.floor(W/ST), GH = Math.floor(H/ST), STEP = PXKM*ST;
const gs = new Float32Array(GW*GH), gw = new Float32Array(GW*GH);
for(let r=0;r<GH;r++) for(let c=0;c<GW;c++){ let ms = 0, mw = 0; for(let a=0;a<ST;a++) for(let b=0;b<ST;b++){ const i = (r*ST+a)*W + c*ST+b; ms = Math.max(ms, slopeA[i]/4); mw = Math.max(mw, wearA[i]/2.55); } gs[r*GW+c] = ms; gw[r*GW+c] = mw; }
function cellCost(i, p){ const sl = gs[i]; let k = 1 + (sl/8)*(sl/8) + p*gw[i]/100*1.6; if(sl > 24) k += 400; return k; }   // > 24° fiilen geçilemez
function astar(a, b, p){   // a, b: [c, r] tam ızgara koordinatı
  const sc = Math.min(GW-1,Math.round(a[0]/ST)), sr = Math.min(GH-1,Math.round(a[1]/ST)), tc = Math.min(GW-1,Math.round(b[0]/ST)), tr = Math.min(GH-1,Math.round(b[1]/ST));
  const N = GW*GH, g = new Float32Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1), closed = new Uint8Array(N), start = sr*GW+sc, goal = tr*GW+tc;
  const hp = [], hf = []; const push = (i,f)=>{ let k = hp.length; hp.push(i); hf.push(f); while(k>0){ const q = (k-1)>>1; if(hf[q] <= hf[k]) break; [hp[q],hp[k]] = [hp[k],hp[q]]; [hf[q],hf[k]] = [hf[k],hf[q]]; k = q; } };
  const pop = ()=>{ const top = hp[0], li = hp.pop(), lf_ = hf.pop(); if(hp.length){ hp[0] = li; hf[0] = lf_; let k = 0; for(;;){ let l = 2*k+1, r2 = l+1, m = k; if(l<hp.length && hf[l]<hf[m]) m = l; if(r2<hp.length && hf[r2]<hf[m]) m = r2; if(m===k) break; [hp[m],hp[k]] = [hp[k],hp[m]]; [hf[m],hf[k]] = [hf[k],hf[m]]; k = m; } } return top; };
  g[start] = 0; push(start, 0);
  while(hp.length){ const i = pop(); if(i === goal) break; if(closed[i]) continue; closed[i] = 1; const r = (i/GW)|0, c = i - r*GW, ci = cellCost(i, p);
    for(let dr=-1;dr<=1;dr++) for(let dc=-1;dc<=1;dc++){ if(!dr && !dc) continue; const rr = r+dr, cc = c+dc; if(rr<0||cc<0||rr>=GH||cc>=GW) continue; const j = rr*GW+cc; if(closed[j]) continue;
      const ng = g[i] + (ci + cellCost(j, p))/2 * (dr&&dc ? 1.41421 : 1) * STEP; if(ng < g[j]){ g[j] = ng; prev[j] = i; push(j, ng + Math.hypot(rr-tr, cc-tc)*STEP); } } }
  const out = []; let k = goal; while(k !== -1){ const r = (k/GW)|0; out.push([(k - r*GW)*ST + ST/2 - .5, r*ST + ST/2 - .5]); k = prev[k]; } out.reverse();
  out[0] = [a[0], a[1]]; out[out.length-1] = [b[0], b[1]]; return out;
}
function smooth(p){ if(p.length < 6) return p; const o = p.map(q=> q.slice()); for(let i=2;i<p.length-2;i++){ o[i][0] = (p[i-2][0]+p[i-1][0]+p[i][0]+p[i+1][0]+p[i+2][0])/5; o[i][1] = (p[i-2][1]+p[i-1][1]+p[i][1]+p[i+1][1]+p[i+2][1])/5; } return o; }
function finish(pts, stops){   // rota istatistikleri ve kümülatif eğriler
  const n = pts.length, d = new Float32Array(n), z = new Float32Array(n), sl = new Float32Array(n), cs = new Float32Array(n), cw = new Float32Array(n);
  let maxS = 0, climb = 0, aa = 0;
  for(let i=0;i<n;i++){ const c = pts[i][0], r = pts[i][1]; z[i] = elevAt(c,r); sl[i] = at(slopeA,c,r)/4; if(sl[i] > maxS) maxS = sl[i];
    if(i){ const ds = Math.hypot(c-pts[i-1][0], r-pts[i-1][1])*PXKM; d[i] = d[i-1] + ds; cs[i] = cs[i-1] + at(sciA,c,r)/255*ds; const wv = at(wearA,c,r)/255; cw[i] = cw[i-1] + wv*ds; if(wv >= .5) aa += ds; if(z[i] > z[i-1]) climb += z[i]-z[i-1]; } }
  return {pts, d, z, sl, cs, cw, len:d[n-1], maxS, climb, aa, sci:cs[n-1], wear:cw[n-1], stops:stops||[]};
}
const P = D.pois, R = {A:null, B:null};
let pen = 5, sel = "B", s = 0, layer = D.img.themis ? "themis" : "elev";
function planA(){ R.A = finish(smooth(astar(P.L0, P.B1, 0)), []); }
function planB(){ let pts = [], stops = []; const seq = [P.L0, P.S1, P.S2, P.S3, P.B1], nm = ["S1","S2","S3","B1"];
  for(let i=0;i<seq.length-1;i++){ const leg = smooth(astar(seq[i], seq[i+1], pen)); if(i) leg.shift(); pts = pts.concat(leg); stops.push([nm[i], pts.length-1]); }
  R.B = finish(pts, stops); R.B.stops = stops.map(x=> [x[0], R.B.d[x[1]]]); }
function planAll(){ planA(); planB(); }
function posAt(rt, dist){ const d = rt.d; let lo = 0, hi = d.length-1; while(hi-lo > 1){ const m = (lo+hi)>>1; if(d[m] <= dist) lo = m; else hi = m; }
  const t = d[hi] > d[lo] ? (dist-d[lo])/(d[hi]-d[lo]) : 0, a = rt.pts[lo], b = rt.pts[hi]; return {c:a[0]+(b[0]-a[0])*t, r:a[1]+(b[1]-a[1])*t, ang:Math.atan2(b[1]-a[1], b[0]-a[0]), i:lo, t}; }
const lerpArr = (arr, p)=> arr[p.i] + (arr[Math.min(arr.length-1,p.i+1)] - arr[p.i])*p.t;

// ---------------- three.js sahnesi
const glc = document.getElementById("gl");
const renderer = new THREE.WebGLRenderer({canvas:glc, antialias:true});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio||1));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x1a0f0b); scene.fog = new THREE.Fog(0x2a1810, 500, 1500);
const camera = new THREE.PerspectiveCamera(45, 1, 0.02, 3000);
scene.add(new THREE.AmbientLight(0xffffff, 0.92)); const sun = new THREE.DirectionalLight(0xffe2c4, 0.4); sun.position.set(-200, 300, -200); scene.add(sun);
const Y = z=> (z - zmin)/1000;   // gerçek ölçek: düşey abartı yok
const TS = 1, TW = Math.floor((W-1)/TS)+1, TH = Math.floor((H-1)/TS)+1;
const geom = new THREE.BufferGeometry(), pos = new Float32Array(TW*TH*3), uv = new Float32Array(TW*TH*2), idx = new Uint32Array((TW-1)*(TH-1)*6);
for(let r=0;r<TH;r++) for(let c=0;c<TW;c++){ const k = r*TW+c, gc = Math.min(W-1,c*TS), gr = Math.min(H-1,r*TS); pos[k*3] = gc*PXKM; pos[k*3+2] = gr*PXKM; uv[k*2] = gc/(W-1); uv[k*2+1] = 1 - gr/(H-1); }
let q = 0; for(let r=0;r<TH-1;r++) for(let c=0;c<TW-1;c++){ const a = r*TW+c, b = a+1, d2 = a+TW, e = d2+1; idx[q++] = a; idx[q++] = d2; idx[q++] = b; idx[q++] = b; idx[q++] = d2; idx[q++] = e; }
geom.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geom.setAttribute("uv", new THREE.BufferAttribute(uv, 2)); geom.setIndex(new THREE.BufferAttribute(idx, 1));
function setHeights(){ for(let r=0;r<TH;r++) for(let c=0;c<TW;c++) pos[(r*TW+c)*3+1] = Y(dem[Math.min(H-1,r*TS)*W + Math.min(W-1,c*TS)]); geom.attributes.position.needsUpdate = true; geom.computeVertexNormals(); geom.computeBoundingSphere(); }
setHeights();
const tex = {}; const loader = new THREE.TextureLoader();
for(const k in D.img){ tex[k] = loader.load(D.img[k]); tex[k].anisotropy = renderer.capabilities.getMaxAnisotropy(); }
const terrainMat = new THREE.MeshLambertMaterial({map:tex[layer], polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1}); scene.add(new THREE.Mesh(geom, terrainMat));
const skirt = new THREE.Mesh(new THREE.PlaneGeometry(W*PXKM*3, H*PXKM*3), new THREE.MeshBasicMaterial({color:0x140c09})); skirt.rotation.x = -Math.PI/2; skirt.position.set(W*PXKM/2, -0.5, H*PXKM/2); scene.add(skirt);
const V3 = (c,r,lift)=> new THREE.Vector3(c*PXKM, Y(elevAt(c,r)) + (lift||0), r*PXKM);
const strips = {}, COL = {A:0xff5a4d, B:0x39d5ff, T:0xffffff};
let stripW = 0.1;   // şerit genişliği (km): kamera uzaklığına göre ayarlanır, böylece hep ince ama görünür kalır
function setStrip(key, pts, width, color, opacity, order){   // araziye yapışık düz şerit
  if(strips[key]){ scene.remove(strips[key]); strips[key].geometry.dispose(); strips[key].material.dispose(); strips[key] = null; } if(!pts || pts.length < 2) return;
  const P2 = []; for(let i=0;i<pts.length-1;i++){ const a = pts[i], b = pts[i+1], n = Math.max(1, Math.ceil(Math.hypot(b[0]-a[0], b[1]-a[1])/0.6)); for(let k=0;k<n;k++) P2.push([a[0]+(b[0]-a[0])*k/n, a[1]+(b[1]-a[1])*k/n]); } P2.push(pts[pts.length-1]);
  const n = P2.length, pos = new Float32Array(n*6), idx = new Uint32Array((n-1)*6), hw = width/2/PXKM, lift = 0.006 + 0.004*order;
  for(let i=0;i<n;i++){ const a = P2[Math.max(0,i-1)], b = P2[Math.min(n-1,i+1)]; let dx = b[0]-a[0], dy = b[1]-a[1]; const l = Math.hypot(dx,dy)||1; dx /= l; dy /= l;
    const lc = P2[i][0] - dy*hw, lr = P2[i][1] + dx*hw, rc = P2[i][0] + dy*hw, rr = P2[i][1] - dx*hw;
    pos[i*6] = lc*PXKM; pos[i*6+1] = Y(elevAt(lc,lr)) + lift; pos[i*6+2] = lr*PXKM; pos[i*6+3] = rc*PXKM; pos[i*6+4] = Y(elevAt(rc,rr)) + lift; pos[i*6+5] = rr*PXKM;
    if(i < n-1){ const q = i*6, v = i*2; idx[q] = v; idx[q+1] = v+1; idx[q+2] = v+2; idx[q+3] = v+1; idx[q+4] = v+3; idx[q+5] = v+2; } }
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setIndex(new THREE.BufferAttribute(idx, 1));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({color, transparent:true, opacity, side:THREE.DoubleSide, depthWrite:false, polygonOffset:true, polygonOffsetFactor:-2-order, polygonOffsetUnits:-2-order}));
  m.renderOrder = order; strips[key] = m; scene.add(m); }
function drawRoutes(){ const o = sel === "A" ? "B" : "A";
  setStrip(o, R[o].pts, stripW*0.7, COL[o], .6, 1);
  setStrip("edge", R[sel].pts, stripW*1.45, 0x000000, .4, 2); setStrip(sel, R[sel].pts, stripW, COL[sel], 1, 3); drawTrail(); }
function drawTrail(){ const rt = R[sel], p = posAt(rt, s); setStrip("T", rt.pts.slice(0, p.i+1).concat([[p.c,p.r]]), stripW*0.62, 0xffffff, 1, 4); }
// duraklar
const POI = {L0:["L0 · İniş","#7dff8a"], S1:["S1 · Maar / base surge","#39ffd0"], S2:["S2 · Zeolit–sülfat alterasyonu","#39ffd0"], S3:["S3 · Evaporit kanalı","#39ffd0"], B1:["B1 · Lav tüpü üssü","#ffd23f"]};
const markers = [];
function label(text, col){ const cv = document.createElement("canvas"), g = cv.getContext("2d"); g.font = "700 30px sans-serif"; const w = Math.ceil(g.measureText(text).width) + 28; cv.width = w; cv.height = 48;
  g.font = "700 30px sans-serif"; g.fillStyle = "rgba(11,14,20,.85)"; g.beginPath(); g.roundRect(0,0,w,48,10); g.fill(); g.strokeStyle = col; g.lineWidth = 3; g.stroke(); g.fillStyle = col; g.textBaseline = "middle"; g.fillText(text, 14, 25);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv), depthTest:false})); sp.userData.asp = w/48; return sp; }
function buildMarkers(){ markers.forEach(m=>{ scene.remove(m.pole); scene.remove(m.sp); }); markers.length = 0;
  for(const k in POI){ const p = P[k], base = V3(p[0], p[1], 0), h = 1.6; const pole = new THREE.Mesh(new THREE.CylinderGeometry(.012,.012,h,6), new THREE.MeshBasicMaterial({color:POI[k][1]})); pole.position.set(base.x, base.y + h/2, base.z); scene.add(pole);
    const sp = label(POI[k][0], POI[k][1]); sp.position.set(base.x, base.y + h, base.z); sp.renderOrder = 5; scene.add(sp); markers.push({pole, sp}); } }
// gezici
const rover = new THREE.Group();
(function(){ const body = new THREE.Mesh(new THREE.BoxGeometry(2.0,.3,1.0), new THREE.MeshLambertMaterial({color:0xf4f4f4})); body.position.y = .62; rover.add(body);
  const deck = new THREE.Mesh(new THREE.BoxGeometry(.95,.05,1.15), new THREE.MeshLambertMaterial({color:0xc1440e})); deck.position.set(-.42,.80,0); rover.add(deck);
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(.035,.035,.75,8), new THREE.MeshLambertMaterial({color:0xdddddd})); mast.position.set(.7,1.15,.28); rover.add(mast);
  const head = new THREE.Mesh(new THREE.BoxGeometry(.3,.16,.2), new THREE.MeshLambertMaterial({color:0x222222})); head.position.set(.76,1.56,.28); rover.add(head);
  const wg = new THREE.CylinderGeometry(.3,.3,.16,14), wm = new THREE.MeshLambertMaterial({color:0x1c1c1c});
  [[-.8,.62],[0,.66],[.8,.62],[-.8,-.62],[0,-.66],[.8,-.62]].forEach(q2=>{ const w = new THREE.Mesh(wg, wm); w.rotation.x = Math.PI/2; w.position.set(q2[0], .3, q2[1]); rover.add(w); });
  const nose = new THREE.Mesh(new THREE.ConeGeometry(.16,.45,10), new THREE.MeshBasicMaterial({color:0xffd23f})); nose.rotation.z = -Math.PI/2; nose.position.set(1.22,.62,0); rover.add(nose); })();
rover.rotation.order = "YZX";   // önce yön (Y), sonra yunuslama (Z): burun yokuş yukarı kalkar, inişte iner
scene.add(rover);
const beacon = new THREE.Mesh(new THREE.CylinderGeometry(.03,.03,1,6), new THREE.MeshBasicMaterial({color:0xffffff, transparent:true, opacity:.55})); scene.add(beacon);

// ---------------- kamera
const cam = {az:Math.PI*0.75, pol:1.2, dist:24, follow:true, tx:0, ty:0, tz:0, overview:false};
function setOverview(on){ cam.overview = on; cam.follow = !on; if(on){ cam.dist = Math.max(W,H)*PXKM*1.0; cam.pol = 0.5; cam.az = Math.PI/2; } else { cam.dist = 24; cam.pol = 1.2; }
  document.getElementById("camO").classList.toggle("on", on); document.getElementById("camF").classList.toggle("on", !on); }
let drag = null;
glc.addEventListener("pointerdown", e=>{ glc.focus(); drag = {x:e.clientX, y:e.clientY}; glc.setPointerCapture(e.pointerId); });
glc.addEventListener("pointermove", e=>{ if(!drag) return; cam.az += (e.clientX-drag.x)*0.006; cam.pol = Math.min(1.55, Math.max(0.12, cam.pol - (e.clientY-drag.y)*0.005)); cam.follow = false; document.getElementById("camF").classList.remove("on"); drag = {x:e.clientX, y:e.clientY}; });
glc.addEventListener("pointerup", ()=>{ drag = null; });
glc.addEventListener("wheel", e=>{ e.preventDefault(); cam.dist = Math.min(900, Math.max(1.2, cam.dist*(e.deltaY > 0 ? 1.12 : 1/1.12))); }, {passive:false});
function resize(){ const b = glc.parentElement.getBoundingClientRect(); renderer.setSize(b.width, b.height, false); camera.aspect = b.width/Math.max(b.height,1); camera.updateProjectionMatrix(); }

// ---------------- paneller
const LAYERS = {themis:"THEMIS IR", viking:"Viking Renkli", elev:"MOLA Yükselti/Eğim", geo:"USGS Jeoloji", cost:"Analog Maliyet"};
function buildLeft(){
  const card = (k, title, sub)=>{ const r = R[k]; return `<div class="route ${sel===k?'sel':''}" data-k="${k}"><div class="nm"><span class="dot" style="background:${k==='A'?'#ff5a4d':'#39d5ff'}"></span>${title}</div>
    <div class="st">${sub}<br>Uzunluk <b>${fmt(r.len,1)} km</b> · en dik <b>${fmt(r.maxS,0)}°</b><br>Bilim <b>${fmt(r.sci,1)}</b> · aşınma <b>${fmt(r.wear,1)}</b> · 'a'a içinde <b>${fmt(r.aa,1)} km</b></div></div>`; };
  document.getElementById("routes").innerHTML = card("A", "Rota A — Salt topoğrafik (kör rota)", "Yalnız mesafe ve makro eğim: L0 → B1") + card("B", "Rota B — A.R.E.S. çift-analog entegre", "L0 → S1 → S2 → S3 → B1, 'a'a zonundan kaçınır");
  document.querySelectorAll("#routes .route").forEach(el=> el.onclick = ()=>{ if(sel !== el.dataset.k){ sel = el.dataset.k; s = 0; refreshAll(); } });
  document.getElementById("layers").innerHTML = Object.keys(LAYERS).filter(k=> D.img[k]).map(k=> `<button data-l="${k}" class="${k===layer?'on':''}">${LAYERS[k]}</button>`).join("");
  document.querySelectorAll("#layers button").forEach(b=> b.onclick = ()=>{ layer = b.dataset.l; terrainMat.map = tex[layer]; terrainMat.needsUpdate = true; buildLeft(); });
  const lg = (c,t)=> `<div class="lg"><span class="sw" style="background:${c}"></span>${t}</div>`;
  document.getElementById("legend").innerHTML = layer === "cost" ? lg("#eb4034","Divlit tipi 'a'a tekerlek aşınması") + lg("#28dcaa","Karapınar tipi bilim puanı") + lg("#963cc8","Eğim > 24° (geçilemez)")
    : layer === "geo" ? D.geoNames.map(g=> lg(g.color, g.unit+" — "+g.desc)).join("") : layer === "elev" ? lg("#d63031","Eğim > 15° (dikkat)") + `<div class="small">Renk: ${fmt(zmin)} m (yeşil) → ${fmt(zmax)} m (beyaz)</div>` : "";
  document.getElementById("penv").textContent = fmt(pen, pen%1 ? 1 : 0);
}
function analogAt(c, r){ const w = at(wearA,c,r)/2.55, sc = at(sciA,c,r)/2.55, sl = at(slopeA,c,r)/4, L = at(lfA,c,r), dB = Math.hypot(c-P.B1[0], r-P.B1[1])*PXKM;
  const near = ["S1","S2","S3"].map(k=> [k, Math.hypot(c-P[k][0], r-P[k][1])*PXKM]).sort((a,b)=> a[1]-b[1])[0];
  if(dB < 6 || (L === 1 && dB < 25 && sl > 6)) return {k:"divlit", col:"#ffd23f", t:"Divlit (Kula) — lav tüpü / çatlak", d:"Çatlak kenarı ve olası tüp açıklığı. Önerilen: <b>LiDAR + GPR</b> ile boşluk ve tavan kalınlığı taraması; kalıcı yeraltı üssü hedefi B1.", w, sc};
  if(w >= 50) return {k:"divlit", col:"#ff8a57", t:"Divlit (Kula) — bloklu 'a'a lavı", d:"Makro eğim düşük, mikro pürüzlülük yüksek. Önerilen: <b>LiDAR</b> ile blok haritalama, <b>tekerlek koruma</b> modu, düşük hız.", w, sc};
  if(sc >= 40 || near[1] < 7) return {k:"karapinar", col:"#39ffd0", t:"Karapınar (Konya) — " + (near[0]==="S1" ? "maar / base surge" : near[0]==="S2" ? "sulu alterasyon" : "evaporit kanalı"), d:"Hidrovolkanik / sulu alterasyon zemini. Önerilen: <b>Raman + LIBS</b> ile zeolit, kalsit, sülfat ve ikincil silika taraması.", w, sc};
  if(w >= 25) return {k:"divlit", col:"#e3c27a", t:"Divlit (Kula) — pürüzlü lav yüzeyi (orta)", d:"Orta düzey tekerlek aşınması. Önerilen: <b>LiDAR</b> ile yüzey takibi.", w, sc};
  return {k:"none", col:"#9aa6c4", t:"Belirgin analog eşleşmesi yok — düz volkanik ova", d:"Ne belirgin bilim hedefi ne tekerlek tehlikesi. Seyir modu; en yakın durak: <b>"+near[0]+"</b> ("+fmt(near[1],0)+" km).", w, sc}; }
function passAt(c, r){ const sl = at(slopeA,c,r)/4, w = at(wearA,c,r)/2.55;
  if(sl > 24) return ["Geçilemez: eğim > 24°", "#ff6b6b"]; if(sl > 15) return ["Dikkat: dik yamaç (> 15°)", "#ffb347"];
  if(w >= 50) return ["Dikkat: Divlit tipi bloklu 'a'a lavı (yüksek tekerlek hasarı)", sl < 6 ? "#ff8a57" : "#ff6b6b"]; if(w >= 25) return ["Orta: pürüzlü lav yüzeyi", "#ffe08a"];
  return [sl < 6 ? "İdeal" : "Uygun (6–15°)", sl < 6 ? "#7dff8a" : "#d7f06a"]; }
let moving = false;
function telemetry(){ const rt = R[sel], p = posAt(rt, s), ll = lonlat(p.c, p.r), g = at(geoA,p.c,p.r), gn = g > 0 ? D.geoNames[g-1] : null, pa = passAt(p.c,p.r), an = analogAt(p.c,p.r);
  const rows = [["X, Y (ızgara)", Math.round(p.c)+", "+Math.round(H-p.r)], ["X, Y (km)", fmt(p.c*PXKM,1)+", "+fmt((H-p.r)*PXKM,1)], ["Enlem / boylam", fmt(ll[1],3)+"° K, "+fmt(ll[0],3)+"° D"],
    ["MOLA yükselti", fmt(elevAt(p.c,p.r))+" m"], ["Makro eğim", fmt(at(slopeA,p.c,p.r)/4,1)+"°"], ["Engebelilik / TRI", fmt(at(triA,p.c,p.r)*2)+" m"], ["Yüzey pürüzlülüğü (MOLA 0,6 km)", fmt(at(roughA,p.c,p.r)/255,2)],
    ["Araç yunuslaması", (pitchS >= 0 ? "▲ " : "▼ ") + fmt(Math.abs(pitchS)*180/Math.PI,1) + "° " + (Math.abs(pitchS) < 0.004 ? "düz" : pitchS > 0 ? "yokuş yukarı" : "yokuş aşağı")], ["Topoğrafik yapı", D.lfNames[at(lfA,p.c,p.r)] || "—"], ["Jeolojik birim", gn ? gn.unit+" — "+gn.desc : "—"], ["Bilim puanı / aşınma", fmt(an.sc,0)+" / "+fmt(an.w,0)],
    ["Toplanan bilim / aşınma", fmt(lerpArr(rt.cs,p),1)+" / "+fmt(lerpArr(rt.cw,p),1)], ["Kat edilen / kalan", fmt(s,1)+" / "+fmt(rt.len-s,1)+" km"]];
  document.getElementById("tele").innerHTML = rows.map(x=> `<tr><td class="k">${x[0]}</td><td class="v">${x[1]}</td></tr>`).join("");
  const ps = document.getElementById("pass"); ps.textContent = pa[0]; ps.style.background = pa[1];
  const a = D.analog[an.k]; document.getElementById("analog").innerHTML = `<b class="t" style="color:${an.col}">${an.t}</b>${an.d}` + (a ? `<div class="small" style="margin-top:5px">${a.ad} · analoji skoru <b>${a.skor}</b>${a.ek ? " · "+a.ek : ""}</div>` : "");
  document.getElementById("analog").style.borderColor = an.col;
  const b = document.getElementById("an"); b.disabled = moving; document.getElementById("anh").textContent = moving ? "Gezici hareket ediyor; analiz için durun." : "Gezici duruyor. Çevresindeki 40 × 40 km için 8 boyutlu analizi açabilirsiniz."; }

// ---------------- grafikler
function prep(id){ const c = document.getElementById(id), dpr = window.devicePixelRatio||1, w = c.clientWidth, h = c.clientHeight; c.width = w*dpr; c.height = h*dpr; const g = c.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,w,h); g.font = "10px sans-serif"; return [g,w,h]; }
function chart1(){ const [g,w,h] = prep("c1"), rt = R[sel], L = 34, Rr = 30, T = 6, B = 16, X = d=> L + d/rt.len*(w-L-Rr); let a = Infinity, b = -Infinity; rt.z.forEach(v=>{ if(v<a) a = v; if(v>b) b = v; }); if(b-a < 50){ a -= 25; b += 25; }
  const Yz = z=> h-B - (z-a)/(b-a)*(h-T-B), SM = Math.max(30, rt.maxS*1.1), Ys = v=> h-B - v/SM*(h-T-B);
  g.fillStyle = "#2a2018"; g.beginPath(); g.moveTo(X(0), h-B); for(let i=0;i<rt.d.length;i++) g.lineTo(X(rt.d[i]), Yz(rt.z[i])); g.lineTo(X(rt.len), h-B); g.fill();
  g.strokeStyle = "#d8a36a"; g.lineWidth = 1.4; g.beginPath(); for(let i=0;i<rt.d.length;i++) i ? g.lineTo(X(rt.d[i]), Yz(rt.z[i])) : g.moveTo(X(rt.d[i]), Yz(rt.z[i])); g.stroke();
  g.strokeStyle = "#ff6b6b"; g.lineWidth = 1; g.beginPath(); for(let i=0;i<rt.d.length;i++) i ? g.lineTo(X(rt.d[i]), Ys(rt.sl[i])) : g.moveTo(X(rt.d[i]), Ys(rt.sl[i])); g.stroke();
  g.setLineDash([5,4]); g.strokeStyle = "#ffd23f"; g.beginPath(); g.moveTo(L, Ys(15)); g.lineTo(w-Rr, Ys(15)); g.stroke(); g.setLineDash([]); g.fillStyle = "#ffd23f"; g.fillText("15° kritik eşik", L+4, Ys(15)-3);
  (rt.stops||[]).forEach(st=>{ g.strokeStyle = "#39ffd066"; g.beginPath(); g.moveTo(X(st[1]), T); g.lineTo(X(st[1]), h-B); g.stroke(); g.fillStyle = "#39ffd0"; g.fillText(st[0], X(st[1])-12, T+9); });
  g.strokeStyle = "#fff"; g.lineWidth = 1.2; g.beginPath(); g.moveTo(X(s), T); g.lineTo(X(s), h-B); g.stroke();
  g.fillStyle = "#d8a36a"; g.fillText(fmt(b)+" m", 1, T+8); g.fillText(fmt(a)+" m", 1, h-B); g.fillStyle = "#ff6b6b"; g.textAlign = "right"; g.fillText(fmt(SM,0)+"°", w-2, T+8); g.fillText("0°", w-2, h-B); g.fillStyle = "#9aa6c4"; g.fillText(fmt(rt.len,0)+" km", w-Rr, h-3); g.textAlign = "left"; g.fillText("0", L, h-3);
  g.fillStyle = "#d8a36a"; g.fillText("— yükselti", w/2-70, h-3); g.fillStyle = "#ff6b6b"; g.fillText("— eğim", w/2+5, h-3); }
function chart2(){ const [g,w,h] = prep("c2"), L = 30, Rr = 30, T = 16, B = 16, maxL = Math.max(R.A.len, R.B.len), maxS = Math.max(R.A.sci, R.B.sci, 1), maxW = Math.max(R.A.wear, R.B.wear, 1), X = d=> L + d/maxL*(w-L-Rr);
  const line = (rt, arr, mx, col, dash)=>{ g.setLineDash(dash); g.strokeStyle = col; g.lineWidth = 1.7; g.beginPath(); for(let i=0;i<rt.d.length;i++){ const x = X(rt.d[i]), y = h-B - arr[i]/mx*(h-T-B); i ? g.lineTo(x,y) : g.moveTo(x,y); } g.stroke(); g.setLineDash([]); };
  line(R.A, R.A.cs, maxS, "#39ffd0", [5,4]); line(R.B, R.B.cs, maxS, "#39ffd0", []); line(R.A, R.A.cw, maxW, "#ff6b6b", [5,4]); line(R.B, R.B.cw, maxW, "#ff6b6b", []);
  const rt = R[sel]; g.strokeStyle = "#fff"; g.lineWidth = 1.2; g.beginPath(); g.moveTo(X(s), T); g.lineTo(X(s), h-B); g.stroke();
  g.fillStyle = "#39ffd0"; g.fillText("Bilim  A: "+fmt(R.A.sci,1)+"  B: "+fmt(R.B.sci,1), L, 10); g.fillStyle = "#ff6b6b"; g.fillText("Aşınma  A: "+fmt(R.A.wear,1)+"  B: "+fmt(R.B.wear,1), Math.max(L+140, w/2), 10);
  g.fillStyle = "#9aa6c4"; g.textAlign = "right"; g.fillText(fmt(maxL,0)+" km", w-Rr, h-3); g.textAlign = "left"; g.fillText("0", L, h-3);
  g.fillStyle = "#39ffd0"; g.fillText(fmt(maxS,0), 2, T+8); g.fillStyle = "#ff6b6b"; g.textAlign = "right"; g.fillText(fmt(maxW,0), w-2, T+8); g.textAlign = "left"; }
function refreshAll(){ buildLeft(); drawRoutes(); buildMarkers(); telemetry(); chart1(); chart2();
  document.getElementById("hint").innerHTML = "İlerlemek için <kbd>W</kbd>, geri için <kbd>S</kbd>. Sürükle: döndür · tekerlek: yakınlaş."; }
const srcEl = document.getElementById("src"); if(D.source === "trek"){ srcEl.textContent = "NASA Mars Trek — canlı veri"; srcEl.style.background = "#1d3b2a"; srcEl.style.color = "#7dff8a"; }
else if(D.source === "cache"){ srcEl.textContent = "Yerel önbellek (NASA verisi, çevrimdışı)"; srcEl.style.background = "#3b3418"; srcEl.style.color = "#ffd23f"; }
else { srcEl.textContent = "SENTETİK ARAZİ — NASA verisine ulaşılamadı"; srcEl.style.background = "#4a1d1d"; srcEl.style.color = "#ff8a8a"; }

// ---------------- girişler
const keys = {};
function key(e, down){ const k = e.key.toLowerCase(); if(k === "w" || k === "arrowup"){ keys.w = down; e.preventDefault(); } if(k === "s" || k === "arrowdown"){ keys.s = down; e.preventDefault(); } }
window.addEventListener("keydown", e=> key(e,true)); window.addEventListener("keyup", e=> key(e,false)); window.addEventListener("blur", ()=>{ keys.w = keys.s = false; });
const hold = (id, k)=>{ const b = document.getElementById(id), up = ()=>{ keys[k] = false; }; b.addEventListener("pointerdown", e=>{ e.preventDefault(); keys[k] = true; }); ["pointerup","pointerleave","pointercancel"].forEach(ev=> b.addEventListener(ev, up)); };
hold("bw","w"); hold("bs","s");
let penTimer = null;
document.getElementById("pen").addEventListener("input", e=>{ pen = +e.target.value; document.getElementById("penv").textContent = fmt(pen, pen%1 ? 1 : 0); clearTimeout(penTimer);
  penTimer = setTimeout(()=>{ const frac = sel === "B" && R.B.len ? s/R.B.len : null; planB(); if(frac !== null) s = frac*R.B.len; refreshAll(); }, 90); });
document.getElementById("camF").onclick = ()=>{ setOverview(false); cam.follow = true; glc.focus(); };
document.getElementById("camO").onclick = ()=>{ setOverview(true); glc.focus(); };
document.getElementById("an").onclick = ()=>{ if(moving) return; const rt = R[sel], p = posAt(rt, s), ll = lonlat(p.c,p.r), an = analogAt(p.c,p.r), pa = passAt(p.c,p.r), g = at(geoA,p.c,p.r);
  document.getElementById("busy").style.display = "flex";
  post("streamlit:setComponentValue", {dataType:"json", value:{evt:"analyze", nonce:Date.now(), route:sel, s:s, pen:pen, layer:layer, lon:ll[0], lat:ll[1], c:p.c, r:p.r, elev:elevAt(p.c,p.r), slope:at(slopeA,p.c,p.r)/4,
    tri:at(triA,p.c,p.r)*2, rough:at(roughA,p.c,p.r)/255, lf:at(lfA,p.c,p.r), geo:g > 0 ? D.geoNames[g-1].unit : "—", sci:an.sc, wear:an.w, analog:an.t, pass:pa[0], len:rt.len, got_sci:lerpArr(rt.cs,p), got_wear:lerpArr(rt.cw,p)}}); };

// ---------------- döngü
let tPrev = performance.now(), tUi = 0, tTrail = 0, heading = 0, pitchS = 0;
function tick(t){ const dt = Math.min(0.1, (t-tPrev)/1000); tPrev = t; moving = false; const rt = R[sel];
  if(rt){ const dir = (keys.w?1:0) - (keys.s?1:0);
    if(dir){ const p0 = posAt(rt, s), sl = at(slopeA,p0.c,p0.r)/4, w = at(wearA,p0.c,p0.r)/255, v = 4.2/(1 + Math.pow(sl/8,2) + 1.5*w); const ns = Math.min(rt.len, Math.max(0, s + dir*v*dt)); moving = ns !== s; s = ns; }
    const p = posAt(rt, s), wp = V3(p.c, p.r, 0), sc = Math.min(1.6, Math.max(.05, cam.dist*0.016));
    const pa = posAt(rt, Math.min(rt.len, s + 0.7)), pb = posAt(rt, Math.max(0, s - 0.7)), dsk = Math.hypot(pa.c-pb.c, pa.r-pb.r)*PXKM;
    const pitch = dsk > 1e-6 ? Math.atan2((elevAt(pa.c,pa.r) - elevAt(pb.c,pb.r))/1000, dsk) : 0; pitchS += (pitch - pitchS)*Math.min(1, dt*6);
    heading = p.ang; rover.position.copy(wp); rover.scale.setScalar(sc); rover.rotation.y = -heading; rover.rotation.z = pitchS;
    beacon.position.set(wp.x, wp.y + sc*4.2, wp.z); beacon.scale.set(sc, sc*5, sc);
    const wantW = Math.min(0.9, Math.max(0.04, cam.dist*0.005)); if(Math.abs(Math.log(wantW/stripW)) > 0.15){ stripW = wantW; drawRoutes(); }
    if(cam.overview){ cam.tx = W*PXKM/2; cam.ty = 0; cam.tz = H*PXKM/2; } else { cam.tx += (wp.x-cam.tx)*.2; cam.ty += (wp.y-cam.ty)*.2; cam.tz += (wp.z-cam.tz)*.2;
      if(cam.follow){ const want = heading + Math.PI; let da = want - cam.az; while(da > Math.PI) da -= 2*Math.PI; while(da < -Math.PI) da += 2*Math.PI; cam.az += da*Math.min(1, dt*1.6); } }
    const sp = Math.sin(cam.pol); camera.position.set(cam.tx + cam.dist*sp*Math.cos(cam.az), cam.ty + cam.dist*Math.cos(cam.pol), cam.tz + cam.dist*sp*Math.sin(cam.az));
    const gy = Y(elevAt(camera.position.x/PXKM, camera.position.z/PXKM)) + 0.08; if(camera.position.y < gy) camera.position.y = gy;
    camera.lookAt(cam.tx, cam.ty, cam.tz);
    markers.forEach(m=>{ const d = camera.position.distanceTo(m.sp.position), k = Math.max(.25, d*0.026), pk = Math.max(1, d*0.09); m.sp.scale.set(k*m.sp.userData.asp, k, 1); m.pole.scale.set(pk, 1, pk); });
    if(moving && t - tTrail > 160){ tTrail = t; drawTrail(); }
    if(t - tUi > 110){ tUi = t; telemetry(); if(moving){ chart1(); chart2(); } }
    else if(!moving && tUi && t - tUi > 100) telemetry(); }
  renderer.render(scene, camera); requestAnimationFrame(tick); }
window.addEventListener("resize", ()=>{ resize(); chart1(); chart2(); });
resize(); planAll(); refreshAll(); cam.tx = P.L0[0]*PXKM; cam.tz = P.L0[1]*PXKM; cam.ty = Y(elevAt(P.L0[0], P.L0[1])); requestAnimationFrame(tick);
window.__sim = {get s(){ return s; }, get sel(){ return sel; }, R, cam};
</script></body></html>
"""


@st.cache_resource(show_spinner=False)
def sim3d_component(n: int = 450):
    import hashlib, json, os, tempfile
    import streamlit.components.v1 as components
    html = SIM3D_HTML.replace("__THREE__", three_js_source()).replace("__DATA__", json.dumps(sim3d_payload(n), ensure_ascii=False))
    d = os.path.join(tempfile.gettempdir(), "volkanik_sim3d_" + hashlib.md5(html.encode("utf-8")).hexdigest()[:12])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return components.declare_component("mars_sim_3d", path=d)


def _nz(v, d=1, unit=""):
    """Sayıyı yazar; çok küçük değerleri 0 yerine '< eşik' olarak gösterir."""
    thr = 10 ** (-d)
    if not np.isfinite(v):
        return "veri yok"
    txt = f"< {thr:g}" if abs(v) < thr else f"{v:.{d}f}"
    return txt.replace(".", ",") + (f" {unit}" if unit else "")


def sim3d_analysis():
    """Gezicinin durduğu noktanın çevresindeki 40 × 40 km pencere için 8 boyutlu analiz ekranı."""
    ss = st.session_state
    p = ss["sim3_point"]
    # Başlık ve geri dönüş düğmesi hesaplamalardan ÖNCE çizilir: böylece analiz sürerken de anında geri dönülebilir
    c1_, c2_ = st.columns([3, 1])
    with c1_:
        st.title("🔬 Detaylı Analiz — 40 × 40 km pencere")
        st.markdown(f"**Rota {p['route']}** · kat edilen yol **{p['s']:.1f} / {p['len']:.1f} km** · konum **{float(p['lat']):.3f}° K, {float(p['lon']):.3f}° D** · "
                    f"jeolojik birim **{p['geo']}** · toplanan bilim **{p['got_sci']:.1f}**, aşınma **{p['got_wear']:.1f}**")
    with c2_:
        st.write("")
        if st.button("← Simülasyona Dön", type="primary", key="sim3_back_top"):
            ss["sim3_view"] = "sim"
            st.rerun()
    set_body("mars")
    L = sim3d_layers()
    lat, lon = float(p["lat"]), float(p["lon"])
    hl = 20_000 / BODY["m_lat"]
    hx = hl / math.cos(math.radians(lat))
    bbox = (round(lon - hx, 4), round(lat - hl, 4), round(lon + hx, 4), round(lat + hl, 4))
    R, Wd, Hd, rpx = L["bbox"], L["dem"].shape[1], L["dem"].shape[0], L["px"]
    c0 = int(np.clip((bbox[0] - R[0]) / (R[2] - R[0]) * Wd, 0, Wd - 8)); c1 = int(np.clip((bbox[2] - R[0]) / (R[2] - R[0]) * Wd, c0 + 8, Wd))
    r0 = int(np.clip((R[3] - bbox[3]) / (R[3] - R[1]) * Hd, 0, Hd - 8)); r1 = int(np.clip((R[3] - bbox[1]) / (R[3] - R[1]) * Hd, r0 + 8, Hd))
    win = lambda a: a[r0:r1, c0:c1]
    try:
        if L["source"] != "trek":
            raise RuntimeError("çevrimdışı")
        dem, px = get_mars_dem(bbox, 300)
        dem_src = "MOLA + HRSC (yerel pencere, canlı)"
    except Exception:
        sub = win(L["dem"])
        dem = ndi.zoom(sub, 300 / max(sub.shape), order=1).astype("float32")
        px = 40_000 / max(dem.shape)
        dem_src = "bölgesel yükseklik modelinden kırpma"
    slope, _ = slope_normals(dem, px)
    tri = roughness_tri(dem)
    shade = LightSource(315, 45).hillshade(dem, vert_exag=3, dx=px, dy=px)
    tes = {k: float(np.nanmean(win(v))) for k, v in L["tes"].items()}
    tes_reg = {k: float(np.nanmean(v)) for k, v in L["tes"].items()}
    sci_w, wear_w, rough_w, acc_w, twi_w = win(L["sci"]), win(L["wear"]), win(L["rough06"]), win(L["acc"]), win(L["twi"])
    geo_w = win(L["geo"])
    names = L["geoNames"]
    unit = names[int(np.bincount(geo_w.ravel()).argmax()) - 1] if geo_w.max() > 0 else "?"
    volc_share = 100 * np.isin(geo_w, [i + 1 for i, nm in enumerate(names) if "v" in nm.lower()]).mean()
    P = mars_pressure(dem)
    q = np.array([insolation(lat, l, "mars") for l in range(0, 360, 5)])
    teq = lambda v: (max(v, 1e-6) * 0.75 / (0.95 * 5.670e-8)) ** 0.25 - 273.15
    opn = openness(dem, px, radius_m=1500)
    hw = resize_to(illumination(dem, px, lat, 355)[0], dem.shape)
    k = max(3, int(round(150 / px)) | 1)
    safe = (slope <= 5) & (ndi.maximum_filter(slope, size=k) <= 8)
    dist, sites = landing_sites(safe, px, bbox)
    pois = L["pois"]
    dkm = lambda rc: math.hypot(rc[0] - p["r"], rc[1] - p["c"]) * rpx / 1000
    near_s = min((dkm(pois[k_]), k_) for k_ in ("S1", "S2", "S3"))
    tube = 100 * ((slope >= 0.3) & (slope <= 6)).mean()
    hyd = float(np.clip(100 * (0.45 * np.clip(tes["glass"] / 0.3, 0, 1) + 0.35 * sci_w.mean() / 100 * 2 + 0.20 * np.clip((twi_w.mean() - 6) / 8, 0, 1)), 0, 100))
    s_water, s_press = math.exp(-near_s[0] / 25), float(np.clip((P.mean() - 400) / 400, 0, 1))
    s_light, s_dust = float(hw.mean() / max(hw.max(), 1e-6)), 1 - tes["dust"]
    hab = 100 * (0.45 * s_water + 0.25 * s_press + 0.15 * s_light + 0.15 * s_dust)
    ext = [bbox[0], bbox[2], bbox[1], bbox[3]]

    if L["source"] == "synthetic":
        st.error("NASA Mars Trek'e ulaşılamadı ve önbellek yok: bu ekrandaki değerler **sentetik araziye** aittir, gerçek Mars verisi değildir.")
    elif L["source"] == "cache":
        st.warning("NASA Mars Trek'e ulaşılamadı; değerler daha önce indirilen yerel önbellekten hesaplandı.")
    m = st.columns(5)
    m[0].metric("MOLA yükselti", f"{p['elev']:.0f} m")
    m[1].metric("Makro eğim", _nz(p["slope"], 1, "°"))
    m[2].metric("Engebelilik / TRI", _nz(p["tri"], 0, "m"))
    m[3].metric("Yüzey pürüzlülüğü", _nz(p["rough"], 2))
    m[4].metric("Geçilebilirlik", p["pass"].split(":")[0])
    st.info(f"**Aktif Türkiye analoğu:** {p['analog']}")
    a1, a2 = st.columns(2)
    for col, key in ((a1, "karapinar"), (a2, "divlit")):
        a = SIM3_ANALOG[key]
        col.markdown(f"**{a['ad']}** — analoji skoru **{a['skor']}**{' · ' + a['ek'] if a['ek'] else ''}  \n"
                     f"<small>{a['surec']}. Önerilen yük: {a['sensor']}.</small>", unsafe_allow_html=True)
    st.caption("Analoji skorları ve saha tanımları sizin verdiğiniz değerlerdir. 'Model' ya da 'indeks' yazan ölçütler ölçüm değil, "
               "yükseklik modeli, TES haritaları ve analog katmanlarından hesaplanan göreli tahminlerdir. Bu bölgede katalogda "
               "kayıtlı sulu mineral tespiti, vadi ağı ya da kumul yoktur; ilgili ölçütler bu yüzden model indeksiyle verilir. "
               f"Yükseklik kaynağı: {dem_src}, piksel ≈ {px:.0f} m.")

    def fig_map(arr, title, cmap, label, vmin=None, vmax=None, mark=True):
        fig, ax = map_fig(arr, bbox, title, cmap=cmap, label=label, vmin=vmin, vmax=vmax)
        if mark:
            ax.plot(lon, lat, "o", ms=9, mfc="none", mec="#00e5ff", mew=2)
            ax.plot(lon, lat, "+", ms=9, mec="#00e5ff", mew=1.5)
        return fig

    def bars(names_, vals, colors, title, xlabel, ref=None):
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        y = np.arange(len(names_))
        ax.barh(y, vals, color=colors, edgecolor="#555", lw=0.5, height=0.55 if ref is None else 0.38)
        if ref is not None:
            ax.barh(y + 0.4, ref, color="#b8bcc6", edgecolor="#555", lw=0.5, height=0.38, label="Bölge ortalaması")
            ax.legend(fontsize=7, frameon=False)
        ax.set_yticks(y + (0.2 if ref is not None else 0), names_, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        return fig

    tabs = st.tabs(["1 · Jeoloji ve mineraloji", "2 · Topoğrafya", "3 · Sıcaklık ve aydınlatma", "4 · Su, buz, kaynaklar",
                    "5 · Toz, rüzgâr, erozyon", "6 · Biyoloji ve biyo-imza", "7 · Yeraltı ve lav tüpleri", "8 · İniş ve hareketlilik"])
    with tabs[0]:
        st.subheader("Jeoloji, mineraloji ve yüzey malzemeleri")
        m = st.columns(4)
        m[0].metric("Baskın jeolojik birim", unit, help=SIM_GEO_TR.get(unit, L["geoDesc"].get(unit, "")))
        m[1].metric("TES yüksek-Ca piroksen (göreli)", _nz(tes["pyrox"], 2))
        m[2].metric("TES tabaka silikat / Si-cam (göreli)", _nz(tes["glass"], 2))
        m[3].metric("Hidratlı mineral olasılık indeksi (model)", _nz(hyd, 0, "/ 100"))
        a, b = st.columns(2)
        with a:
            vk = mlayer("Mars_Viking_MDIM21_ClrMosaic_global_232m", bbox, dem.shape) if L["source"] == "trek" else None
            if vk is not None:
                fig, ax = base_map(bbox, vk, "Viking renk mozaiği (yerel pencere)")
                ax.plot(lon, lat, "o", ms=9, mfc="none", mec="#00e5ff", mew=2)
                show(fig)
            else:
                show(fig_map(shade, "Gölgeli kabartma (yerel pencere)", "gray", "aydınlanma"))
        with b:
            nm = ["Yüksek-Ca piroksen", "Plajiyoklaz", "Tabaka silikat / Si-cam", "Albedo", "Isıl atalet"]
            ks = ["pyrox", "plag", "glass", "albedo", "ti"]
            show(bars(nm, [tes[k_] for k_ in ks], "#c1440e", "TES yüzey bileşimi: pencere ve bölge", "göreli değer (0–1)",
                      ref=[tes_reg[k_] for k_ in ks]))
        st.caption(f"Birim: {unit} — {SIM_GEO_TR.get(unit, L['geoDesc'].get(unit, ''))} (USGS SIM 3292). Hidratlı mineral indeksi, TES tabaka "
                   "silikat haritası, Karapınar bilim katmanı ve topoğrafik nemlilikten hesaplanır; OMEGA/CRISM tespiti değildir.")
    with tabs[1]:
        st.subheader("Arazi, yükseklik, eğim, pürüzlülük, kraterler")
        m = st.columns(4)
        m[0].metric("Rölyef", f"{dem.max() - dem.min():.0f} m")
        m[1].metric("Ortalama eğim", _nz(float(slope.mean()), 1, "°"))
        m[2].metric("Ortalama TRI", _nz(float(tri.mean()), 1, "m"))
        m[3].metric("MOLA pürüzlülük (0,6 km, göreli)", _nz(float(rough_w.mean()), 2))
        a, b = st.columns(2)
        with a:
            fig, ax = map_fig(dem, bbox, "Yükseklik (gölgeli)", cmap=ELEV_CMAP, label="m", vmin=float(dem.min()), vmax=float(dem.max()), shade=shade)
            ax.plot(lon, lat, "o", ms=9, mfc="none", mec="#00e5ff", mew=2)
            show(fig)
        with b:
            fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
            ax.hist(slope.ravel(), bins=np.linspace(0, max(30, float(np.percentile(slope, 99.9))), 45), color="#b5651d",
                    weights=np.full(slope.size, 100 / slope.size))
            for thr, col, txt in ((6, "#3a9a5c", "6°"), (15, "#d4a017", "15° dikkat"), (24, "#b2301e", "24° sınır")):
                ax.axvline(thr, color=col, lw=1.2, ls="--")
                ax.annotate(txt, (thr, ax.get_ylim()[1] * 0.9), fontsize=7, color=col, xytext=(3, 0), textcoords="offset points")
            ax.set_yscale("log")
            ax.set_xlabel("Eğim (°)")
            ax.set_ylabel("Alan payı (%, logaritmik)")
            ax.set_title("Eğim dağılımı", loc="left")
            show(fig)
        show(fig_map(slope, "Eğim haritası", "magma_r", "derece", 0, float(max(15, np.percentile(slope, 99.5)))))
    with tabs[2]:
        st.subheader("Sıcaklık, aydınlatma ve çevresel aşırılıklar")
        m = st.columns(4)
        m[0].metric("Yıllık ort. güneş ışınımı", f"{q.mean():.0f} W/m²")
        m[1].metric("Tahmini günlük ort. sıcaklık (en sıcak)", f"{teq(q.max()):.0f} °C")
        m[2].metric("Tahmini günlük ort. sıcaklık (en soğuk)", f"{teq(q.min()):.0f} °C")
        m[3].metric("TES ısıl atalet (göreli)", _nz(tes["ti"], 2))
        a, b = st.columns(2)
        with a:
            show(fig_map(hw, "Doğrudan güneş süresi — kuzey kış gündönümü", "YlOrBr_r", "saat / sol", 0, float(hw.max())))
        with b:
            ls = np.arange(0, 360, 5)
            fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
            ax.plot(ls, q, color="#c1440e", lw=1.8, label="Güneş ışınımı")
            ax.set_xlabel("Güneş boylamı Ls (°)")
            ax.set_ylabel("Günlük ort. ışınım (W/m²)", color="#c1440e")
            a2_ = ax.twinx()
            a2_.plot(ls, [teq(v) for v in q], color="#2b6cb0", lw=1.4, ls="--")
            a2_.set_ylabel("Tahmini günlük ort. sıcaklık (°C)", color="#2b6cb0")
            a2_.spines["right"].set_visible(True)
            ax.set_title("Mevsimlik ışınım ve ışınım dengesi sıcaklığı", loc="left")
            show(fig)
        st.caption("Sıcaklıklar ölçüm değildir: atmosfer üstü güneş ışınımından kaba ışınım dengesiyle (albedo 0,25) hesaplanmıştır. "
                   "Gerçekte gündüz–gece farkı 60–100 °C'yi bulur.")
    with tabs[3]:
        st.subheader("Su, buz, permafrost ve diğer potansiyel kaynaklar")
        m = st.columns(4)
        m[0].metric("Ortalama yüzey basıncı", f"{P.mean():.0f} Pa")
        m[1].metric("Üçlü nokta üstü alan (> 611 Pa)", _nz(100 * float((P > 611).mean()), 0, "%"))
        m[2].metric("Ort. topoğrafik nemlilik (TWI)", _nz(float(twi_w.mean()), 1))
        m[3].metric("En büyük katkı alanı (kanal)", _nz(float(acc_w.max()), 1, "km²"))
        a, b = st.columns(2)
        with a:
            fig, ax = map_fig(np.log10(np.maximum(resize_to(acc_w, dem.shape), 0.01)), bbox, "Akış birikimi (eski akış yolları)", cmap="GnBu", label="log₁₀ km²")
            ax.plot(lon, lat, "o", ms=9, mfc="none", mec="#c1440e", mew=2)
            show(fig)
        with b:
            res = [("Güneş enerjisi (kış güneşi)", 100 * s_light, "#e9a23b"), ("Sıvı su için basınç", 100 * s_press, "#2b6cb0"),
                   ("Hidratlı mineral indeksi (model)", hyd, "#2ca25f"), ("Yapı malzemesi: volkanik birim payı", volc_share, "#5a2a27")]
            show(bars([r_[0] for r_ in res], [max(r_[1], 0.5) for r_ in res], [r_[2] for r_ in res], "Kaynak potansiyeli (göreli)", "0–100"))
        st.caption(f"Enlem {lat:.1f}°: bu enlemde yüzeye yakın kalıcı yer buzu günümüz ikliminde beklenmez; su kaynağı olarak hidratlı "
                   "mineraller ve derindeki buz değerlendirilir. Basınç, yükseklikten 10,8 km ölçek yüksekliğiyle hesaplanmıştır.")
    with tabs[4]:
        st.subheader("Toz, rüzgâr, erozyon ve diğer yüzey süreçleri")
        ls_f = ((np.minimum(np.maximum(resize_to(acc_w, dem.shape) * 1e6 / px, px), 5000) / 22.13) ** 0.4
                * (np.maximum(np.sin(np.radians(slope)), 1e-3) / 0.0896) ** 1.3)
        m = st.columns(4)
        m[0].metric("TES toz örtüsü (göreli)", _nz(tes["dust"], 2))
        m[1].metric("TES albedo (göreli)", _nz(tes["albedo"], 2))
        m[2].metric("Ortalama açıklık", f"{opn.mean():.1f}°")
        m[3].metric("Ort. yamaç etkeni (LS)", _nz(float(ls_f.mean()), 2))
        a, b = st.columns(2)
        with a:
            show(fig_map(opn, "Topoğrafik açıklık (rüzgâra maruziyet)", "viridis", "derece (90 = tam açık)", float(np.percentile(opn, 1)), 90))
        with b:
            nm = ["Toz örtüsü", "Albedo", "Isıl atalet"]
            ks = ["dust", "albedo", "ti"]
            show(bars(nm, [tes[k_] for k_ in ks], "#b98b6a", "Toz göstergeleri: pencere ve bölge", "göreli değer (0–1)", ref=[tes_reg[k_] for k_ in ks]))
        st.caption("Ölçülmüş rüzgâr hızı bu kaynaklarda yoktur. Yüksek toz + yüksek albedo + düşük ısıl atalet kalın gevşek toz örtüsüne işaret eder.")
    with tabs[5]:
        st.subheader("Biyoloji, ekstremofiller, yaşanabilirlik ve biyolojik imzalar")
        m = st.columns(4)
        m[0].metric("Göreli yaşanabilirlik puanı (model)", _nz(hab, 0, "/ 100"))
        m[1].metric("Karapınar bilim puanı (pencere en yükseği)", _nz(float(sci_w.max()), 0, "/ 100"))
        m[2].metric("En yakın bilim durağı", f"{near_s[1]} · {near_s[0]:.0f} km")
        m[3].metric("Biyo-imza korunma indeksi (model)", _nz(0.6 * hyd + 0.4 * 100 * s_dust, 0, "/ 100"))
        a, b = st.columns(2)
        with a:
            show(fig_map(resize_to(sci_w, dem.shape), "Karapınar tipi bilim puanı katmanı", "YlGn", "0–100", 0, 100))
        with b:
            parts = [("Su izine / bilim durağına yakınlık", 100 * s_water, "#2ca25f"), ("Basınç (alçak arazi)", 100 * s_press, "#2b6cb0"),
                     ("Kış ışığı", 100 * s_light, "#e9a23b"), ("Az toz", 100 * s_dust, "#b98b6a")]
            show(bars([r_[0] for r_ in parts], [max(r_[1], 0.5) for r_ in parts], [r_[2] for r_ in parts], "Yaşanabilirliği sınırlayan etkenler", "0–100"))
        st.caption("Mars'ta doğrulanmış biyolojik imza yoktur. Puanlar Karapınar analoğuna dayanır: sulu alterasyon mineralleri ve evaporit "
                   "çökelleri organik maddeyi koruyabildiği için, o tür zemine benzeyen yerler yüksek puan alır. Önerilen yük: Raman + LIBS.")
    with tabs[6]:
        st.subheader("Mağaralar, lav tüpleri ve yeraltı ortamları")
        fav = np.clip(np.minimum((slope - 0.2) / 0.8, (8 - slope) / 3), 0, 1) * np.clip(1 - tri / max(float(np.percentile(tri, 95)), 1e-6), 0, 1)
        m = st.columns(4)
        m[0].metric("Tüpe elverişli eğim payı (0,3–6°)", _nz(tube, 0, "%"))
        m[1].metric("Volkanik birim payı", _nz(float(volc_share), 0, "%"))
        m[2].metric("B1 lav tüpü hedefine uzaklık", f"{dkm(pois['B1']):.0f} km")
        m[3].metric("Ort. tüp elverişlilik indeksi (model)", _nz(100 * float(fav.mean()), 0, "/ 100"))
        a, b = st.columns(2)
        with a:
            show(fig_map(fav, "Lav tüpü elverişlilik indeksi", "inferno", "0–1", 0, 1))
        with b:
            fig, ax = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
            ax.hist(slope.ravel(), bins=np.linspace(0, 20, 41), color="#7a1f1f", weights=np.full(slope.size, 100 / slope.size))
            ax.axvspan(0.3, 6, color="#ffd23f", alpha=0.25, lw=0)
            ax.set_xlabel("Eğim (°)")
            ax.set_ylabel("Alan payı (%)")
            ax.set_title("Eğim dağılımı (sarı: tüp oluşumuna elverişli aralık)", loc="left")
            show(fig)
        st.caption("Divlit analoğu: 3D haritalanmış lav tünelleri hafif eğimli bazalt akıntılarında gelişir. Bu piksel boyunda tüp girişleri "
                   "görünmez; doğrulama için LiDAR ve yer radarı (GPR) gerekir. B1, Cerberus Fossae çatlağında seçilmiş üs hedefidir.")
    with tabs[7]:
        st.subheader("İniş, hareketlilik, navigasyon ve inşaat koşulları")
        m = st.columns(4)
        m[0].metric("İnişe uygun alan (≤ 5°)", _nz(100 * float(safe.mean()), 0, "%"))
        m[1].metric("En büyük engelsiz daire", f"{int(sites['Engelsiz yarıçap (m)'].iloc[0])} m" if len(sites) else "< 1 piksel")
        m[2].metric("Divlit tekerlek aşınma indeksi (ort.)", _nz(float(wear_w.mean()), 0, "/ 100"))
        m[3].metric("Tekerlekli araca açık (≤ 15°)", _nz(100 * float((slope <= 15).mean()), 0, "%"))
        a, b = st.columns(2)
        with a:
            show(fig_map(resize_to(wear_w, dem.shape), "Divlit tipi 'a'a tekerlek aşınma katmanı", "OrRd", "0–100", 0, 100))
        with b:
            MOB = {1: ("≤ 6° ideal", "#3a9a5c"), 2: ("6–15° uygun", "#b7d98a"), 3: ("15–24° dikkat", "#f1cf6b"), 4: ("> 24° geçilemez", "#b2301e")}
            fig = class_fig((np.digitize(slope, [6, 15, 24]) + 1).astype("uint8"), bbox, "Eğime göre geçilebilirlik", MOB)
            fig.axes[0].plot(lon, lat, "o", ms=9, mfc="none", mec="#00e5ff", mew=2)
            show(fig)
        s_ = np.sort(slope.ravel())
        fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
        ax.plot(s_, np.linspace(0, 100, s_.size), color="#333", lw=1.5)
        for thr, col in ((6, "#3a9a5c"), (15, "#d4a017"), (24, "#b2301e")):
            ax.axvline(thr, color=col, lw=1.1, ls="--")
            ax.annotate(f"{thr}°: %{100 * (slope <= thr).mean():.0f}", (thr, 12), fontsize=7.5, color=col, xytext=(3, 0), textcoords="offset points")
        ax.set_xlim(0, 30)
        ax.set_xlabel("Eğim (°)")
        ax.set_ylabel("Bu eğimin altındaki alan (%)")
        ax.set_title("Kümülatif eğim eğrisi", loc="left")
        show(fig)
        st.caption("Tekerlek aşınma katmanı, MOLA'nın 0,6 km tabanlı pürüzlülüğünün düz (< 6°) volkanik zeminde yüksek olduğu yerlerdir: "
                   "Divlit'teki 'düz görünen ama keskin bloklu 'a'a' durumunun yörüngeden görülebilen en yakın karşılığı. Metre ölçeğindeki "
                   "bloklar bu veride doğrudan ölçülmez.")
    st.divider()
    if st.button("← Simülasyona Dön", type="primary", key="sim3_back_bottom"):
        ss["sim3_view"] = "sim"
        st.rerun()


def sim3d_page():
    ss = st.session_state
    if ss.get("sim3_view") == "analysis" and ss.get("sim3_point"):
        sim3d_analysis()
        return
    set_body("mars")
    st.title("🎮 3D Rota Oluşturma Simülasyonu — A.R.E.S. çift-analog gezici rotası")
    st.caption("Mars · Elysium Planitia – Cerberus Fossae (9,64° K, 161,98° D çevresi). Arazi MOLA + HRSC yükseklik modelinden, yüzey dokuları "
               "NASA Mars Trek'ten (THEMIS IR, Viking, USGS SIM 3292) gelir. Yer gerçeği analogları: Karapınar (bilim durakları S1–S3) ve "
               "Divlit / Kula (tekerlek aşınma zonları, lav tüpü üssü B1).")
    try:
        with st.spinner("3D arazi ve analog katmanları hazırlanıyor (ilk açılışta 15–30 sn)…"):
            comp = sim3d_component()
    except Exception as e:
        st.error(f"3D simülasyon hazırlanamadı. İnternet bağlantınızı kontrol edin.\n\n{e}")
        return
    val = comp(route=ss.get("sim3_route"), s=float(ss.get("sim3_s", 0.0)), pen=ss.get("sim3_pen"), layer=ss.get("sim3_layer"),
               key="mars_sim_3d", default=None)
    if isinstance(val, dict) and val.get("evt") == "analyze" and val.get("nonce") != ss.get("sim3_nonce"):
        ss["sim3_nonce"] = val["nonce"]
        ss["sim3_route"], ss["sim3_s"], ss["sim3_pen"], ss["sim3_layer"] = val["route"], float(val["s"]), float(val["pen"]), val["layer"]
        ss["sim3_point"] = val
        ss["sim3_view"] = "analysis"
        st.rerun()
    with st.expander("Bu simülasyon nasıl kuruldu? (katmanlar, duraklar, varsayımlar)"):
        st.markdown(
            "- **Katman 1 — Karapınar bilim puanı (0–100):** S1, S2, S3 çevresinde yüksek ödül; arka planda TES tabaka silikat haritası ve "
            "topoğrafik nemlilik. **S1** iniş noktasına en yakın belirgin kraterin kenarı (maar / base surge analoğu), **S2** TES tabaka "
            "silikat / yüksek-Si cam haritasının en yüksek olduğu erişilebilir yer (zeolit–sülfat alterasyon analoğu), **S3** düz zeminde "
            "en büyük akış birikimi (evaporit kanalı analoğu).\n"
            "- **Katman 2 — Divlit tekerlek aşınması (0–100):** eğimi 6°'den düşük ama MOLA 0,6 km pürüzlülüğü yüksek volkanik zemin. "
            "**B1** Cerberus Fossae çatlağının tabanı.\n"
            "- **Katman 3 — eğim ve TRI:** 15° üzeri dikkat, 24° üzeri geçilemez.\n"
            "- **Rota A** yalnız mesafe ve eğime bakar (L0 → B1). **Rota B** duraklara uğrar ve aşınma cezasını maliyete ekler; sürgü "
            "değiştikçe A* ile yeniden hesaplanır.\n"
            "- Bu bölgede katalogda kayıtlı sulu mineral tespiti ya da vadi ağı yoktur: durakların yeri **analog modelinin önerisidir**, "
            "Mars'ta doğrulanmış bulgu değildir. 'a'a blokları metre ölçeğindedir ve yörünge verisinde doğrudan görünmez.")


# --------------------------------------------------------------------------------------
# BENZERLİK ORANLARI — "Gezegen Analogları Puanlama Metodolojisi ve Çok Boyutlu Karşılaştırma Analizi"
# (A.R.E.S. teknik dokümanı; metin ve puanlar proje ekibinin belgesinden aynen aktarılmıştır)
# --------------------------------------------------------------------------------------
def _mafi_figure(b64_lines, caption, key):
    """Şekli gösterir: üstüne gelince sağ üstte çıkan ⛶ simgesiyle tam ekran; ayrıca 'Büyüt' düğmesiyle büyük pencere."""
    import base64
    data = base64.b64decode("".join(b64_lines) if isinstance(b64_lines, (list, tuple)) else b64_lines)
    st.image(data, caption=caption)
    dlg = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
    if dlg is not None:
        @dlg(caption.split(":")[0], width="large")
        def _big():
            st.image(data)
            st.caption(caption)
        if st.button("🔍 Şekli büyüt", key=key):
            _big()
    st.caption("İpucu: şeklin üstüne gelince sağ üst köşede çıkan **⛶** simgesine tıklayarak tam ekran açabilirsiniz.")


def benzerlik_page():
    st.markdown("<small style='letter-spacing:.08em;opacity:.75'>A.R.E.S. TEKNİK DOKÜMANTASYON • "
                "KARŞILAŞTIRMALI GEZEGEN ANALOJİSİ RAPORU</small>", unsafe_allow_html=True)
    st.title("Gezegen Analogları Puanlama Metodolojisi ve Çok Boyutlu Karşılaştırma Analizi")
    st.markdown("*Meke Gölü – Karapınar Volkanik Alanı (Konya) ile Divlit Volkanik Parkı / Kula–Salihli Jeoparkı (Manisa) "
                "Sahalarının Nicel Değerlendirmesi*")
    st.divider()
    st.markdown(
        "Dünya üzerindeki hiçbir coğrafi bölge, Ay veya Mars ortamını fiziksel, kimyasal ve atmosferik açıdan %100 oranında "
        "taklit edemez. Farklı jeolojik sahalar; gezegen keşfinin farklı bilimsel ve mühendislik problemlerine çözüm üretir. "
        "Bu nedenle **A.R.E.S. (Analog-Driven Rover Exploration & Site-Selection)** projemizde, Anadolu’nun iki özgün volkanik "
        "laboratuvarı olan **Meke Gölü – Karapınar Volkanik Alanı (Konya)** ile **Divlit Volkanik Parkı / Kula–Salihli UNESCO "
        "Global Jeoparkı (Manisa)** sahalarını öznel yorumlarla değil, uluslararası gezegen bilimi literatüründeki resmi "
        "derecelendirme sistemlerini birleştiren nicel bir matematiksel modelle değerlendirdik.")

    # ---------------------------------------------------------------- 1
    st.header("1. Uluslararası Literatürde Kullanılan Resmi Analog Puanlama Sistemleri")
    st.markdown("Gezegen jeolojisi ve astrobiyoloji literatüründe karasal analog sahaların Mars veya Ay’a uygunluğunu ölçmek "
                "için geliştirilmiş beş temel akademik çerçeve bulunmaktadır:")
    st.subheader("1.1. Ryan vd. (2024, Geosphere) — 14 Puanlık Hidrovolkanik Mars Analog Skalası")
    st.markdown("Amerika Jeoloji Topluluğu (*Geological Society of America — GSA*) bünyesinde yayımlanan bu güncel çalışmada "
                "(**Ryan vd., 2024**), dünya genelindeki **45 hidrovolkanik alan**; magma–su etkileşimi mekanizmaları, piroklastik "
                "tabakalaşma, düşük sıcaklıklı sulu alterasyon mineralleri, morfolojik korunmuşluk ve biyo-imza koruma potansiyeli "
                "kriterleri üzerinden **14 puanlık** bir skalayla puanlanmıştır.")
    st.success("**Projemiz Açısından Önemi:** Çalışmamızın ana sahalarından biri olan Karapınar Volkanik Alanı, bu uluslararası "
               "veri tabanında doğrudan değerlendirilmiş ve **11 / 14 puan** alarak Mars araştırmaları için ileri düzeyde önerilen "
               "**11 seçkin küresel analog** arasına girmiştir.")
    st.subheader("1.2. NASA / CSA — Terrestrial Analog Fidelity (Sadakat Derecesi) Çerçevesi")
    st.markdown(
        "NASA ve Kanada Uzay Ajansı (CSA) tarafından yürütülen gezegen analog programlarında (**Snook & Mendell, 2004; "
        "Osinski vd., 2006; Léveillé, 2010**), bir sahanın gezegensel benzerliği tek bir düz puan yerine dört temel eksende "
        "**Yüksek (High)**, **Orta (Medium)** ve **Düşük (Low)** sadakat (*fidelity*) sınıflarıyla derecelendirilir:\n\n"
        "- **Jeomorfolojik ve Topografik Sadakat:** Yüzey şekillerinin, eğimlerin ve krater geometrisinin gezegen karşılığıyla uyumu.\n"
        "- **Jeokimyasal ve Mineralojik Sadakat:** Kayaç bileşiminin, izotopik imzaların ve ikincil alterasyon fazlarının örtüşmesi.\n"
        "- **Çevresel ve İklimsel Sadakat:** Sıcaklık, kuraklık, radyasyon, atmosfer basıncı ve yerçekimi koşulları.\n"
        "- **Operasyonel Sadakat:** Gezici (rover) hareketliliği, enstrüman kalibrasyonu, astronot eğitimi ve üs inşaatına elverişlilik.")
    st.subheader("1.3. İşlevsel Analog Sınıflandırma Sistemi (Preston & Dartnell, 2014; Foucher vd., 2021)")
    st.markdown(
        "Avrupa Uzay Ajansı (ESA) ve astrobiyoloji konsorsiyumlarının kullandığı bu sistem (**Preston & Dartnell, 2014; "
        "Foucher vd., 2021**), analog sahaları görev işlevine göre dört kategoride derecelendirir:\n\n"
        "1. **Süreç Analogları:** Volkanizma, sedimantasyon, rüzgar aşınımı veya göl kuruması gibi dinamik jeolojik "
        "mekanizmaları temsil eden sahalar.\n"
        "2. **Çevresel Aşırılık Analogları:** Dondurucu soğuk, aşırı kuraklık veya yüksek UV radyasyonu sunan sahalar.\n"
        "3. **Astrobiyolojik / Biyo-İmza Analogları:** Ekstremofil mikroorganizmaları barındıran ve mineral tuzaklarında "
        "(silika, karbonat, sülfat) yaşam izlerini koruyan sahalar.\n"
        "4. **Teknoloji ve Mühendislik Test Sahaları:** Gezici tekerlek dayanıklılığı, yeraltı radarı (GPR), LiDAR "
        "navigasyonu ve yerinde kaynak kullanımı (ISRU) testlerinin yapıldığı sahalar.")
    st.subheader("1.4. Gezegen ve Kayaç Benzerlik Endeksleri: ESI, MSI ve RSI")
    st.markdown("**Schulze-Makuch vd. (2011)** tarafından geliştirilen *Earth Similarity Index* (ESI) ve **Kaczmarek vd. (2020)** "
                "tarafından kayaç/regolit ölçeğine uyarlanan *Mars Similarity Index* (MSI) ile *Rock Similarity Index* (RSI) "
                "modelleri; her bir fiziksel/kimyasal parametreyi 0 ile 1 arasında ağırlıklı geometrik ortalama formülüyle "
                "normalize ederek sayısal bir yakınlık skoru üretir.")
    st.subheader("1.5. NASA Space Apps Challenge — 8 Resmi Değerlendirme Boyutu")
    st.markdown("NASA Space Apps *\"Permanent Moon Base Locations & Mars\"* görevi ise bir analog sahanın bütünsel olarak "
                "değerlendirilebilmesi için şu **8 zorunlu analiz boyutunu** tanımlar: (1) Jeoloji ve Mineraloji, (2) Arazi ve "
                "Topografya, (3) Sıcaklık ve Çevresel Aşırılıklar, (4) Su, Buz ve Potansiyel Kaynaklar, (5) Toz, Rüzgar ve Yüzey "
                "Süreçleri, (6) Biyoloji ve Biyo-İmzalar, (7) Mağaralar ve Yeraltı Ortamları, (8) İniş, Hareketlilik, Robotik ve "
                "İnşaat Koşulları.")

    # ---------------------------------------------------------------- 2
    st.header("2. Geliştirdiğimiz Yeni Hibrit Puanlama Formülü: M.A.F.I. (Mars Analog Fidelity Index)")
    st.subheader("Neden Yeni Bir Hibrit Formüle İhtiyaç Duyduk?")
    st.markdown(
        "Mevcut tekil sistemlerin her birinin önemli bir sınırı vardır:\n\n"
        "- **Ryan vd. (2024)** yalnızca ıslak/patlamalı (hidrovolkanik) sahaları puanlar; kuru lav akıntılarını ve astronot "
        "sığınağı olabilecek lav tüplerini kapsamaz.\n"
        "- **NASA Fidelity** çerçevesi nitel (High/Medium/Low) kaldığı için iki saha arasında hassas sayısal sıralama yapmayı "
        "zorlaştırır.\n"
        "- **MSI / RSI** endeksleri ise kayaç kimyasına odaklanır; rover tekerlek aşınması, otonom navigasyon ve kalıcı üs "
        "inşaatı gibi mühendislik koşullarını ölçmez.")
    st.markdown("Bu boşluğu doldurmak için yukarıdaki tüm resmi sistemleri birleştiren **M.A.F.I. (Mars Analog Fidelity Index)** "
                "modelini geliştirdik. Modelde NASA’nın 8 resmi analoji boyutunun her biri 0.0 ile 10.0 arasında puanlanır "
                "(S<sub>i</sub> ∈ [0, 10]). Hiçbir sahanın zayıf yönlerini (örneğin Dünya atmosferi ve 1g yerçekimi farkını) "
                "gizlememek için tüm boyutlara eşit ağırlık (w<sub>i</sub> = 1) verilmiş ve toplam skor 80 puan üzerinden "
                "yüzdeye dönüştürülmüştür:", unsafe_allow_html=True)
    st.latex(r"\text{M.A.F.I.}\;(\%) \;=\; \frac{\sum_{i=1}^{8} S_i}{80} \times 100")
    st.markdown("Ayrıca, iki farklı sahanın birbirini nasıl tamamladığını matematiksel olarak kanıtlamak için 8 boyutu iki "
                "**Görev Odaklı Uzmanlık Alt Endeksine** ayırdık:")
    st.markdown("1. **Astrobiyoloji ve Sulu Süreçler Alt Endeksi (ASI):** Jeoloji/Mineraloji (S<sub>1</sub>), Su/Kaynaklar "
                "(S<sub>4</sub>), Yüzey Süreçleri (S<sub>5</sub>) ve Biyoloji/Biyo-İmza (S<sub>6</sub>) boyutlarının ortalamasıdır:",
                unsafe_allow_html=True)
    st.latex(r"\text{ASI}\;(\%) \;=\; \frac{S_1 + S_4 + S_5 + S_6}{40} \times 100")
    st.markdown("2. **Mühendislik, Mobilite ve Yeraltı Üs Alt Endeksi (EMI):** Arazi/Topografya (S<sub>2</sub>), Mağara/Lav "
                "Tüpleri (S<sub>7</sub>) ve Robotik/İnşaat (S<sub>8</sub>) boyutlarının ortalamasıdır:", unsafe_allow_html=True)
    st.latex(r"\text{EMI}\;(\%) \;=\; \frac{S_2 + S_7 + S_8}{30} \times 100")
    st.subheader("0.0 – 10.0 Puanlama Cetveli (Bilimsel Rubrik)")
    st.markdown(
        "| Puan aralığı | Sınıf | Tanım |\n|---|---|---|\n"
        "| **9.0 – 10.0** | Birebir Süreç ve Yapısal Eşleşme | Hem morfoloji/mineraloji hem de oluşum mekanizması Mars veya "
        "Ay’daki hedef bölgeyle doğrudan örtüşür ve hakemli literatürle doğrulanmıştır. |\n"
        "| **7.5 – 8.9** | Güçlü Bilimsel Uyum | Temel jeolojik veya operasyonel süreç ortaktır; yalnızca gezegensel ölçek ya da "
        "ikincil mineral oranı farklılık gösterir. |\n"
        "| **5.5 – 7.4** | Kısmi / Tek Yönlü Uyum | Sadece belirli bir alt süreci temsil eder; diğer mekanizmalar eksik veya zayıftır. |\n"
        "| **3.0 – 5.4** | Fiziksel Sınır / Düşük Sadakat | Dünya’nın 1g yerçekimi, yoğun N₂–O₂ atmosferi, manyetosfer koruması "
        "veya bitki örtüsü nedeniyle gezegensel koşullardan belirgin sapma vardır. |\n"
        "| **0.0 – 2.9** | Uyumsuz | Hedef gezegen ortamıyla ters çalışan koşullar. |")

    # ---------------------------------------------------------------- 3
    st.header("3. Karapınar ve Divlit (Kula) Sahalarının Karşılaştırmalı Puan Tablosu")
    st.markdown(
        "| # | NASA Analoji Boyutu | Meke & Karapınar (Konya) | Divlit / Kula (Manisa) | Üstün Olan Saha ve Görev Rolü |\n"
        "|:-:|---|:-:|:-:|---|\n"
        "| 1 | Jeoloji, Mineraloji ve Yüzey Malzemeleri | 9.0 / 10 | 8.5 / 10 | Karapınar (OIB + Kümülat + Jezero uyumu) |\n"
        "| 2 | Arazi, Yükseklik, Eğim, Pürüzlülük ve Kraterler | 9.5 / 10 | 9.5 / 10 | Eşit (Maar konisi vs. Skorya + ’A’a lavı) |\n"
        "| 3 | Sıcaklık, Aydınlatma ve Çevresel Aşırılıklar | 3.5 / 10 | 3.5 / 10 | Eşit (Dünya atmosferi & 1g sınırı) |\n"
        "| 4 | Su, Buz, Permafrost ve Potansiyel Kaynaklar | 8.0 / 10 | 6.0 / 10 | Karapınar (Freatomagmatik su & alterasyon) |\n"
        "| 5 | Toz, Rüzgar, Erozyon ve Yüzey Süreçleri | 9.5 / 10 | 8.0 / 10 | Karapınar (Base surge & göl evaporitleri) |\n"
        "| 6 | Biyoloji, Ekstremofiller ve Biyolojik İmzalar | 8.5 / 10 | 6.5 / 10 | Karapınar (Silika/karbonat biyo-imza tuzakları) |\n"
        "| 7 | Mağaralar, Lav Tüpleri ve Yeraltı Ortamları | 7.5 / 10 | 9.5 / 10 | Divlit (3D taranmış gerçek lav tünelleri) |\n"
        "| 8 | İniş, Hareketlilik, Navigasyon, Robotik ve İnşaat | 9.0 / 10 | 9.5 / 10 | Divlit (’A’a tekerlek testi & yeraltı üssü) |\n"
        "| — | **TOPLAM HAM PUAN (80 Üzerinden)** | **64.5 / 80** | **61.0 / 80** | **Karapınar +3.5 Puan Önde** |\n"
        "| — | **GENEL M.A.F.I. BENZERLİK ORANI** | **%80.6** | **%76.3** | **Her İkisi de \"High Fidelity\" Sınıfında** |\n"
        "| — | **Astrobiyoloji ve Sulu Süreçler Endeksi (ASI)** | **%87.5** (8.75/10) | **%72.5** (7.25/10) | **Karapınar** (+%15.0 Farkla Bilim Lideri) |\n"
        "| — | **Mühendislik, Mobilite ve Üs Endeksi (EMI)** | **%86.7** (8.67/10) | **%95.0** (9.50/10) | **Divlit** (+%8.3 Farkla Mühendislik Lideri) |")

    # ---------------------------------------------------------------- 4
    st.header("4. Neye Kaç Puan Verdik? — 8 Boyutta Ayrıntılı Bilimsel Gerekçelendirme")

    def dim(title, score, why, why_not):
        with st.container(border=True):
            st.markdown(f"**{title} — {score}**")
            st.markdown(f"*Neden {score.split(' /')[0]} Verildi?* {why}")
            if why_not != "—":
                st.markdown(f"*Neden 10 Değil?* {why_not}")

    st.subheader("4.1. Meke Gölü ve Karapınar Volkanik Alanı (Konya) — 64.5 / 80 (%80.6)")
    st.info("**Uluslararası Veri Tabanı Konumu:** Ryan vd. (2024, Geosphere) — 11 / 14 Puan • **Birincil Mars Harita Eşleşmesi:** "
            "MC-13 Syrtis Major (Jezero Krateri) ve MC-14 Amenthes (Nepenthes/C27 hidrovolkanik alanı)")
    dim("1. Jeoloji, Mineraloji ve Yüzey Malzemeleri", "9.0 / 10",
        "Karapınar volkanitleri, levha tektoniği olmayan Mars manto plümü ergimesiyle örtüşen astenosferik OIB-benzeri bir manto "
        "kaynağından (δ¹⁸O: 5.7–6.5‰; **Gençoğlu Korkmaz vd., 2022**) türemiştir. Ana mineral üçlüsü olan olivin (Fo₇₀–₉₀), "
        "klinopiroksen ve plajiyoklaz (An₄₀–₈₀) ile bazaltlar içindeki dünit/klinopiroksenit kümülat anklavları, NASA Perseverance "
        "gezicisinin Jezero Krateri tabanında incelediği Séítah olivin kümülatı ve Máaz bazaltlarıyla aynı magmatik ayrışım "
        "mekanizmasını yansıtır. Ayrıca kabuk asimilasyonuyla kazanılan kuvars ve zirkon ksenokristalleri, Mars şergottit "
        "meteoritlerindeki kabuk kirlenmesi süreçlerini modellemek için eşsiz bir laboratuvardır.",
        "Mars bazaltları Dünya bazaltlarına kıyasla demirce daha zengindir (yüksek Fe/Mg oranı) ve göktaşı çarpmalarına bağlı "
        "yüksek basınç fazı olan maskelinit içerir.")
    dim("2. Arazi, Yükseklik, Eğim, Pürüzlülük ve Kraterler", "9.5 / 10",
        "Meke, Acıgöl ve Yılan maarları; dik iç krater duvarları, dairesel tüf halkaları ve Meke Maarı’nın merkezinde yükselen iç "
        "cüruf konisiyle (~1.4 km ölçek) Mars’ın MC-14 Amenthes bölgesinde **Brož ve Hauber (2013)** tarafından haritalanan 170’ten "
        "fazla freatomagmatik tüf halkası ve maar yapısının yüzeydeki en yakın morfolojik ikizidir.",
        "Mars’taki düşük yerçekimi (0.38g) nedeniyle Mars maar ve tüf halkalarının balistik püskürük saçılım yarıçapı ve yamaç "
        "eğim açıları Dünya’dakilerden biraz daha geniştir.")
    dim("3. Sıcaklık, Aydınlatma ve Çevresel Aşırılıklar", "3.5 / 10",
        "Bu düşük puan bilimsel dürüstlüğümüzün gereğidir. Karapınar orta enlem karasal iklime, 1g yerçekimine, yoğun N₂–O₂ "
        "atmosferine ve manyetosferik radyasyon kalkanına sahiptir. Mars’ın ortalama −60°C sıcaklığı, 6 mbar ince CO₂ atmosferi "
        "ve yüksek kozmik radyasyonu burada doğrudan simüle edilemez.",
        "—")
    dim("4. Su, Buz, Permafrost ve Potansiyel Kaynaklar", "8.0 / 10",
        "Yükselen bazaltik magmanın yeraltı akiferiyle patlamalı etkileşimi ve ardından Meke volkanitlerinin gaz boşluklarında "
        "(veziküllerde) çökelen ikincil kuvars (SiO₂), zeolit, kalsit ve sülfat dolguları (**Ryan vd., 2024; Güllü & Kadıoğlu, "
        "2019**), Jezero Krateri tabanındaki kayaç gözeneklerinde tespit edilen karbonat ve sülfat fazlarıyla aynı düşük "
        "sıcaklıklı su–kaya etkileşimini kanıtlar. Ayrıca hidratlı zeolit ve sülfatlar yerinde kaynak kullanımı (ISRU) için su "
        "geri kazanım modelidir.",
        "Mars’ta magma ile etkileşen su kaynağı genellikle donmuş yeraltı buzu (kriyosfer / permafrost) iken, Karapınar’da sıvı "
        "yeraltı suyu ve göl sistemi rol oynamıştır.")
    dim("5. Toz, Rüzgar, Erozyon ve Yüzey Süreçleri", "9.5 / 10",
        "Freatomagmatik patlamalarla oluşan düşük açılı çapraz tabakalı taban yayılımı (*base surge*) katmanları, dış suyun "
        "tükenmesiyle krater içi konileşmeye geçiş, mevsimsel kuruyan kapalı havza gölündeki evaporitik sülfat/karbonat tuz "
        "kabukları ve piroklastik katmanlardaki rüzgar erozyonu; Mars’taki Gale ve Jezero kraterlerinin antik göl kuruma ve "
        "eoliyen aşınım süreçlerini neredeyse birebir yeniden üretir.",
        "Mars atmosferi çok daha ince olduğu için rüzgarla taşınan toz taneciklerinin saltasyon eşiği Dünya’dan farklıdır.")
    dim("6. Biyoloji, Ekstremofiller ve Biyolojik İmzalar", "8.5 / 10",
        "Meke ve Acıgöl’ün tuzcul-alkali su kimyasında yaşayan halofilik/ekstremofil mikroorganizmalar ve düşük sıcaklıklı "
        "hidrotermal/göl ortamında çökelen amorf silika, karbonat ve sülfat mineral tuzakları, antik Mars göllerinde olası "
        "mikrobiyal yaşam izlerinin (biyo-imzaların) milyarlarca yıl boyunca nasıl korunabileceğini test etmek için birinci "
        "sınıf bir astrobiyoloji laboratuvarıdır.",
        "Modern bitki örtüsü ve atmosferik oksijen nedeniyle organik madde bozunma hızı steril Mars yüzeyinden yüksektir.")
    dim("7. Mağaralar, Lav Tüpleri ve Yeraltı Ortamları", "7.5 / 10",
        "Bölgede fay kontrollü magma beslenme kanalları, hidrovolkanik diyatrem/boru yapıları ve derin alt-orta kabuktan "
        "taşınmış kümülat anklavları yeraltı magmatik mimarisini mükemmel yansıtır.",
        "Karapınar patlamalı (freatomagmatik) bir alan olduğu için, astronotların içine girip radyasyondan korunabileceği geniş, "
        "boş lav tüpü mağaraları bulunmaz.")
    dim("8. İniş, Hareketlilik, Navigasyon, Robotik ve İnşaat", "9.0 / 10",
        "Yörünge görüntülerinde (HiRISE/CTX) birbirine karışan freatomagmatik maarlar ile meteor çarpma kraterlerini ayırt etme "
        "algoritmalarının eğitimi, piroklastik yamaçlarda rover eğim/tutunma testleri ve Raman (SHERLOC) – LIBS (SuperCam) "
        "spektrometrelerinin bazalt içindeki sülfat/karbonat fazlarını tanıma kalibrasyonu için ideal bir saha test merkezidir.",
        "Keskin bloklu ’a’a lavları ve yeraltı tüneli içi GPS’siz otonom navigasyon testleri açısından Divlit’in gerisindedir.")
    _mafi_figure(MAFI_FIG_KARAPINAR,
                 "Şekil 1. Karapınar Volkanik Alanı M.A.F.I. Modeli Doğrulaması: Sol panelde eşit ağırlıklı M.A.F.I. ana skorumuzun "
                 "(%80.6; 64.5/80), Ryan vd. (2024) resmi veri tabanı puanı (%78.6; 11/14), Schulze-Makuch vd. (2011) geometrik "
                 "ortalama testi (%77.5) ve Kaczmarek vd. (2020) ampirik RSI formülü (%93.2) ile tutarlılığı; sağ panelde ise 8 eksenli "
                 "rubrik puanları ile ampirik veri kalibrasyonunun örtüşmesi (Δ = 1.3 puan) gösterilmektedir.", "fig1")

    st.subheader("4.2. Divlit Volkanik Parkı / Kula–Salihli Jeoparkı (Manisa) — 61.0 / 80 (%76.3)")
    st.info("**Uluslararası Veri Tabanı Konumu:** UNESCO Global Jeoparkı — Holosen (<11–4.7 ka) Genç Bazaltik Sistem • "
            "**Birincil Mars ve Ay Harita Eşleşmesi:** Mars MC-09 Tharsis (Ulysses Colles / Pavonis Mons), MC-23 Aeolis "
            "(Gusev Krateri), MC-15 Elysium (Athabasca Valles) ve Ay Marius Hills")
    dim("1. Jeoloji, Mineraloji ve Yüzey Malzemeleri", "8.5 / 10",
        "Divlit Tepe’nin III. evre lavları silise doygun olmayan alkali bazanit ve fonotefrit bileşimindedir. Porfirik dokulu "
        "kayaçlarda fenokristal olarak zonlu ve özşekilli klinopiroksen, kısmen serpantinleşmiş/iddingsitleşmiş olivin ve "
        "kuvvetli pleokroik hornblend; hamurda ise plajiyoklaz mikrolitleri ile feldispatoid grubu mineraller (nefelin ve lösit) "
        "yer alır. Bu kimya, NASA Spirit gezicisinin Gusev Krateri’nde (Columbia Hills — Irvine, Backstay, Wishstone) ve Curiosity "
        "gezicisinin Gale Krateri’nde (Jake_M) tanımladığı Mars alkali kayaçlarıyla doğrudan örtüşür.",
        "Mars kabuğunun büyük bölümü toleyitik bazaltlardan oluşur; Divlit’teki sulu hornblend ve feldispatoid mineralleri Mars’ın "
        "yalnızca lokalize alkali bölgelerinde görülür, Ay’daki susuz mare bazaltlarında ise bulunmaz.")
    dim("2. Arazi, Yükseklik, Eğim, Pürüzlülük ve Kraterler", "9.5 / 10",
        "Deniz seviyesinden 860 m yükseklikte (170 m nispi yükseklik) yükselen Divlit cüruf/skorya konisi, parazit koniler, "
        "sıçratma (*spatter*) konileri, lav hendekleri ve 20,73 km² alana yayılan, yüzeyinde birkaç santimetreden 1 metreye kadar "
        "keskin girinti ve çıkıntılar barındıran bloklu ’a’a lavları tek bir kompakt sahada toplanmıştır. Bu yapı, Mars MC-09 "
        "Tharsis skorya konilerinin ve MC-15 Elysium genç pürüzlü lav akıntılarının birebir saha karşılığıdır.",
        "Düşük yerçekimi (0.38g) nedeniyle Mars’taki skorya konileri Dünya’dakilere oranla ortalama 2.6 kat daha geniş taban "
        "çapına ve daha yatık yamaç eğimine sahiptir (**Brož & Hauber, 2014**).")
    dim("3. Sıcaklık, Aydınlatma ve Çevresel Aşırılıklar", "3.5 / 10",
        "Karapınar ile aynı fiziksel sınırlara tabidir: Ege–karasal geçiş iklimi, 1g yerçekimi, kalın N₂–O₂ atmosferi ve yerel "
        "bitki örtüsü nedeniyle Mars’ın dondurucu ince atmosferi veya Ay’ın ultra-yüksek vakum/termal şok koşulları burada yeniden "
        "üretilemez.",
        "—")
    dim("4. Su, Buz, Permafrost ve Potansiyel Kaynaklar", "6.0 / 10",
        "Divlit sistemi temel olarak kuru (stromboliyen ve efüzif) bir volkanizmadır. Mars’ın geç Amazoniyen dönemindeki kuru "
        "bazaltik volkanizmasını ve hidratlı amfibol/volkanik cam fazlarını temsil etse de, Jezero Krateri gibi antik göl veya "
        "hidrotermal su–kaya alterasyon ortamlarını temsil etmez. Bu nedenle su/hidroloji boyutunda Karapınar’ın (8.0) gerisinde kalır.",
        "—")
    dim("5. Toz, Rüzgar, Erozyon ve Yüzey Süreçleri", "8.0 / 10",
        "Holosen yaşlı (<11–4.7 ka; **Ulusoy vd., 2019**) olması sayesinde lav akıntılarının alt (camsı breş), orta (sütunsal soğuma "
        "yapılı masif bazalt) ve üst (bloklu ve köpüksü) soğuma zonları erozyondan minimum etkilenmiş \"taze\" bir yüzey sunar. "
        "Ayrıca Kula lavlarının Gediz Nehri vadisini tıkayarak lav seti gölleri oluşturması, Mars’taki lav akıntısı–akarsu kanalı "
        "blokajlarıyla paraleldir.",
        "Karapınar’daki gibi çok katmanlı base surge piroklastik tortulları ve kuruyan tuz gölü evaporit kabukları Divlit’te sınırlıdır.")
    dim("6. Biyoloji, Ekstremofiller ve Biyolojik İmzalar", "6.5 / 10",
        "Lav tünelleri karanlık ve radyasyondan korunaklı kemolitotrofik mikro-habitatlar için önemli bir yeraltı modeli sunsa da, "
        "yüzeydeki çok genç ve kuru ’a’a lavları fosil biyo-imza tuzaklama açısından Karapınar’ın sulu silika/karbonat ortamlarının "
        "gerisindedir.",
        "—")
    dim("7. Mağaralar, Lav Tüpleri ve Yeraltı Ortamları", "9.5 / 10",
        "Divlit’in en güçlü ayırt edici özelliğidir. Lav akıntısının üst kabuğu katılaşıp içerideki akışkan bazaltın boşalmasıyla "
        "oluşan ve 3D lazer taraması yapılmış gerçek lav tünelleri (*lava tubes*) ve lav hendekleri; Ay’daki (Marius Hills) ve "
        "Mars’taki (Pavonis ve Arsia Mons çökme çukurları / *skylight*) radyasyon ve mikrometeorit korumalı doğal yeraltı "
        "sığınaklarının doğrudan saha karşılığıdır.",
        "Ay ve Mars’ın düşük yerçekimi (0.16g ve 0.38g) nedeniyle gezegensel lav tüpleri onlarca metre genişliğe ulaşabilirken "
        "Divlit tünelleri daha küçük ölçeklidir.")
    dim("8. İniş, Hareketlilik, Navigasyon, Robotik ve İnşaat", "9.5 / 10",
        "NASA Curiosity gezicisinin alüminyum tekerleklerinde ciddi hasara yol açan keskin bloklu bazaltik arazilerin birebir "
        "örneği olan 1 metre pürüzlü ’a’a lavları, yeni nesil rover tekerlek dayanıklılığı ve otonom tehlike kaçınma (*hazard "
        "avoidance*) testleri için benzersiz bir mühendislik parkurudur. Aynı zamanda GPS ve uydu iletişiminin kesildiği lav tüneli "
        "içlerinde LiDAR/SLAM otonom haritalama, yeraltı radarı (GPR) ile tavan kalınlığı ölçümü ve gözenekli skorya/bazalt "
        "regolitinden 3D yazdırma/sinterleme yoluyla kalıcı üs inşaatı (ISRU) testlerine imkân tanır.",
        "Ay’ın elektrostatik yüklü keskin regolit tozu ve vakum ortamı saha koşullarında bulunmamaktadır.")
    _mafi_figure(MAFI_FIG_DIVLIT,
                 "Şekil 2. Divlit Volkanik Parkı (Kula) M.A.F.I. Modeli Doğrulaması: Sol panelde M.A.F.I. Mühendislik ve Üs Alt Endeksi "
                 "(%95.0), Kaczmarek vd. (2020) kayaç/morfoloji RSI skoru (%94.5), ana M.A.F.I. skorumuz (%76.3; 61.0/80) ve "
                 "Schulze-Makuch vd. (2011) geometrik ortalama kontrolü (%72.9); sağ panelde ise 8 boyutlu rubrik puanları ile saha "
                 "ölçüm kalibrasyonunun uyumu (Δ = 0.7 puan) sunulmaktadır.", "fig2")

    # ---------------------------------------------------------------- 5
    st.header("5. Sentez: Neden İki Sahayı Tek Rotada Birleştiriyoruz?")
    st.markdown("M.A.F.I. puanlama sonuçları, projenin temel bilimsel tezini matematiksel olarak kanıtlamaktadır: **Karapınar "
                "(%80.6 genel skor; %87.5 Astrobiyoloji Endeksi)** Mars’ın ıslak, hidrovolkanik ve yaşam izi barındırma potansiyeli "
                "yüksek geçmişini çözerken; **Divlit / Kula (%76.3 genel skor; %95.0 Mühendislik ve Üs Endeksi)** gezici araçların "
                "yüzeydeki tekerlek aşınma risklerini ve astronotların lav tüpü içi kalıcı üs güvenliğini modellemektedir. "
                "**A.R.E.S.** simülasyonumuzda bu iki Anadolu laboratuvarının katsayıları **Elysium Planitia – Cerberus Fossae** "
                "yörünge verilerine entegre edilerek, gezici aracın keskin ’a’a lav bloklarından kaçınıp sulu alterasyon "
                "duraklarından geçerek güvenli lav tüpü üssüne ulaşması sağlanmıştır.")

    # ---------------------------------------------------------------- 6
    st.header("6. Kaynakça")
    st.markdown(
        "1. Brož, P., & Hauber, E. (2013). Hydrovolcanic tuff rings and cones as indicators for phreatomagmatic explosive eruptions on Mars. *Journal of Geophysical Research: Planets*, 118(8), 1656–1675.\n"
        "2. Brož, P., & Hauber, E. (2014). A unique volcanic field in Tharsis, Mars: Pyroclastic cones as evidence for explosive eruptions. *Icarus*, 232, 244–253.\n"
        "3. Foucher, F., Hickman-Lewis, K., Hutzler, A., Joy, K. H., Folco, L., Bridges, J. C., ... & Westall, F. (2021). Definition and use of functional analogues in planetary exploration. *Planetary and Space Science*, 197, 105162.\n"
        "4. Gençoğlu Korkmaz, G., Kurt, H., Asan, K., & Petrelli, M. (2022). Mineral chemistry, whole-rock geochemistry, and O-isotope constraints on the petrogenesis of the Karapınar Volcanic Field (Central Anatolia, Turkey). *International Geology Review*, 64(18), 2557–2582.\n"
        "5. Güllü, B., & Kadıoğlu, Y. K. (2019). Petrography, mineral chemistry, and crystallization conditions of Meke and Acıgöl volcanic rocks (Karapınar, Konya). *Bulletin of the Mineral Research and Exploration (MTA)*, 159, 139–162.\n"
        "6. Kaczmarek, A., et al. (2020). Quantitative similarity indices for evaluating terrestrial analog rocks and regolith simulants for Mars exploration. *Planetary and Space Science*, 182, 104833.\n"
        "7. Léveillé, R. (2010). A half-century of terrestrial analog studies: From craters on the Moon to searching for life on Mars. *Planetary and Space Science*, 58(4), 631–638.\n"
        "8. Osinski, G. R., Léveillé, R., Berinstain, A., Lebeuf, M., & Bamsey, M. (2006). Terrestrial analogues to Mars and the Moon: Canada’s role. *Geoscience Canada*, 33(4), 175–188.\n"
        "9. Preston, L. J., & Dartnell, L. R. (2014). Planetary habitability: Lessons learned from terrestrial analogues. *International Journal of Astrobiology*, 13(1), 81–98.\n"
        "10. Ryan, C. H., et al. (2024). Global evaluation of terrestrial hydrovolcanic fields as analogs for Martian phreatomagmatic deposits and biosignature preservation. *Geosphere* (Geological Society of America).\n"
        "11. Schulze-Makuch, D., Méndez, A., Fairén, A. G., von Paris, P., Turse, C., Boyer, G., Davila, A. F., Resendes de Sousa António, M., Catling, D., & Irwin, L. N. (2011). A two-tiered approach to assessing the habitability of exoplanets. *Astrobiology*, 11(10), 1041–1052.\n"
        "12. Snook, K. J., & Mendell, W. W. (2004). The NASA Terrestrial Analog Program: Optimizing planetary science and exploration preparation. *Lunar and Planetary Science Conference (LPSC)*.\n"
        "13. Ulusoy, İ., Sarıkaya, M. A., Çubukçu, H. E., & Şen, E. (2019). Holocene volcanic activity at Kula Volcanic Geopark (Western Turkey) constrained by cosmogenic ³⁶Cl and ³He dating. *Quaternary Geochronology*, 54, 101016.")
    st.caption("A.R.E.S. • Gezegen Analogları Puanlama Metodolojisi (M.A.F.I.)")


# --------------------------------------------------------------------------------------
# ANA SAYFA
# --------------------------------------------------------------------------------------
THEME_TITLES = {
    "🪨 Jeoloji ve mineraloji": "Jeoloji, mineraloji ve yüzey malzemeleri",
    "⛰️ Arazi ve topoğrafya": "Arazi, yükseklik, eğim, pürüzlülük, kraterler ve diğer topoğrafik özellikler",
    "🌡️ Sıcaklık ve aydınlanma": "Sıcaklık, aydınlatma ve çevresel aşırılıklar",
    "💧 Su, buz ve kaynaklar": "Su, buz, permafrost ve diğer potansiyel kaynaklar",
    "🌬️ Toz, rüzgâr ve erozyon": "Toz, rüzgâr, erozyon ve diğer atmosferik veya yüzey süreçleri",
    "🧬 Biyoloji ve yaşanabilirlik": "Biyoloji, ekstremofiller, yaşanabilirlik ve biyolojik imzalar",
    "🕳️ Mağaralar ve lav tüpleri": "Mağaralar, lav tüpleri, yeraltı ortamları ve diğer yeraltı özellikleri",
    "🚀 İniş, hareketlilik ve inşaat": "İniş, hareketlilik, navigasyon, robotik operasyonlar, insan keşfi ve inşaat koşulları",
}
MARS = "Mars"
COMPARE = "↔ Karşılaştırma"
SIMILARITY = "↔ Benzerlik Oranları"
SIM = "2D Simulation"            # ilk 2D simülasyon: kodda duruyor, menüde gösterilmiyor
SIM_LABEL = ":red-background[**🎮 2D SIMULATION**]"
SIM3 = "3D Rota Oluşturma Simülasyonu"   # A.R.E.S. rota simülasyonu (sim3d_page)
SIM3_LABEL = ":red-background[**🎮 3D ROTA OLUŞTURMA SİMÜLASYONU**]"
GRID_N = 450                     # ızgara çözünürlüğü (uzun kenar, piksel) — sabit
MEKE = "Meke Gölü ve Karapınar – Konya"
KULA = "Kula Volkanik Jeoparkı – Manisa"
PLACE_LABEL = {MEKE: "Meke Gölü ve Karapınar (Konya)", KULA: "Kula Volkanik Jeoparkı (Manisa)"}
PLACES = [MEKE, KULA, MARS]      # menüdeki mekânlar; diğer üç Dünya bölgesi REGIONS içinde duruyor ama listelenmiyor


def mars_page(region, n):
    set_body("mars")
    reg = MARS_REGIONS[region]
    bbox = tuple(reg["bbox"])
    W, H, px = grid_shape(bbox, n)
    st.title(f"Mars — {region}")
    st.markdown(f"**{reg['tip']}** — {reg['ozet']}")
    st.caption(f"Sınır kutusu: {bbox[1]:.1f}–{bbox[3]:.1f}° enlem, {bbox[0]:.1f}–{bbox[2]:.1f}° doğu boylamı · "
               f"{W * px / 1000:.0f} × {H * px / 1000:.0f} km · piksel ≈ {px:.0f} m · "
               "bölge özeti genel literatür bilgisidir, veriden türetilmemiştir.")
    try:
        with st.spinner("Mars yükseklik modeli indiriliyor…"):
            dem, px = get_mars_dem(bbox, n)
    except Exception as e:
        st.error(f"Mars yükseklik verisi alınamadı (trek.nasa.gov). İnternet bağlantınızı kontrol edin.\n\n{e}")
        st.stop()
    funcs = [mars_geology, mars_topo, mars_thermal, mars_water, mars_processes, mars_biology, mars_subsurface, mars_operations]
    themes = dict(zip(THEME_TITLES, funcs))
    theme = st.radio("İnceleme başlığı", list(themes), horizontal=True, label_visibility="collapsed")
    st.header(THEME_TITLES[theme])
    themes[theme](bbox, n, dem, px)


def main():
    st.set_page_config(page_title="Volkanik Alan ve Mars Gözlem Paneli", page_icon="🌋", layout="wide")
    with st.sidebar:
        st.title("🌋 Volkanik Alanlar ve Mars")
        name = st.radio("Mekân seçin", PLACES + [COMPARE, SIMILARITY, SIM3], index=0,
                        format_func=lambda o: {SIM3: SIM3_LABEL}.get(o, PLACE_LABEL.get(o, o)))
        place_a = place_b = None
        if name == COMPARE:
            place_a = st.selectbox("1. mekân", PLACES, index=None, placeholder="Mekân seçin",
                                   format_func=lambda o: PLACE_LABEL.get(o, o))
            place_b = st.selectbox("2. mekân", [p for p in PLACES if p != place_a], index=None, placeholder="Mekân seçin",
                                   format_func=lambda o: PLACE_LABEL.get(o, o),
                                   help="İlk kutuda seçilen mekân burada listelenmez; aynı mekân kendisiyle karşılaştırılamaz.")
        mars_region = None
        if name == MARS or MARS in (place_a, place_b):
            mars_region = st.selectbox("Mars bölgesi", list(MARS_REGIONS), index=0)
        st.divider()
        this_year = date.today().year
        year = st.selectbox("Uydu verisi yılı (Dünya)", list(range(this_year - 1, 2018, -1)), index=0,
                            help="Sentinel-2 ve Landsat için yaz sahnesi, MODIS için tüm yıl kullanılır. Mars verisini etkilemez.")
        n = GRID_N
        st.divider()
        st.caption("**Dünya verileri — kaynakça**  \n"
                   "[Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/): Copernicus DEM GLO-30, "
                   "Sentinel-2 L2A, Landsat C2 L2, MODIS, ASTER L1T, ESA WorldCover, JRC Global Surface Water.")
        st.caption("**Mars verileri — kaynakça**  \n"
                   "1. [NASA Mars Trek](https://trek.nasa.gov/mars) — MOLA + HRSC yükseklik modeli, Viking mozaiği, "
                   "jeoloji haritası, vadi ağı, kumul ve buzul katalogları.  \n"
                   "2. [Mars Trek harita servisleri (API)](https://trek.nasa.gov/tiles/apidoc/trekAPI.html?body=mars) — "
                   "verilerin çekildiği WMTS karo servisi.  \n"
                   "3. [ASU Mars Space Flight Facility](https://www.mars.asu.edu/data/) — TES (albedo, ısıl atalet, toz, "
                   "mineraller) ve THEMIS gündüz/gece kızılötesi ürünleri; Mars Trek üzerinden alındı.  \n"
                   "4. [ESA — Mars'ın küresel sulu mineral haritası](https://www.esa.int/media/global-map-of-hydrated-minerals-on-mars) — "
                   "referans görsel; sayısal karşılığı OMEGA + CRISM tespitleri (Carter vd., 2013).")
        st.caption("Her başlık ilk açılışta 10–40 sn sürebilir; sonuçlar önbelleğe alınır.")

    if name == SIMILARITY:
        benzerlik_page()
        return
    if name == SIM3:
        sim3d_page()
        return
    if name == COMPARE:
        if place_a and place_b:
            compare_view(place_a, place_b, mars_region, n, year)
        else:
            st.title("Karşılaştırma")
            st.info("Sol menüdeki iki kutudan **iki farklı mekân** seçin. Meke–Karapınar, Kula ve Mars arasından istediğiniz "
                    "ikiliyi karşılaştırabilirsiniz; Mars seçildiğinde hangi Mars bölgesinin kullanılacağını da seçersiniz.")
        return
    if name == MARS:
        mars_page(mars_region, n)
        return

    set_body("earth")
    reg = REGIONS[name]
    bbox = tuple(reg["bbox"])
    W, H, px = grid_shape(bbox, n)
    st.title(PLACE_LABEL.get(name, name))
    st.markdown(f"**{reg['tip']}** — {reg['ozet']}")
    st.caption(f"Sınır kutusu: {bbox[1]:.2f}–{bbox[3]:.2f}°K, {bbox[0]:.2f}–{bbox[2]:.2f}°D · "
               f"{W * px / 1000:.0f} × {H * px / 1000:.0f} km · piksel ≈ {px:.0f} m · "
               "bölge özeti genel literatür bilgisidir, uydu verisinden türetilmemiştir.")

    try:
        with st.spinner("Yükseklik modeli indiriliyor…"):
            dem, px = get_dem(bbox, n)
    except Exception as e:
        st.error(f"Yükseklik verisi alınamadı, internet bağlantınızı kontrol edin.\n\n{e}")
        st.stop()
    wc = ls = None
    try:
        with st.spinner("Yüzey örtüsü indiriliyor…"):
            wc = get_worldcover(bbox, n)
    except Exception as e:
        st.warning(f"WorldCover alınamadı: {e}")
    try:
        with st.spinner("Landsat yüzey sıcaklığı indiriliyor…"):
            ls = get_landsat_lst(bbox, n, year)
    except Exception as e:
        st.warning(f"Landsat yüzey sıcaklığı alınamadı: {e}")

    lst = ls["lst"] if ls else None
    themes = {
        "🪨 Jeoloji ve mineraloji": ("Jeoloji, mineraloji ve yüzey malzemeleri",
                                     lambda: tab_geology(bbox, n, year, wc)),
        "⛰️ Arazi ve topoğrafya": ("Arazi, yükseklik, eğim, pürüzlülük, kraterler ve diğer topoğrafik özellikler",
                                   lambda: tab_topo(bbox, dem, px, wc, lst)),
        "🌡️ Sıcaklık ve aydınlanma": ("Sıcaklık, aydınlatma ve çevresel aşırılıklar",
                                      lambda: tab_thermal(bbox, dem, px, year, wc, ls)),
        "💧 Su, buz ve kaynaklar": ("Su, buz, permafrost ve diğer potansiyel kaynaklar",
                                    lambda: (tab_hydro(bbox, n, year, dem, px, wc),
                                             frost_and_resources(bbox, n, year, dem, px, wc))),
        "🌬️ Toz, rüzgâr ve erozyon": ("Toz, rüzgâr, erozyon ve diğer atmosferik veya yüzey süreçleri",
                                      lambda: tab_processes(bbox, n, year, dem, px, wc)),
        "🧬 Biyoloji ve yaşanabilirlik": ("Biyoloji, ekstremofiller, yaşanabilirlik ve biyolojik imzalar",
                                          lambda: tab_biology(bbox, n, year, dem, px, wc, lst)),
        "🕳️ Mağaralar ve lav tüpleri": ("Mağaralar, lav tüpleri, yeraltı ortamları ve diğer yeraltı özellikleri",
                                        lambda: tab_subsurface(bbox, n, year, dem, px, wc, lst)),
        "🚀 İniş, hareketlilik ve inşaat": ("İniş, hareketlilik, navigasyon, robotik operasyonlar, insan keşfi ve inşaat koşulları",
                                            lambda: tab_operations(bbox, n, year, dem, px, wc)),
    }
    theme = st.radio("İnceleme başlığı", list(themes), horizontal=True, label_visibility="collapsed")
    st.header(themes[theme][0])
    themes[theme][1]()



# Şekil 1 ve Şekil 2 (yüksek çözünürlüklü, JPEG, base64)
MAFI_FIG_KARAPINAR = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/"
    "2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCALLBkADAREAAhEBAxEB/8QA"
    "HgABAAEFAQEBAQAAAAAAAAAAAAcDBAUGCAIBCQr/xAB6EAABAwMCAwMEBxAMCAsDABsCAAMEAQUGBxIIERMJFCIhMTJCFRkjUldi0hYXM0FRU1hhcpGSlZaX"
    "09QYGiQ4VVZxdneClLQlQ4GisrO21TQ1NzlUY3N1obXCCjZEdPCTo8HiJkdIZGWDw0WEsdHyRmZnaHmkpcTF/8QAHQEBAAMAAwEBAQAAAAAAAAAAAAECAwQF"
    "BgcICf/EAEgRAQABAwEEBQcKBAUEAgIDAQABAgMREgQFITEGE1FSkQcUFyIyQVMVFiMzVGFxgaHRNKKx8Bg1QpLBJDZDcgjhRPElVWKC/9oADAMBAAIRAxEA"
    "PwD9/EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBB2b8XkrD+PTBOCsMCbkM5ngl2yMsjK5VAolYT7DfQ6HT8e/q89+8du30arSmj1Jl"
    "WqrEvPGVxfSeE696R2qLgTd8+ehq1bsLcccuVY3seMpiQ73mnuZ9Wo9Dl0/Du3elTkot2pqkrqxCX77luL4tRj5pMnt9u7290onf5rbPWc94G+tNxfaoq4qn"
    "3J1UruZNi2+K5OnyG2WWgqbzzrlBAKU89a1r5qKNMytwULDkdiym2N3vGb3DuMJ7n0JkCUDzZ/yGFajVRMTHvQpW3L8VvV1lWK0ZPbpU2BXlOhxpwG9H+7AS"
    "3B/lTTn3o1QvJ06Fb4xzZ8xqO03Tm4885QQGn2618ymmJjBFSHuP/ivkcEXB/mfFPBwpvJjxONFdCyuXLugyutNYjcutRtzZy6270K+jyVrVEziFKq0mHn2M"
    "WzFYWVZffINmjzWWTFy4TgaChmNCoG8+VK1VMV5XzDNNGw+3R5oxICpzEhrzpWiYqNULCBkmN3S4S7Ta8hhSZdvMQnxWJQG5GKvmo4I15hWv21eKakdZD7j+"
    "T43lMVyZi+QQrkyy6TTzsCYDwg5T0grUK15FT6irVqhbOXwcvxM8hrh4ZPbiu4s9QrWM4O8CHvunz38vt8lOmVdSnl8vJ4GKXOdhNkjXG8swHnLVb5s2sdmT"
    "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"
    "4/Ms+tiDRLIMcMlWtNLHp3XNefsLqDTKKzK276N/hh25d32dXw/ROlv519Hdt9VU18WmiXrNOGksxxzUTH65n3b5v77DuXWpbt/cehFgMbOXUp1d3ct27wcu"
    "ry5V28ynrDRLRg4E7rXVSNm56hWALdA1Gey6JRnBmqXZ556Q68caRPJ+pG0PWIAqIAQiLdK7hDaV+u9XCnVSxM3s+MsyQr980Os9ohjdrV3DljGEDbgml36P"
    "L75PZCT0pUkehsEwBrb1XC8u7ZSev4J6p1KuM3EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQFXSCaQTSCaQU4yCYwCnIU+2mcAojkCTyHnp/bUh0/toHT+2ozAdP7aZgekiMBS"
    "vPypMZHwa1rTnVSiZw+1rypzVdJE5FM8UiRwBRpBNI+c+VPFVTmAqXOldtUzA+qNIVpSvkqpiMIzh8Edv01JM4fa8/pUUZgjiKUiD5sH6iIzD5yDnyUZhOXp"
    "SCAgICrpBNKM4eahWtedS/8ABWS9IPg1rWnOqieKJnAVa0pzokcCJy+qUirMZBWRnLwQ1GqJe1EAkxlXUKNKwmkeen9tTmA6f20zA9KQQfNg/URGYNg/UQzD"
    "50/tok6f20HqteVOarpRE5KV505pgmcCskQFGMgmMAkxkFGkFOMAmcDz0/tqQ6f20Dp/bQOn9tB6VJgFaIwCkFWZnIKwIFa8qc1XThETkpXnTnyTGSZwKYjB"
    "nIpSICAq6cD5WnOtK81YfUViYyKuFhWBRjI89P7akOn9tB92D9REZg2j9RDMPqJEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE"
    "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE"
    "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEHLHFvxBcZ3DgOQa+Ugadt6cY5kVpgW/FpwynL1ksaS7HYeeZlA6LUWR1"
    "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"
    "Q308VLa6eOSODU8E7OnjHd1H06yHUvA5lyzHEtWbbkGomrVy1jmyYmXQmZbjpdzs3U6MWgjVoqNG2FGulsaE928dar9E0zCvVpWm8CmuxdnNjXDZDsVrDKrX"
    "rbFyiXHG4hRkYQZwV5M6Oeap90ru5elu8PpLKb0a8nVpL4NsRvGZ8S+uXF7dLZOgQcvvNuxbFYlwhmw47bbED7BzdhiJUF6ZJmbN1PE002dPIayr5L0cnSao"
    "tPIRIgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgxdrxnG"
    "rBOuNysWOQoUm7TKS7tIhxAbOa/RoGqOvENObp9NpoNxcy2tgPmGiDKICAgICAgICDEvYfij2YM56/i1tO9x7e5Aj3o4IVltRTMTNgXuW+jZGAFUOe2tRpX6"
    "SDLICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgKs8ZHylaVrXkpnkPqkEBV1AmoFYKVpXy0UTGR5q59SimAodOflookelGoE1Dz1PtKwdT7SD0gICrqBNQKwKJjIKQQeep9pA6n"
    "2kH2gU+n5VXSPqmZwCjUPPU+0rD6Na1pzqozCJjIVOdOVFJEYfUSc+dPJVVmUYgTUkTUCagTUCROR8rs586+dTMZAa86eWqlExkKvKnnQiMPqrqSJqHwgpWv"
    "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"
    "l9Wg+qPhHwqLnCFqeqnm5z4idFtSeHfVGdxy8H+KvXK5yG2/nuaY2/pNBnFuZEv3XHEtoheY4fQneY95AO7u/wCJNrPHBeJ4p70Q1t0z4jNLbNrNpDlLF4sF"
    "9iUfhSma8qjXnyNpwPSaeA6VA2j5G2YEJUoQ1oqzGV24KQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE"
    "BAQEBAQEBAQEBBofFJ+9k1E/mHd/7k6piMyrXVoomrsfkLwk/wDtAubaTcKumml0fhmtU1rG8AtFrbmuZM6FXxjw2mhOo9Atu7Zu27l29jd010xL5Fvfyj3N"
    "g2mq3FjlMxz7Eh/tkfPPsVLR+VTv6st692YcGvyqXJo+o/U/bI+efYp2j8qnf1ZR8mK+lO58D9T9sj559inaPyqd/Vk+TD0p3PgfqftkfPPsU7R+VTv6snyY"
    "elO58D9T9sj559inaPyqd/Vk+TD0p3PgfqftkfPPsU7R+VTv6snyYelO58D9T9sj559inaPyqd/Vk+TD0p3PgfqftkfPPsU7R+VTv6snyYelO58D9T9sj559"
    "inaPyqd/Vk+TD0p3PgfqftkfPPsU7R+VTv6snyYiryq3fgfqr2//ANoo1AyOR7Gt8KMCpdM3f3LlB1P3MSP0SYHw+DxfF5q0bslrb8qN2r/wfqoftkfPPsU7"
    "R+VTv6sq/JjOfKncz9R+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WT5MP"
    "Snc+B+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+x"
    "TtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WU/JkkeVW7P/g/VXb/APaKM+lW97Iv2KMHlDcBoibyY+l7ru9OvQ5iXg8Ph+qnyZK0eVC7MZ6j9VD9sj559ina"
    "Pyqd/VlHyYr6U7nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyfJh6U7"
    "nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyfJh6U7nwP1P2yPnn2Kdo"
    "/Kp39WT5MPSnc+B+p+2R88+xTtH5VO/qyn5Mk9Kdz4H6qr3/ALRVqBBis3ouFGByn7tvUyh2jXgrt8Feh4vjJ8mVLR5UruPqP1Uv2yPnn2Kdo/Kp39WUfJiv"
    "pTufA/U/bI+efYp2j8qnf1ZPkw9Kdz4H6n7ZHzz7FO0flU7+rJ8mHpTufA/U/bI+efYp2j8qnf1ZPkw9Kdz4H6n7ZHzz7FO0flU7+rJ8mHpTufA/U/bI+efY"
    "p2j8qnf1ZPkw9Kdz4H6n7ZHzz7FO0flU7+rJ8mHpTufA/U/bI+efYp2j8qnf1ZPkw9Kdz4H6n7ZHzz7FO0flU7+rJG6vWqRPlUu00U/Qfq3LRvtycw18n3C2"
    "zOHe222lqh0kiTeRG71SIxDb4mB2+kuh35sE0xS9b0V8ole211RNj9TSHtGuJjH8gy6VrFbcfyC33DInJGGxYO6MdptpEWyK7UWvdzEeXjL3q6a5ss66fVw9"
    "dHS3RTVToZPWDtlMs4fcdh5NF0Dg3A7jMKOTMi/G1sHZ1d24Wi3ef8Fc7c+wTVtNXrPO9JunHydstNcWf1R1+2R88+xTtH5VO/qy9T8mPD+lO58D9T9sj559"
    "inaPyqd/VlHyYelO58D9T9sj559inaPyqd/VlandetFXlTuXLcT1GPzdP9mT2nmQdoBlOWY/etIoeNUxu3xJDTkW8HJ6/WMx21Emw27dn+cuBtOyzs8vZdE+"
    "lMdIK6qaqcYdgLhvdCAgICAgo3Iepb5A9cWtzJe6F6vh9JKBWHzU/kSsEBAQQTxP/vrOGT+ky9f7JXtUnmvHJ0UoSICAgICAgICAgICAgICAgICAgICAgICC"
    "KOLnigs/Cvpi3kzeLz8lym/XALNgOGWod0vIbw6BkzFD3gUEDdeer4WmGnXS8gKYjKJnDV+D3hbveike96xa2ZBEyfWDP3GpWoGWR45A14N3QtkOheJq3xBO"
    "oMh6ReN09xmRK6kc0yzbtFszVJcvfXcQgDbY8zcMvMI09aqiKJguYrvxjksLrp7p3kFtvNpvmF26ZGyJ4XL7HkwwMJ5iDbVKuiXp1oDTY+L3lPqKfWpZ127H"
    "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"
    "bNfuTYsxqz2uO+1o4zMy0DsNg0l0hvXcMlyEjkzJwtgZx4AiTW2lCEtpGda7S9IekW30vDrZootbnvbRf9iOEfnweP3pvbeU772bd+7ZzcrnM/hHH+jojsk+"
    "GgOG3hjgjlENtnLMuAL1ew/xwMuU/c7Va+fwNecfVMzXF6H7nt7s2KbtUYqqnM/8PYdPekW0b62ymjVqiiIpj3cf9X6x+jqflTnzXrddVc5p44eGzOnRTVzF"
    "KBAQEGOcBy13aj0WG1SNLIu+ObtpC5tHadd3q15bfutqurSyJCJDtId1CVFnMPC7V7g04rb9wSzosaHgGd1nZhoo6D20Ir5O9W9WEA9UWnXaTY4D/iZDwj4I"
    "/hzaOskBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBh5OG4rMzGJn8zHYrl7t9ufgwbq41Qno8Z42jeaAvVEzZZIqU9LpBz9GiDMI"
    "CAgICAgICAgICAgICAgICAgICAgIND4pP3smon8w7v8A3J1TT7UM7v1VX4S/l40iZd+dRjPJuvL5n4Xq/wDUgvV7PiKI9Z+WOkdq/c39cqppnGZ9zYOi/wDW"
    "y/Bqt6tM/wCp0k7NtU1zOmfCTov/AFsvwaqPV7x5vtXdnwk6L/1svwaqfV7x5vtXdnwk6L/1svwap6vePN9q7s+EnRf+tl+DVPV7x5vtXdnwk6L/ANbL8Gqe"
    "r3jzfau7PhJ0X/rZfg1T1e8eb7V3Z8JOi/8AWy/Bqnq94832ruz4SdF/62X4NU9XvHm+1d2fCTov/Wy/BqpzT3lvN9p0+zPhLPac2+5TsmrHt844bnsbMPrC"
    "zz8IxXSqO0vfCNR/rKmY7zfZrG0xPsz4SwPRf+tl+DVT6vecedn2rPsz4SdF/wCtl+DVT6veR5vtXdnwk6L/ANbL8Gqer3jzfau7PhJ0X/rZfg1T1e8eb7V3"
    "Z8JOi/8AWy/Bqnq94832ruz4SdF/62X4NU9XvHm+1d2fCTov/Wy/Bqnq94832ruz4SdF/wCtl+DVPV7x5vtXdnwk6L/1svwap6vePN9q7s+EnRf+tl+DVPV7"
    "x5vtXdnwk6L/ANbL8GqnMd5M7JtXdnwlsFptt0c05vU5meYR27lCB6H0fC7Uhf2lu847eRfhKM095yrWz7TFONM+Etf6L/1svwap6vecWdn2rPsz4HRf+tl+"
    "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"
    "cFBISHaP2iVqReiQuALjZcxr5RJKieMYfljxa4MXCn20GG6sQy7vZNTnGzcc9EBkvCUJ8K/1yadL/tfirr987PRf3f1tOIqo/wCXnuj+07dunpjVs0Z6vaf0"
    "0w7nXnrlNVFcV0c55vsl2KJpxM5FCggp+JuR4W/A75SL4y3mdUONVHrZVFhVEuVM6qcCIEBCeIo14KaFvK29ShM7e8bS6YkXpU+MuVYjrPb9lxb/ANH7HN7Y"
    "2+k4NKPVEep75L8dXPqeyWPpPb5qq4+vLmaRQqICAgIKTPT709t3c/DuWl36uhlZ+trVVWfrJaU+xAqpEBAQE4x7JVmrlyW5dQXhGCIFQnP3R8X/AOmXJtzE"
    "+04d2IifouarF7uLA9227PVVLvrNbVWOb2uPEYciqNUCszjmxuYYrZ86xK5YXkUWj0C7wXokxkvWacAgqP4JLG5TVVExTwy2pmimYqqjMRxx+DlrsTNTLlpX"
    "q/qhwJ5nPKkuw3RyfaG3vJR2rR0ZfIKe9KnQdEfe1IvfLldGt0bZuaNN+vVFeZp/5dN0o6Xbq6WbZXGz2urrsYice/P9/q/RyVLjwYrkyS5QG2hI3CL1RovV"
    "xFdUxng8pVVs9yNHJb2dmY3HN+dK6pyHquD6XIAL0RHd8Xb/AFualPJdoCAgILK/vPDb6xYcwGJMkujHIi2+Wv1PjCPMv6q0VqXrYi2NGx8w+FZrCAgICtSL"
    "OC5IkXSXI7wBsBsaZFsue0h3b93vS3V2/wBVKhonGBrM9w78LOoOtkMROZjeJzpltZ9Z+ZRohjNCPrEbxNgI+sRUFUnkmOapwOcPJ8J3CBpvw5S7n36biGHw"
    "YF2uG8i77PFqhS5G4vLXqSCdc/rqi6V0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEGh8Un72T"
    "UT+Yd3/uTqCPOzfbbLs9NCiIKVr853GfLWn/ANzI6t1du5xnHgU3btEacz4pp6TX1sfwVEWbXPh4Ii9emec+L702/rY/eTqbP3eCdd/tk6bf1sfvJ1Nn7vA1"
    "3+2Tpt/Wx+8nU2fu8DXf7ZOm39bH7ydTZ+7wNd/tk6bf1sfvJ1Nn7vA13+2Tpt/Wx+8nU2fu8DXf7ZOm39bH7ydTZ+7wNd/tk6bf1sfvJ1Nn7vA13+2TphXz"
    "hT7yr1NqJxw8E03ase1Pit7nuCHuaii4XWa8NQ5+uPlV+qtUx7vBSblVXKqfGVx02/rY/eUdTZ+7wTrv96fE6bf1sfvJ1Nn7vBOu/wBsnTb+tj95Ops/d4Gu"
    "/wBsnTb+tj95Ops/d4Gu/wBsnTb+tj95Ops/d4Gu/wBsnTb+tj95Ops/d4Gu/wBsnTb+tj95Ops/d4Gu/wBsnTb+tj95Ops/d4Gu/wBsnTb+tj95Ops/d4Gu"
    "/wBsnTb+tj95Oqs/d4Gu/wBsnTDly2U+8ops2tc8vBHW1zdxmfFavDULkw2EQaiTbm49nlH0fDz/APl9FWptWotzPDwRN2uq77U8PvlddNv62P3lHU2vu8Ft"
    "d/vT4nTb+tj95Ops/d4Gu/2ydNv62P3k6mz93ga7/bJ02/rY/eTqbP3eBrv9snTb+tj95Ops/d4Gu/2ydNv62P3k6mz93ga7/bJ02/rY/eTqbP3eBrv9snTb"
    "+tj95Ops/d4Gu/2ydNv62P3k6mz93ga7/bJ02/rY/eTqrP3eBrv55ydMPeU+8oiza9aeHgrVeq6yY1T4reLSpT5LZRRoI7Nh7PKfh/zlPVWqop5eC1Vyqm9O"
    "Kp8Vx02/rY/eTqbX3eBrv96fE6bf1sfvJ1Nn7vA13+2Tpt/Wx+8nU2fu8DXf7ZOm39bH7ydTZ+7wNd/tk6bf1sfvJ1Nn7vA13+2Tpt/Wx+8nU2fu8DXf7ZOm"
    "39bH7ydTZ+7wNd/tk6bf1sfvJ1Vn7vA17R2y+dJv6bY/eVppsTw0x4FNe0UxEzMsNmAADMbaFKe7V81Pirpd526aI4RHg7LYLtVyaszPD70AcLGX6T5RnOqs"
    "DTfhpn4BMtmeSY2SXadi7VvHKpwme+5NGHlmAddxdcvEW5ddtFr/AKanNX6Ow2e9E1ziJ8U6YKVSnSK1boHuZeiO31yXN3X1dO0Vfk4G8oru00etyy2Zd+6g"
    "QEBAQEBAQEBBQuRENveJpmh1ozXa2Q7hLw+jtQVx81P5EBAQEEE8T/76zhk/pMvX+yV7VJ5rxydFKEiAgICAgICAgICAgICAgICAgIIY7QDQZ/iU4PM80qtb"
    "ZezZ2b2TxN4SrQo97gOBOtr1OXl8ExiOVaU9IaEPrIM1wx65Y3xMcO+E8QGIufuDMMZh3RlsvTYq80JGyfvTA94EPqkBCtGbaJ/doF6jXR0nOb/7kIR27PLu"
    "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"
    "EBAQEBAQEBAQaHxSfvZNRP5h3f8AuTqmOaJ5P5r9C/8A2hrtM9KNE8P0vwvUbG2bPjeLwLXa2XsRiuGEaPHBpoSMh3EW0KeVdza2Omp1035y2r9ss9qz8J2L"
    "/kXE+StqthphSdpk/bLPas/Cdi/5FxPkqfMbHYt18n7ZZ7Vn4TsX/IuJ8lPMdnOvk/bLPas/Cdi/5FxPkp5js518n7ZZ7Vn4TsX/ACLifJTzHZzr5P2yz2rP"
    "wnYv+RcT5KeY7OdfJ+2We1Z+E7F/yLifJTzHZzr5P2yz2rPwnYv+RcT5KeY7OdfJ+2We1Z+E7F/yLifJTzGwdfJ+2We1Z+E7F/yLifJVPMaTr5ZjBf8A2lDt"
    "LpmQE1mmqmMtwe4y67m8LjbusMcyZ9EOf0UQTzCk6+WH/bLPas/Cdi/5FxPkq/mNjsOvk/bLPas/Cdi/5FxPkp5js518n7ZZ7Vn4TsX/ACLifJTzHZzr5P2y"
    "z2rPwnYv+RcT5KeY7OdfJ+2We1Z+E7F/yLifJTzHZzr5P2yz2rPwnYv+RcT5KeY7OdfJ+2We1Z+E7F/yLifJTzHZzr5P2yz2rPwnYv8AkXE+SnmOznXyftln"
    "tWfhOxf8i4nyU8x2c6+T9ss9qz8J2L/kXE+SnmNg6+T9ss9qz8J2L/kXE+SqeY0s/OJZm2/+0n9pY5gt1m3DVbGhvTc6JS2tjhcbaTNaP9bn4NvpUZ8/vvD9"
    "NPMaU+cSw37ZZ7Vn4TsX/IuJ8lX8xsdi/XyftlntWfhOxf8AIuJ8lPMdnOvk/bLPas/Cdi/5FxPkp5js518n7ZZ7Vn4TsX/IuJ8lPMdnOvk/bLPas/Cdi/5F"
    "xPkp5js518n7ZZ7Vn4TsX/IuJ8lPMdnOvk/bLPas/Cdi/wCRcT5KeY7OdfJ+2We1Z+E7F/yLifJTzHZzr5P2yz2rPwnYv+RcT5KeY7OdfJ+2We1Z+E7F/wAi"
    "4nyU8xsHXyftlntWfhOxf8i4nyVHmVJ5xLLXj/2k/tL28Vs0i06qY0V0c7x7LNlhcXkO0/cuW4OXoe9TzKk6+WJ/bLPas/Cdi/5FxPkqfMbHYdfJ+2We1Z+E"
    "7F/yLifJTzHZzr5P2yz2rPwnYv8AkXE+SnmOznXyftlntWfhOxf8i4nyU8x2c6+T9ss9qz8J2L/kXE+SnmOznXyftlntWfhOxf8AIuJ8lPMdnOvk/bLPas/C"
    "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"
    "11Qnq6VNv/hDvue3zeL3y1u1VdXQwtU09ZWqLOqqrrZaRTT1cCiYqlaJpgVdFWV4qowJVmFaZiZEppyVCkEBAQ5iiZxXphNEVUVzTPNouYcTGgeBXa9Y/lmr"
    "ligXCwQQl3aHInU6sVoy2hup6W4iIdoD4vFTw+IU22ad324uVcIcTdm0W997xq2Pd85vR7ofmLdrHxjdqXrrIsMG7UvNog3Ym7fVgXYdmtrRkQg7Xf4gIgHd"
    "tLe8Q8/S27VxY6TXNtq6jc9GqqeE18sflLr6/JjXb2yd4dMNoxRTxptR/q/OH6U8CPZT6L8KkKFmmU2wMgzNkicG7TAp04fOg8qMt+iFaeXxcyLxF4vVHsd1"
    "dGtm2K/51tUdZen3z+zm746V13d3+Ybvjqdno4RTH7us/JSnKlF6iqr1tNEcXjKI11xVXOZj39rEag3N2zaeZBeYzc83IlklPNjbA5ySqDNSp0RL0nfJ4ft7"
    "VblOipx9puzXYuVW+blrgo1RxfAtRbjY73qJitY98xLHiiDjmSHKtr14kOyg91N8qlS4SS2+QfE6DG4txCuVct08Hlt0bTNibkT7Uw+aAZBb3p+kF6tGVOva"
    "u3q8PN6vW5u4k7IqDcOV38JrG6tGWo8voA0VRHZXpAHhPaVblGIlbd1Wa7Vc56zM55uvi9Ivulx3r45PiDHypB3aQVttdw2d2kB34hHyiPpbKV9HcXk3fFJX"
    "Zr8RFsRbaERpTyCIqjRynxF3CRx964TOAXBprbunOMkw/wAQ96ZI+UpsxF1jFGjDb7rJAhdl7S8EXweEpA7aTOV4jDrW02u3WG1xrLZoDUWHEYBiJFjt0Btl"
    "oBoIAA08gjSlKUpRVmcLQvFKBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBofFJ+9k1E/"
    "mHd/7k6pjmieT+S7TXs+uPbK9ObBlGL8EertyttyssWTb7hb9Nbq9HlMm0JA60YMFQwIaiQkJbSEl39naLdMcZdZVRXMs37W32iH2BWtH5rrv+rrararXaxm"
    "1Xk9rb7RD7ArWj8113/V086t9poqPa2+0Q+wK1o/Ndd/1dPOrfaaKj2tvtEPsCtaPzXXf9XTzq32mio9rb7RD7ArWj8113/V086t9poqPa2+0Q+wK1o/Ndd/"
    "1dPOrfaaKj2tvtEPsCtaPzXXf9XTzq32mio9rb7RD7ArWj8113/V086t9poqPa2+0Q+wK1o/Ndd/1dPOrfaaKj2tvtEPsCtaPzXXf9XTzq12mipn9Muz27Ru"
    "wZQVyicCmrDJ+xc5rqXTTe7Ms8nIroEO7oemQ12gPrGQj6yr5zbzzNFTAe1t9oh9gVrR+a67/q6t51b7TRUe1t9oh9gVrR+a67/q6edW+00VHtbfaIfYFa0f"
    "muu/6unnVvtNFR7W32iH2BWtH5rrv+rp51b7TRUe1t9oh9gVrR+a67/q6edW+00VHtbfaIfYFa0fmuu/6unnVvtNFR7W32iH2BWtH5rrv+rp51b7TRUe1t9o"
    "h9gVrR+a67/q6edW+00VHtbfaIfYFa0fmuu/6unnVvtNFR7W32iH2BWtH5rrv+rp51b7TRUe1t9oh9gVrR+a67/q6jzm12mipsNn7PjtG4umV7x9vgU1Xo1M"
    "ukB1xl7Te7DLOrVJG0mg6HjAd9d5eruD3yecWu00VNe9rb7RD7ArWj8113/V1PnVvtNFR7W32iH2BWtH5rrv+rp51b7TRUe1t9oh9gVrR+a67/q6edW+00VH"
    "tbfaIfYFa0fmuu/6unnVvtNFR7W32iH2BWtH5rrv+rp51b7TRUe1t9oh9gVrR+a67/q6edW+00VHtbfaIfYFa0fmuu/6unnVvtNFR7W32iH2BWtH5rrv+rp5"
    "1b7TRUe1t9oh9gVrR+a67/q6edW+00VHtbfaIfYFa0fmuu/6unnVvtNFR7W32iH2BWtH5rrv+rqPObXaaKmdyDs9u0bm4Rj1nc4FNWDCEMvptx9Nbsb4b3d1"
    "esHQ8G71ffCnnFrtNFTBe1t9oh9gVrR+a67/AKup86t9poqPa2+0Q+wK1o/Ndd/1dPOrfaaKj2tvtEPsCtaPzXXf9XTzq32mio9rb7RD7ArWj8113/V086t9"
    "poqPa2+0Q+wK1o/Ndd/1dPOrfaaKj2tvtEPsCtaPzXXf9XTzq32mio9rb7RD7ArWj8113/V086t9poqPa2+0Q+wK1o/Ndd/1dPOrfaaKj2tvtEPsCtaPzXXf"
    "9WTzi1qnimq3cw6E7PngZ42MByjLZWdcHuqdmal2EGorl20/uUYXz7wBbAI2B3Ft8W0V53fl6iqKcS7rddFcP1u7H3TF7TTRHJm7pqpe7xc5+XS3Lpi96mby"
    "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"
    "BflSWvSYHaGncJWosSx8cd/w6brRZc/vmS5tmIzLbAyu8UuuIwY8s3Y7Uy1Sn+gzHBru8UHQYaoRGybRug/uQdN/PM4h/ns/Mh+xlH5le/dL5sPm0i/8H2/R"
    "u6bOp5/U3bkHLPH+5qNoXrfe9bMV1inUGZjci4XiDbu8lMsOPMwgiySHfKCCw0Uj3Vl14A2S3avGfd47+0Ic4juNO1au5vjeouB5VIsEKz4Fb38dteQaiVZl"
    "SLkzksiBJPHawn3WMjnmEUmjEnCaIH4I7v3U6KCV8V1PwVvWuzvs62THeIH5/Fyt2XYfKzCR3ljGAmyi2OWwndjVuCz92lNOi2LROk07vJ10t4ZLVjjjnnxm"
    "WO+Ydq3hN6xViPjtvwnTWPkDoXfN4+QEwTmQwgaIgnxWY5l0vAbQjDnkRB4TAJs4ObZBxfhqybTzG4YR8cxLMMosOIx2focW2Q7hIYZit09VqPsKKA+qEcBV"
    "7U5lWuGP4hf3yfC1/SFdv9kLyl2IVjg6UVGggICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKMwCZgFIICAgICAgICA"
    "gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIND4pP3smon8w7v/cnUEfdm"
    "7/zeehX9DuM/+WR1ozTSgICAgICAgICAoiBRuXeO6/uUNx9QPV5+HcPP/NV4iRWVQQEBAQEBAQEBAU4kW73evZJnpj7j0z6hbPW8O3/6KYkXCgEBRV9wx2WR"
    "5EzGZ8WK3U3DinQRH0qlyWtqYglbBmmEORWaTXSqYNjQqOW92tR8Po+isruyRX7lqNproffmw06+1+Lnf0axjd9Pda+eVdpTL9O6fU/Fzv6NI2CI/wBKJ2yr"
    "tVmdQcKZbFtmYYDT0RGC78hbU7Lp9zKb8yoHlmnT9zbvb7YHMabJtmYVrOrrYV840PZzGlfqKdCesUL/AHvSbLIzUbKrVFubbTm9pu4WYnhA/fDQ2y2kmg6x"
    "VfyfTSW7EkTI7LxwC3wXHLUZFHrt28w8Hg8nh8KaDrGu4TYtKcIznMtQ4t+u026Zvco8m6PXJt1wWWmIjUdmIwItD0o4CBns8Xusl4yKtXE0HWM6zdtJmLbW"
    "zMWqIESpBXuo2Yha3BUahXZs5eHlTl73amg6wdvOkz5MnItMM6xxOkfqWYi6VDrzLb7n4edfFVNB1jxd7npDf7nHvN/ssKbMicu6ypljJxxnlXnTYZBzHy+L"
    "wpoOsVnsk0wdu1L49DjnN5B+7CtR1c8G7b4+nz8O4uX8tU0HWPD120nkiIP2qGdBecdESspV2unQqGfoecudeZetuTQdYrNZTpszs6LDYVakddvbajptc27d"
    "9Pc/S21rTn5+RJoOsW/sppH7Dnj3sRD7g451HIXsLXome7dzqHT2lXd5f5U0HWL5nOcFjmb0d/YbvLqENvdoR8vDTn4PFyTQdYt5GSaZzLc5aZUdlyK49V12"
    "K5aTJpw6nvqdR2balv8AFu994lHUXKp5lVcQpTcn0xI2Hno7FDj9Tu5Faz5tdT0+Xg8O71vfLO/aminjK9u7c9zFYzmGndvekNx8biWoIz1QhuR4PLqhtEd4"
    "7A8HkpQdvxaLCqiq20iZ7F9GvWkhe50tsGtBHkI+w5V2jtcGn+L9665T7kz98S2tWqrrKu7NPuXrWT6Zx+RR4rIbZHXHbaTHa5s2b/Q9LZ4d3vfCttCvWKDF"
    "00jiX53Ko9niN3R4eT1ybsxUkOU5cvE7s3F5KU+mmg6x6j3vSiI7V+JbIrRk91SdbsxCRObt2/6H6W7y7vqpoOse5WQaVzho3Lt8Z4Qld5EXLOZbXvrviD0/"
    "jekmg6x7t2V6c2ho41paajNHt6jUe1mA15AIU50EPpAAj9yNB+kmg6xaYnN0iwfE4OB4jbWYFmtsEIcG2x7a7RliOA7RaEdnoDTyck0HWLuHk2mlvNt23xmW"
    "DZijHZqzajAm2aeIWh2h4Q+15k0HWAZVpyFvO00aapFdccN2N7Fn0zMzIzrUdm0iqdSKvvq1qSaDrHprLtO2Lg9eGBAJUkRF+S3bDFx2g+iJV6e4uX0k0HWL"
    "r54uI/wg7/YX/kpoOsUZOe4RJZcafkkdHGyFwXLe7Wlae9r4PRTQdYtqZBpbthB3KPttv/Fo+xB/uXw7fc/B4PD4fCmg6xV+azTn2Tre+mHfCZ6JTPYxzqk3"
    "z57N+znt5+qmg6xTrk2mdX48sI7QuQ2asxXBtTnNhutOVQCuzwj5KeEfepoOsYWwsaU6e6WtaVaWWdu22y3Wk4dptcOE6INDyqNKU5j4irWvOpFXcRFUi5kp"
    "oomJUrr1I54imXI/ExwtsvDtMNRLvQh+38yN7VLiaIdIqrQQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEHyg0pXnSiieI+1pSvno"
    "kRgFFQJSCsCAonkCRyBSCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgINC4qCIeGPUarYUIvmDvG0aly5/uJ1BytwAZ32h0TgS0YjYnwqaWzrW3pTjwW6bN1nmx3pEeluY2GbQ2M6NGQ8iIKGe2vh3F6SnMoxCXfm+7S"
    "/wCw90k/PnP/ANwJmTEHzfdpf9h7pJ+fOf8A7gTMmIPm+7S/7D3ST8+c/wD3AmZMQfN92l/2Hukn585/+4EzJiD5vu0v+w90k/PnP/3AmZMQfN92l/2Hukn5"
    "85/+4EzJiD5vu0v+w90k/PnP/wBwJmTEHzfdpf8AYe6Sfnzn/wC4EzJiD5vu0v8AsPdJPz5z/wDcCZkxB833aX/Ye6Sfnzn/AO4FGtGloHEtxa8fPDlpeOpO"
    "f8HGmjtvLJrJZ6Da9bJbj3eLjdYtvjltOxBTZ1pLW8t24Q3EIkXhK8VGlv8A833aX/Ye6Sfnzn/7gVcynEHzfdpf9h7pJ+fOf/uBMyYg+b7tL/sPdJPz5z/9"
    "wJmTEHzfdpf9h7pJ+fOf/uBMyYg+b7tL/sPdJPz5z/8AcCZkxB833aX/AGHukn585/8AuBMyYg+b7tL/ALD3ST8+c/8A3AmZMQfN92l/2Hukn585/wDuBMyY"
    "g+b7tL/sPdJPz5z/APcCZkxB833aX/Ye6Sfnzn/7gTMmIPm+7S/7D3ST8+c//cCnUjSj/LuLrj4xPiTwnhzunBxpqV/zLGb5eLO4xrZLKKLFtOAD4unWxCQn"
    "zns7BECEtp7iHaO5qNKQPm+7S/7D3ST8+c//AHAozKcQfN92l/2Hukn585/+4EzJiD5vu0v+w90k/PnP/wBwK2YMQfN92l/2Hukn585/+4E1YMQfN/2l/wBh"
    "7pJ+fOf/ALgUaon3K1RSVz/tL6efg90k/PnP/wBwJ1mPcRTTJ833aX/Ye6Sfnzn/AO4E6yew0QfN92l/2Hukn585/wDuBNUT7k4pfPngdpdT/wCw+0k/PnP/"
    "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"
    "yhWli4SGz8Ep1qC73gdwltOb1fSESQdE6WYtj3D723WR6SaH4xCsmL6kcPNMwziy2mLRmI3fYV5ZgRZ1Gx8DTsiNIkAe2g9XuYEW4gKqDthAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBx3xl8elt0K4q8ew4eIbGsasWHRbbP1Dx65SIgvXdm6zhhtCBO+6h3RkHppi1tIh6XPcJclyL"
    "dvVSyquTDO5Hx4ajaYZ9qXBz7QmfPs2O6iY7i2J+wtyiVfmP3Ubc0wBCTg+m7M6u4ttBCu2vlFOqhOtm7Rx2ZJ80T1uzPhqv9kttrz2Hh2S3l2+wnmYFylCw"
    "TBNCB75Ecu9xNztBGok/y212HspNuUZh8t3aAxpOLWfVC56B5U1h2WWybNwW+QXo8p69dCE/OaZ7oB9RlyTGjPOMiW7dtoJ9IyoKmbXHCYry0TWvjk1kzjQn"
    "Hcj0FwsbdPyHVKyY1IuNkyuz3FyMxKdbJyjVXN7TUitK1ZqMhr3Mi3bTpyJWptR7ya8NygdozpwOuDeh7dndlg1mNMSevFb3C76dyE+gbtYAFR7u/ePcidoN"
    "ORcz2dLxqlVmY4qxUk/hl1wvfEXplA1ak6Xz8YtV5jNS7E1dLgw9IlRjHmLpgyRUa5+QqDUqltKnPaXMRzu0THBrE5SOoSICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAghPi44KMO4ryxjLKajZTgmdYLNfk4RqFhMt"
    "pm5WgpDdGpLXJ5t1l+O8AiLjLoGJbRr4SESoHzhH4KsR4U5GTZtM1OyvUDPc5kRncy1BzeY07cLkMYCCNHEGG2mY8dkTPYy0AjTeda7iKpIJtQEBAQEBAQEB"
    "BBPaCcBmnXaMaDscO+rGp2a4vYwyOHeJT2D3CIw9cDjbybjSBmRZLT0fqEDpNE34jYb8vKlRqGH0w7O61Yri+V6c61cWGr2s2J5ljLtiumJ6p3W1SIARXfCd"
    "WqQLdEMDqHMOe+tBGvhpSvKqDWNKuyix7DtQ8RyXV7i11b1Vx7Ta5N3DTXCc+vUR6BY5jQVCPLcNiM09PkRwIhZdkmZN7ql4j5Gg61QEBAQEBAUZgEzAKRqe"
    "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"
    "IzlyFsOuTNC3Uaq5y57N3l5c+XNOqk1wv/m/wv8AjJE/+aJ1VRrh4LO8JIqGWRQ+dPRLqJ1VRrhioD2jdnehPWqJYop24Xht5R4rQVi0erSrtG9tPBvrQSLl"
    "6XLyp1UmuGVPOcIcGonkUOtCHkW5zz0TqpNcMYE3SFqVDnx27K3ItsSsW3PtMAJxGK0pSrTdaU5thyEfIPKnkopm1JFUsTZYelOnGlIaXaa9zh2u22goVpt0"
    "Zwj6Y7KiI8y3VIqlXykValWpVrWvnVqbUq1VN8t4k3AYbMeRUZGhU+p5Fm1V0BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQ"
    "EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBB8IRIdpDzpVBT7nE/6G3+BRTmTEHc4n/Q2/wACiZkO5xP+ht/gUTMmIO5xP+ht/gUTMh3OJ/0Nv8Ci"
    "ZkO5xP8Aobf4FEzIdzif9Db/AAKJmQ7nE/6G3+BRMyHc4n/Q2/wKJmQ7nE/6G3+BRMyHc4n/AENv8CiZkO5xP+ht/gUTMh3OJ/0Nv8CiZkO5xP8Aobf4FEzI"
    "dzif9Db/AAKJmQ7nE/6G3+BRMyHc4n/Q2/wKJmQ7nE/6G3+BRMyHc4n/AENv8CiZkO5xP+ht/gUTMh3OJ/0Nv8CiZkO5xP8Aobf4FEzIdzif9Db/AAKJmQ7n"
    "E/6G3+BRMyHc4n/Q2/wKJmQ7nE/6G3+BRMyHc4n/AENv8CiZkO5xP+ht/gUTMh3OJ/0Nv8CiZkO5xP8Aobf4FEzIDGiiW4I7dK09YRomZFVQCAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICD//2Q=="
)

MAFI_FIG_DIVLIT = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/"
    "2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCALLBkADAREAAhEBAxEB/8QA"
    "HgABAAEFAQEBAQAAAAAAAAAAAAQDBQYHCAIBCQr/xAB5EAABAwMCAwQCCQoNCxICAhMCAAMEAQUGBxIIERMJFCIyIUIVGSMxQVJXYpUWFzdRWGFyltLTGBok"
    "M1ZxdoKRkpe01TRDVVl4gaKyttHUCiU1Njg5U3N0dXeTlKGztcHiRGOxwlRlo/AmJ0VHZOEoOkZmZ4OEpfH/xAAdAQEAAwADAQEBAAAAAAAAAAAAAQIDBAUG"
    "BwgJ/8QARxEBAAEDAgMDCAYJAgcAAQUBAAECAxEEEgUhMQYTURUXMkFSU5GhBxQiVGFxFiMzNEJigbHRouEkNTZDcsHw8SVEVZKyJv/aAAwDAQACEQMRAD8A"
    "/fxAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEHwvKq/xIjk0dxB8X83Q/iq0Q4bWsCbuLesNyvcR68Fcqs1tXcIPeqFRrpl1t/l8wbfN6fe"
    "W1uxut1VK1Th84/OL6TwS6I2/WCHgTeSHPziyY/WA5cqxKBSfNbi1e39M+ezfu28vFy5cx99VooqqqiEzU2/kGV4zicKlyyvJIFsjE5RsZFwlgyFTr7w7jrS"
    "nOv2lGJ5ciKoT6PtdLvFDHZt5793o5fbVcV56G6JQbBlOM5bBrdMVyGDc41HKtlIt8sHgodPMO4K1pzp9pW2+KcZfI+XYjJyB7FYeT29y6R2+pItrc0CfaD4"
    "xN7tw0/vJtnBuhOkSY0KOcmXIBttsak4ZlyoNPt1qmJT1a/4p9di4c+GLOeIuBYG759R+Jzb0zbe+dEJvd2SdoHVET2ULb5tpftKaaMzhXcraNa42bUXh1wz"
    "iEy04WNxcrxK2XqQzNuI9GFWZGaf6XWPZQttXNu7kO7l71Emjmbmbwp0O4RW50GS28y6FDZdZOhAY19NK0rT36Km2rMo3wjMZLjsu9P41DyGE5cYrYuSre3K"
    "An2Qr7xGHPcNK/BWtFbZVyT3kFoyjGshdlxrBkUCe5Ae6M5qHMBwo7vxDoJeAvvVSYmk3PD+XYlFyBnFJmTwGro+31GLa5NAZDofGFvnuKnor6eSnbVVHQ7y"
    "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"
    "0mMgmMD4Jc/RSijaPqmIwCiYBIgeen99TmA6f30zA+iW74FI+0pypyVZjKM4FMxlIkRgEmMgo2ozApxCRRtBTjAKJiQVgUYhGcClIgKMYHyo0rXnWilGYfKB"
    "WledC/7kSVCta86l/wByD0q7QTAKwICrE5CtaU9NUqBWHyp8q8uSrtH2nP4aqwKJjKN0PNAr8KZhL7QRrT7alEzgEdvwqsxuhIQ7vhUxGB9UTOArSlfRVIRn"
    "Dz0/vqcwk6f30zA9JMZBMYClKfAkZRmBSkVYpwCmYyjOBOsJ6CRGEZyKeqRU2grYyCYwPIVr7yiYHpTEYBSCjEAmMgmMAo2gm0fKVrWn2qpkBrWtOdUmMomc"
    "FefL0KxE5B58vShM4fUxhLz0/voPSAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAg5w4x864ydMLbmWsOAaiabYhgeD4UV3YcyWzSbjJvk1oX3Ho7xDIjhAZoINABh1zMn+fKmzYWluI6Szq6NyaI57d9V"
    "tGsR1Qv+HS8cnZLjEC6zMen193tjsiODpxXPRTxtVOoF6KekVnVTipalx5YOKPPdQOJ3VfTvK+0twvS8cP1PpYMZwm42qyVly4ncoTwFTvjgvOEbshwKVpT1"
    "eQreaYxHJXc2pivHRZcFyfU+z64X8pUi3a6NYTp5YbLbOrcbn1bJaJlI7LIel0hdmSXTdLkLbQ1IyEQ3KJtzM8jeyHFu0E0YzPWJ3SPF8Nz+ezGyyRjL2ZR8"
    "HlnYgu7NC6sSswR5U2EJBV2o9LfTbv5qvd1J7yFtsPaacNV9zKFY4cHMgx66ZUWNWnUiRh8oMbnXWj5R6R2p9R2FukDVkHeVGXD8IGSju6jvEZjtReG5/NpG"
    "KRrBnblut+oT2D3rMqYTKpYrXfAuBQO6vzq02DukUAOpTcA9VveQVLkndSU1PXaM6+ZRoRj+mtLBxA2XS+DlepDdmv8AmuQRojka3Q/Yu4yeZd7IWQqT0ZkN"
    "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"
    "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE"
    "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQQgtFrC6uX5q2RxmusAw9LFmlHTbAiIQqfmIaVIq0p9+v20E1AQEBAQEBAQQ5Nrt02VGnT"
    "IDLz0NypxXnGhI2a1Go1qFfVrUa1p6PgqgmICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg+CFKV5qsU4H1JkFYEBUqnmHNWnoPlabRrySIwiOYNa1pzqpJ6PtK1+Gir"
    "MpK1rT3qc0iR56n3lYOp95B6QEBV3Am4FaEZgUYMwKNyRNwVrypzSJyPPU+8pmOY9KQQFXcCbgVkZgQzDz1PvIk6n3kHqtaU9NUClaV9NEBAQFETzHwgpWvN"
    "JjKH1SkQFXcCbgTcCbh56n3lYOp95B6VdwJuBJjIJEYBTEYBRUCsCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAqYkFcE"
    "BAQEBRM4BSCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA"
    "gICAgICAgICAgIOfO0xzvih0s4Qs01R4Wsvx/H7piuKXi+3W9Xy21mvR40K3SJQhFjl7kbzjrTbW54tjYmR7Hduwpo59VK+TeeNTHp2PQJ8tyhOvQmjdLly3"
    "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"
    "+JRFqZhbfC8aa8crerGrc3A8L4d80mY9bsyuWLTM4jyLa5Fi3GE6+y/V+MEusyPH6sdwBfNkRKuyvlcAiVW5iDfDYvD1rli/Eho5ZdbMNtk+Hbb8265FjXJs"
    "AfCjbzjVd9AIh8zdfeKvo5KlUYnC7OVSJyCkEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEGHZrrvpFp3qFjGlWY57Cg5J"
    "mctyNjVkIqnJmmDLr5kIBQqiFAZc90PaHMdu7dUaVnAzFQMW0s1Yw/WHE6ZtiEl6sH2YuFtoUxvpHV+HMfhvDt+11WD2/GHlVTiUZhkpPNAdGTcpQi8o7vTV"
    "Qlimk+smD6x2K55Dhsp+sa0ZVdsemFLY6daTbfOdhSRp9sesye0vWpyqg8XnV2BZM1+odzBstkuUlQI9LlDxt92FzldfafXGmzY10a9Uvea6je7zirbZU3ws"
    "up/FHgmlmqVm0gn43k95vV3jNzHG8bsDs0bdDcmNQwlSNniFrrvANdlDIREzIaABFSYpmYN8NmKi4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICDX3FNpDceILhj1G0GtF3Zt8vN8EvFgi3CQ3U24rk"
    "yG7GF0xH01EauUKtKfFU08lJiXrSzEtcsYzrKHdQtRLNdcTfatreEWe32Uo8m1C1H2S+u9Uy7x1Xdph6B2DStPTz9E1TE9DErBwp8Od14fD1KO932Jca5xqv"
    "d8tilGYIO6szKM0Bk93vmPSrzKno8SmqrPRM0uScq7G7Vq9OXfDIeW6TPWuZrHTPW8+umCG/mErneguXcXphGQhspSrFHw9JtAAbAEjW1F6KWc229dRuA7Ks"
    "24J9c+FeJn9vYuGrN3yybb7s5FMmYFLtKdeaF0PfLp0coJbff2+hZxcxOU7JYDlvZNXDGtLs64duGfMbDiem+e2G0HKxV6C8YWu/wZUUjuEaol6Akx4wi8Ff"
    "ETzYO7uZO7te+jOZNkuj+HzQy56N5Hqde7lfGJoZ7qQ7ksRthoqVitnbbdD6J8/fLdDMudPRtOiwqq5rbIc4at9kaeq+lWn2DzdVqWubjeeZA/ls62sGFL/i"
    "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"
    "No640d1vUy31juGb4EBm7GcjuiQboxNbh3kRCMzaqhGcullk1EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBbcqyK2YfjFwyy8EQxLZBemSqtt7j6bYVM+VPhrtGqDlHCu2S0R1Kw61ah4JwrcR93sd8trNws9zt"
    "+hd1eYmRXgE2n2zbAhMDCokJDXlWhIjMLv7bDp79xrxOfyCXb82hmD22HT37jXic/kEu35tTiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb82mJMwe2"
    "w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb"
    "82mJMw8l2s2nAuC2XBxxNjU/KNdA7v4v/uaYkzD17bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb8"
    "2mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xr"
    "xOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMPIdrPpu5u6fBxxNV2ltLboJd/RX+ImJMw9e2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrx"
    "OfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7"
    "bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMPLnazabsh1HuDjibGnxi0Eu/wCbTEmYevbYdPfuNeJz+QS7fm0xJmD22HT37jXic/kEu35tMSZg9th0"
    "9+414nP5BLt+bTEmYPbYdPfuNeJz+QS7fm0xJmD22HT37jXic/kEu35tMSZg9th09+414nP5BLt+bTEmYPbYdPfuNeJz+QS7fm0xJmEd3tctLWnatOcIPExQ"
    "h8wloNd/R/8Ac1w7t+ii/ETLa3Yrqp5Mb1D7cvhd0jhW656oaA8QGPx7tdmbXbH7xotc2AlTnt3Sjt1MfE6ew9oU9NdtVrF+iqvCO4ripk7Pa1aaywqUbg84"
    "mToBci26C3f8hKbkXOity3Vbnmqe2v6ec+f6DXic/kEu/wCQt9sypNURD57bPpv1Ol+g44mupt3bfrB3fny/iJiTMPXtsOnv3GvE5/IJdvzaYkzB7bDp79xr"
    "xOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB"
    "7bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzB7bDp79xrxOfyCXb82mJMw8t9rNpw82LzPBzxNEBDzEh0Eu/i/wDuaYkz"
    "D17bDp79xrxOfyCXb82mJMwe2w6e/ca8Tn8gl2/NpiTMHtsOnv3GvE5/IJdvzaYkzCti3at6JZHqliGkuQaE624lPzrIBsmOzs30nuFqgvTqsPSBZq++NAoV"
    "Wo7xbff5AShWOrqNFxAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAVaRheqWiWJ6uZHhmS5NKlCeDZK7erbGZIOlKeO3zIBNPiYFub"
    "6U12vIdtdwh4tu4S0iqY6DT9g7L/AEBxnSqJpHZsiykYUW1ZJAKa9cGXZUgLyANPE4bjJbiZZbZaZ9G0AaChCe1afWJyy2Szmfwiaa3XGQxiddLu9DpqT9Wh"
    "CbrJdSb3jr9CvuXKsepV27eW7b6/wqs353ZNksAg9mRpfE00umirupeSVxSViL+N2u3R7fbGXrfBdoA/1SEPrSHQAKABvEfhqW7eRVJWm9MmyV84huATBuIX"
    "KMlyW56l5FYvqzxBjGsoYtEe3uVlQmCkk1RpyVGdOKdKy3txNVHdzH4RoSRdmDZK5XDgys8DMntSdM9XcnxXI5Fyush+6QW4UmjjNwfZekRiZkRzCrdHGGjA"
    "uW8K085CRDWu/kbJYJg/ADaoeqF3tt2bucDErMeFy8auI3VlybeLlaLtcby/JkU2FQAdlzqUOm0CKvV2bB2VV+9+zg2S6nWDUQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEGJ67/AGDMz/cncf5q4g0t2TX+9bcN"
    "/wD0E4p/5PFV8QzdApiAUggICAgICAgIKEjvHfY/Tb5h4+oW3y+HwoK6AgICAgICAgICAgj2/vHu1JDfL9UHt8O3cPqoJCAgICAgICAgICCjcu8DDPu7W891"
    "No7efrIKyAgICAgICT0I6sTvP+zMv/jB/wAUV5PXb41L0mjmiaObUvFVm12wfHcVmWnhjl6onNzqBCet8NloysgGLvO7FvA9os8tpEO0vdfMKtYmua+qL0Ux"
    "X0bvw/8AqKR/yj/0FdrwqZuUzl1/EYx0XddpE/bw6yImqEcO8eyhD0/ce7+bb627y7lIkICAgICAgICAgIPh+WqClbu8Vt7Pem9jvTHqDt5baoKyAgIOWu0f"
    "+zHwj/3UEP8AyayBZpjq65RcQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBV3Dz0/vqcwHT++pH2tK/BVViB9pStPfrzSYBWBVmZ"
    "yCsCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgxP"
    "Xf7BmZ/uTuP81cRE9Gluya/3rfhv/wCgnFP/ACeKtFad02tuXQKbohM0TTaxkRXFXiInFXiIYq8RDFXiIYq8RDFXiIYq8RDFXiImImPWjyGyK4xnetSm3f7n"
    "u9NfCilW6fWkItirxEMVeIhirxEMVeIhirxEMVeIhirxEMVeIhirxEMVeIh9qfWj21smwe5vCe6QdfCW7l95ETFXikInFXiIYq8RDFXiIYq8RDFXiIYq8RDF"
    "XiIYq8RDFXiIYq8Ue6Nk5BMReo34h8RFt9YURiqfWkIYqj1iJxV4iGKvEQxV4iGKvEQxV4iGKvFid5/2Zl/8YP8AiivKa2K/rleZej0sZ0dExDWvEVB4mJ1j"
    "x0eF+9YxBuAZdDcygsqZdNt2yDQ+8hH2CW2QRdLYReHw1VNNNvHOFtRX1jDb+H/1FI/5R/6Cu34VjEus18TjGV3XbU1RFbqY7yjnlFFsvZgnO8Dt7vSnT3en"
    "zebarzMSvmqfWlKk4TirxFJirxEMVeIhirxEMVeIhirxEMVeIo5SRHe3d+BTtpRH2rwhPUL3q/tII9pbJu2sNk8LhC3T3QS3UJWqVpSFVYQEHLXaP/Zj4R/7"
    "qCH/AJNZAonomOrrlUXEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEHzYP2kDYP2kH1AQEBAQEBAQEBAQEBAQEBAQEBAQE"
    "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEGJ67/YMzP9ydx/mrimOcq1VbaZnwfkJw"
    "I9vraNEeCTSDRp7hsk3A8T0xsVnKeGSCAySi29lrfs6Bbd2zdt3LtbGhm5TEvlnGvpFp4dxCqx3OdszHXwbW/TI9j+5Ul/jVT8wtquFy66fpTpqqx3Pzff0y"
    "PY/uU5f41U/MJ5MnxT50Kfc/M/TI9j+5Tl/jVT8wp8mT4nnRp9z8z9Mj2P7lOX+NVPzCeTJ8Tzo0+5+Z+mR7H9ynL/Gqn5hPJk+J50afc/M/TI9j+5Tl/jVT"
    "8wnkyfE86NPufmfpkex/cpy/xqp+YTyZPiedGn3PzP0yPY/uU5f41U/MJ5MnxPOjT7n5n6ZHsf3Kcv8AGqn5hPJk+J50afc/M/TI9j+5Tl/jVT8wnkyfFXzp"
    "UzP7H5pEX/VDtpultl5Q3wtyaBaiCjjZZUO4uqRDTb+pvmp5MlyKPpMoqjPc/NH/AEyPY/uU5f41U/MJ5MnxY+dCn3PzP0yPY/uU5f41U/MJ5MnxPOjT7n5n"
    "6ZHsf3Kcv8aqfmE8mT4nnRp9z8z9Mj2P7lOX+NVPzCeTJ8Tzo0+5+Z+mR7H9ynL/ABqp+YTyZPiedGn3PzP0yPY/uU5f41U/MJ5MnxPOjT7n5n6ZHsf3Kcv8"
    "aqfmE8mT4nnRp9z8z9Mj2P7lOX+NVPzCeTJ8Tzo0+5+Z+mR7H9ynL/Gqn5hPJk+J50afc/M/TI9j+5Tl/jVT8wnkyfE86NPufmfpkex/cpy/xqp+YTyZPimj"
    "6T6Zn9j80m7f6oatOJFHbkcLchz2QihPHp5UPhF3xbf6m+8nkyV6/pOpj/s/NG/TI9j+5Tl/jVT8wnkyfFn50Kfc/M/TI9j+5Tl/jVT8wnkyfE86NPufmfpk"
    "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"
    "QT1fD8tUFC1k8VvZKQ1Ro+nTc2I7dv71WqVpSFVYQEHLXaP/AGY+Ef8AuoIf+TWQKJ6Jjq65VFxAQEBAQEBAQEBAQEBAQEBAQEBAQclcMEuLxfcWOaccJXl6"
    "fiWIPTdPtH2ajtjkEd4Rvl2a+PWROj90Ex/rVtoQ/rvhtEKzLp6Q90I5yOmZ7BItrY7iL5tKfGVlUezxCbj1nSIosyZe1yUIkRePaI7f3o02qZkS1BzUbhcI"
    "lpt79zuD4NMR2ScedcLbQAEd1a1/vLK9Xbop31TjbEtbFq7qNVTbiM1VTEPwg1FyDIu0m7RgWBA3YF/ySggDfli2aN734Jd3bIi+MZ/OXjOzNNrVarUcau84"
    "6U/2d79KtVenp4f2UsZiec3Jj/8As/XmDBh2uCzbbbFbjx4zNGo7LI7QAKDtERp6oiK5VyuKqpl2ulsdzZppn1RD24422PUcLaIqKaM821dzbyh4jtuU92kC"
    "O8vi+qPxVrcnMKW49cqiypbVCgFMTMHVR29z2i23To7S3Fu8n/tW0xTfpzPVx/tWKsR6KssIiZqxPRvmmIzT0fHnW2W+o4fhVoo3Sia9qn0SkFue/W6j+skP"
    "+Mtd8WoYTRN2eSqscy5EREChIgICAgICClI8ge6bfdKLWz6DG91VVnV6Tan0RQCAgICAgKK+UlMZ6vDzPWHaLhBX1SFa27kQzuW89Hlt5zqE281t2+UvgJTN"
    "G9Smru1VZTm3yb04uQ8POEIdNgRJ31RIv8JaU045sq7u7kNsg2RObfGfmJTVXjominPN7WMTXdoia/WtVXRE7qp6ODdYBh9pF2j2P8LdmZedxPT1913I5FPE"
    "LtGyApRU+LzKjUehV9YiJcPW8L43HErMWqsW6uqOGdpOymu0Gts3LcV3beMTPTMv1TatVvYtgWdmI3SKDItCxt8FGxHbQdvxdq93ao2UxDwt2YrqmVCDKlR5"
    "h2u5PM8zcIoJCXKptfa5fbHntWkwxzOeacqriAgICmkWu191styPH6OO8nt8iL1B8Iju8YCXzSru/fK081ali4g9EcL4k9E8n0I1CbeKz5TaXYEpyK5sej8x"
    "8DzJ+o60dAdAvVMBL1VnMZWjkwjs9teMr1e0Sk4Dq9eqS9S9L769h+pJE10jfuEQRq1O2fAEyIcaaG30bZW31aqjRvxAQEBAQEBAQEBAQEBAQEBAQEBAQEBA"
    "QEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQYnrv9gzM/3J3H+auIZw0t2TXo7Lfhvp/wD0JxT/AMnirWimIpViYrnvJ6ugVWia"
    "c9WeaapwK26PaWxSJuj2jFIm6PaMUibo9oxSJuj2jFIm6PaMUibo9oxSJu/mMUij7PiirbFEyjyBcK4RjF4aUHfuHd5vCrYp8U1TTTXTHikKN0e0nFIm6PaR"
    "ikTdHtGKRN0e0YpE3R7RikTdHtGKRN0e0YpE3R7RikTdHtGKRN0e0YpFOf5iYphHt7bjfW6j1D5yDqO0t20fipn+ZFNUVckhRuj2lsUibo9pGKRN0e0YpE3R"
    "7RikTdHtGKRN0e0YpE3R7RikTdHtGKRN0e0YpE3fzGKUe6C45BMWnqNluHxEW31hUfZ8UUTFNMpCfZ8U0xTNcindHtJxSJuj2kYpE3R7RikTdHtGKRN0e0Yp"
    "E3R7Rtplid5/2Zl1/wDmD/iivI63up11b0mmiKNJRbpjGGteIoeJcrHjtOGEsYG4/VdD+qj6qOr0vYTafeehs/8AiOfT2bvD76tam3FMF6i/FNMxU2/h/wDU"
    "Uj/lH/oK7ThcUxHJ1nFIorqpXddtEUzV1dXMXdsxSji257LE51h2d3pTp7vTu3ebar4paUTG3mkKv2fFSmKdwp3R7S+KRN0e0jFIm6PaMUibo9oxSJuj2jFI"
    "m7+ZOKRUmqvOMqVYmZz6hWmiY5k26ZmmuZEWfD8tUFC1i41bY7bjwuVFsdzglu3fvlapWlIVVhAQctdo/wDZj4R/7qCH/k1kCieiY6uuVRcQEBAQEBAQEBAQ"
    "EBAQEBAQEBAQaC7RbVzUTTjh7LBtDprTOoupl6jYZgTznppEnTt1HJ1afCMSIEmYX3ov30GwdFNJcP0F0jxrRXT+AMay4rY41rtrfw9JloQEq/GMtu4i8xER"
    "EXiJaM13llHul4atoSDHuJBIeEfLXzbKVL8LxfvaK/KGa4qjQUxHLJlyV2yfEgxoXwiXDELZdOhes4OtqgttlyPutabpR/tdLwVL4zwrxfbTXXLHDadPZn7d"
    "zlH/ALfQvo94bb1vFPrl+P1VmJqqmen4NQ9lPgOgczh1sWqGC4w09kjYyYd4yCVa+jJJ4zE3WaH67QeARLdt8FfLuIVxuG8L1HANBRpb8/Znn8ebLiHaHgnb"
    "fjVzidinnTOIn8uTqtc2afBealEx7w9tcbHpByqPzqrSicQwqp3Kyzmcy2iMChIgICiczVup6IpxjbPRGcL2NHcLfNkR8o+ahfkrm0xTqKcR1cK5NWmqzPoq"
    "rLdCc7w4VSqQj4d3hFYT+qb0/rYyqLKf1jemO7gQEBAQEBAQEFKR5A9z3e6UW1mJ2sb3VVWdXpNqfRFUEBB5bcbe3E24JbS5FtL1lOJhGYl6UJEBR1OgmyTd"
    "l8eZbebq24PMSWlFzapXayjuSHI5FH8G7bTo7nP8ZcmmzF6MuJVem1OIVmWgboRejfX9cL4ywrnE4cqm1nm9rLGWsZpar41NfmuGvhvyLU6K8I3RuL3WwtkO"
    "4jmu+BrbT1tvpMvmgS10+p09vUU0ajlDq+NxxOOG11aS3uuTGIx+LG+w44WrhpFw6S9e86YI8k1JkDN60jxPBAoRdPfUvFudMjdr8YSD1l6C1V9Zu1VxP2Z6"
    "PAcG4bc4Nwum1ejF2Zma5/rydwLlbtrsaZ3It0t4zmRq02HXZLqRXHN20HNvoLw+qrU1ZKoVLbKemQwflR+i9t92Z3eQviqEqyAgICmeQiXuNOeh77XIBuQ0"
    "Qm2RD4SoJekK/NKnhVqcSrUrxZcadHCZFeE2nRobbgl6CoomCJc16sOXXhf7QDCdfLYMZnC9aI7OB6gBQaB0b8z1X7DOrX54lLgGReIidhh4toCNJjK2cOrF"
    "RoICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIMT13+wZmf7k7j/NXEGluya/3rbhv/AOgn"
    "FP8AyeKtGboFAQEBAQEBAQEBBFldH2Si9Qj3+Pp7fL5fTuVqRKVQQEBAQEBAQEBAQRrX0dr9WSP+qj3bvjesgkoCAgICAgICAgIIt4qz7Hn3gq0DcO7b7/mF"
    "KRKSsEBAQEBAUT0GJ3n/AGZl/wDGD/iivJ6zlrZej0lX/Bw05xfYxi+VYziEfKuKa66UNxdQLdJi3K05IFtK+SAF3ZZzMyHrtPcyImR8R9Knh8Ktpa8RKdRE"
    "zhvjD/6ikf8AKP8A0Fdvwn0XW8RXddp/G62n0UQax/ZovGfU7qP7XLcrVCWqggICAgICAgICAXvV/aQRbP0fYtju5GQdGm3qebarVK0pSpHRYUgg5a7R/wCz"
    "Hwj/AN1BD/yayBRPRMdXXKouICAgICAgICAgICAgICAgICAgIOT8YrZeKXtMsj1DGY/KsPDnZPqWtLVf6mLKbqyxMuD1PjHHt/cGBL1azZIq8dFJ6umpEhmK"
    "yciS4IAA8yIi5CNFelCNZ2ZzcPrXQQpJcIjc6fq+Lwju9baO0UqEtV2bo2wj0qufqFn3lW2YnpDTZTFUVx634qdsTrTlnE9xa3DA9P7RMuNqwerdjj1gsm6N"
    "ZJuiL7h7fJueqLNPjdKnxl890+qtca7XRP8ABZ/9w+gdrKNT2X+iuadPMxd1ePziIn/D9CeGTRez8Peg+MaR2dkQ9iLWATHB/r8s/HId/fOkZf4K9Fr7/wBZ"
    "uVT6vU892Y4Nb4Lwe1an0uv9Z5s1kPdPk34tzhch2/4ywtRjq7q9VmMx1l7ZbFlqjLflFUuzkt07Z2+p6VWggICD445Rturm2pbfVFWojcrXVFFO71vDLO4u"
    "8PCW8h5bd24R+atN3ddGW3vad09Xn+o/iCwI/wARW+zqKcz6SPtWKsR6Kt5vSNFhzztnq35Y3R0FCRAQFG71I28hWijKO8w8FIZEib6g8xHntHxFt/aV4s5V"
    "7547w8XImYpkNfWIuW1Xi1TCs3apff1VuPn0h/4PzF/GTbTCN1cqcgJnTD9VNgXU8XuZeL/CWlmuIpY3qL2fTVdszw7nG6/G8JLOquNzSmi9t9MJyQIVIo4l"
    "y8oiXiL+Mo20y03VQd6bGgdYSAj9UxTuaTvaoWjUKZmEPCbjJ08toTbx0dlvZJwKDvqQjv8AGQ08I1ItpEO7bt9ZXtURRc/WMtRc30/q+bD+F+05BYcNu9ny"
    "HE7paibyacbPstKB56QJnuJ0jAioW4ql4vKXq+Fb6+5TVbiLamjpnd+s5Nlrr3PqiBRtyROBT0U9IU7srbcQ8PPba9JmlKuVHmIktKaNrCqvdyfBitm3UXhE"
    "6mPJwtvmV5vbeikWcvjbjjR1Ze28ql7mW7zfNUVUbqO9jqW7lVurNz0fUqrCK67sZ9bkRRqL2aukODeKG4ZDx6doVivBDiAVdxfFpQycslRS8hCO6SZF5R2N"
    "cmB/+a6Q+sKjivZudbp7M1Tir1Ol7P8A0hanhfHL+ksUxXTRGKp8MxyfqhYbHacZskTHLDb2osKBFCPDjMjtBpsB2gFB9WlKUoK9XYsRYsRRHqdBqdROrqqq"
    "rnNVUzKUtmPQQWyZ3Wx3L2WLq0Ca80xI2+QT8oHX8Lwh/ArxOVYhc1WuFhQCAgILfaB7jOk2dm3kywG12O4O7YW8i3D80hLd4fikKvVzVpYFxjcPNv4qOGnL"
    "dDJFydt8y8WsjsN4jltdtd1ZIX4M1qvx2ZLTLo/gbVSOSytwQcRUfip4XMS1ldZpHusyAUPKrdy5Fb73EcKLcohU+21LZfD79BpX4Vm0bbQEBAQEBAQEBAQE"
    "BAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQYnrv9gzM/3J3H+auINLdk1/vW3Df/ANBOKf8Ak8VaM3QKAgIC"
    "AgICAgICCPIcEbhGb6I13b/dPi+FWpEhVBAQEBAQEBAQEBBGtrguDI2siHKUdPD63zkElAQEBAQEBAQEBBHuzgtwTImKO+IfCX4QpSJCVggICAgICT0GJ3n/"
    "AGZl/wDGD/iivJ6znrZej0lP/BQ0bxvZVwy4nieESOKDTd/JrfL1KtcbF48e0lMrDvZi/wB2lVESHYIDR3x+Ud3zlOlozEp1E4mHQ+H/ANRSP+Uf+grtuE4i"
    "l13EV3Xa/wAbrKfRRhcH2YJvu9P6lpXqet5vKrVCSqggICAgICAgICAXvV/aQR7S4LlrYcFmjXNsa9MfeFWqVpSFSOiwpBBy12j/ANmPhH/uoIf+TWQKJ6Jj"
    "q65VFxAQEBAQEBAQEBAQEBAQEBAQEGHa/wCs2I8Omh+W68586Q2bDsdmXi5dPzm1HaJ2oB84tu2n360Qaw7O/SnLNKeE3HPrmWYIebZWUnK8+bEdp+zd0eOd"
    "LA6+Yukb3QHd5QYAfKIq8dFJ6tu3zbNcjWNyORtyyKrxCW3aAeL/AAi5Dt+crwhPSoFTd3dVUq3K/s1Yau4zte43DNw05brE4+yEq224gtTbv9cmuV6TFOXw"
    "+M6VrT7VKrpO0PE6eGcFrv8Ar6R/V6bsrwWri/G7Omn0Z5z+Uc5/w/NrsVdRL9ftQM/x9zCazhlNtT71mD06u8XN5C3HICb8ZGRvHu3D5C8y8twDg13Q8Ht6"
    "qfTu5n5u+7V9v6eP9r7vA7GnibOnxEVZ5c458n6HLt4nFUUM66I7ymM5x61JkiccN7qcwr6GxFa1/ZhjbnNU1T0lVWGcuRTTtjb4iAgICmIyiZwos8pLgyvG"
    "IjuFsS/xlvXEWo5MKM3a909FZcaZ3y5G2KJ3+oVuec09UfZxirookRQ/FWojHoPl2+T/ANq3+zfpxHpMPtWKsz6Kt5vSNFx8THJyMxMZgQ6iRzRM4eHJAi53"
    "cBKpkPPb+UtItx1ZTceOg882PenOXi57Wy2irTVEIiJlVFtsSIxbESLzfOVZu4Xiy+rLdVLXbTApxUZpiVORs2huHd7pRbWfRYXps7uiosqozU1pmzs6CjbU"
    "ndSKd1UIxTKmUJvaVY9ekVfTubWtNzMZuKRYzT9jkUcebLa83zAR/XB/JVsd5OKGVH6uf1nN6bcbeHqNkJUL1hWc0Ya017npZTOGu3MCtVz6FMRCm89tLot7"
    "au1HmIl/6rS3aywu3ZjlD6230/ERbjLzFt8ypVc9SbdD2qxG6ct87Xxxpt4drg7vFzV6K5ou8+jKuKblvZU1/wASmt9t4fdC8i1UvBgEi3wTC2sl4u8TD8DA"
    "cvikdabvijzL1Vx+J62jhuL+OTkaLhnEOO250WmnFcxOPg1n2GHDJc8P0rv3FnqI45IyfUae5WPIkekxhA6RE7zr6ak89vKvzQAl6LRa2riVunV/wz6Mf3fN"
    "LHZrU9kIu6fUzm9M/bnrnw+TvVdlXuotTM9W04nnjE+oUwCDy831mTZ3EO8SpuEvSKmiREsch0ofcZkwHpMUulIcEdu6u3dQv3wkKtMZVmU1UWEBAQW++CMc"
    "mL05ONkIblavbR3CYEO0hL99yLd81XjmrSuCrVyWc18NU9vQXtANXeGErMMGyagQY2qGFuNhybdku7YF9ZH4KEMlqHLr6xFcnCqsmjqhAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBieu/wBgzM/3J3H+auIdWleybMB7LfhvoR0p/wDiJxP36/8A2nir"
    "aimKqd2HHuaizaq2VVRy/GHQPUb+PT+FVmK6Z5Uqxf0sxnfHxg3h8an8Kvt/lO9se1Hxg3h8an8Kbf5TvbHtR8YN4fGp/Cm3+U72x7UfGDeHxqfwpt/lO9se"
    "1Hxg3h8an8Kbf5TvbHtR8YN4fGp/Cm3+U72x7UfGDeHxqfwpt/lO9se1Hxg3h8an8Kbf5TvbHtR8YOo3z5b6fwqJt/yo+s6bpvj4woyDf78xVpynS8fUrv8A"
    "veFTtx/CRqNPVGd8fGFbeHxqfwpt/lT3tj2o+MG8PjU/hTb/ACne2Paj4wbw+NT+FNv8p3tj2o+MG8PjU/hTb/Kd7Y9qPjBvD41P4U2/yne2Paj4wbw+NT+F"
    "Nv8AKd7Y9qPjBvD41P4U2/yne2Paj4wbw+NT+FNv8p3tj2o+MG8PjU/hTb/Kd7Y9qPjBvD41P4U2/wAp3tj2o+MHUb+PT+FKaJiPRRTqtPHLfHxhQhHIHrd7"
    "cpX9UH0/H6nqqlNE7vRJ1Gnpn04+MK+8PjU/hV9v8qe9se1Hxg3h8an8Kbf5TvbHtR8YN4fGp/Cm3+U72x7UfGDeHxqfwpt/lO9se1Hxg3h8an8Kbf5TvbHt"
    "R8YN4fGp/Cm3+U72x7UfGDeHxqfwpt/lO9se1Hxg3h8an8Kbf5TvbHtR8YN4fGp/Cm3+U72x7UfGDeHxqfwpt/lJu2Paj4wo3A3u5nSE7Sjvh2+P5ybMfwpt"
    "6jTzEfbj4wq9Vv8A4Qf4UmjP8LOnU6eMfbj4w+7w+NT+FNv8q/e2Paj4wbw+NT+FNv8AKd7Y9qPjBvD41P4U2/yne2Paj4wbw+NT+FNv8p3tj2o+MG8PjU/h"
    "Tb/Kd7Y9qPjBvD41P4U2/wAp3tiP4o+MMUvFaVvEutK/1wf8UV5HW26o1tVGJem0Wot/U6KqKoxP4tacRmYcQWG2PHZfDvo3AzSbMy6HEyKLcL01BpbrSYn3"
    "icBOkPVNoqNbWh3EW/wj4Va1bp+1yle/cppinbVGfzbfxAxGFI3FSn6o+H9oV2nCqIrsVfZl1XFNRatVUxVVHxhd+o38en8K7bbXTREbXWRf0lVM5rj4wjib"
    "/snWvVp0e7+XePn5q2yZj0VKtTpqZjNUT/WEjdQqeEqf3lnERjdchtFdF37FqqJ/rD6p5LdBAQEBAQEBAQfK89vhQUrb3j2PZ74W5zpj1PF8KtUKyqCAg5a7"
    "R/7MfCP/AHUEP/JrIFE9Ex1dcqi4gICAgICAgICAgICAgICAgICDlztI7pcdQbto7wb2SCMmmqmp0RzKBIeYs49ZhK7zSP5jrkWJEru9Bd82+spjqiejo9XU"
    "QLbTvV2mXJu4VdbHbHbZEi5AQbt/77dX/BVqlaU9VW6CTj0aeqJqimJrrjMy/NDtrNScs151kwrga0cilPuwFW6T4ovCAOSjacFhqpF6B2Mi6ZVLw7XB++vl"
    "/a3V1a7jdnhs84zmf7vtPYvhur4R2W1XGYo+3NOLfzif/vwZx2YfCrkHC7w+HC1AtfdcqyG5Ozb5FJwDKONPAyzvAiEtoDv8NfM6S9jrtRTVVTbp9GmIx8Hy"
    "LsXwO/odJc1er/bXKpmr4zh0ZIJ0G/cW9xVLkuHYjF78JetrrtzRNE+vo+tNtsti22O0RHkIqlU5lpRRth6VWs1CIEBRjKJnCi8XWc7u25yqO2rm34q5FH2Y"
    "YVZqlWWM9W8coFCRASJxJMRPKVEi7nucJz3HaPh2+T/2rea4n82FUVWqsR6KruEadTdTbt57lnNvd+bWblNqndT0UvdJHMfK1UfC4JekvwVpExDGYm7OVVts"
    "WW6NtjtEVlXOWlMd2+qrQQEBB4kb9oVFzb7pT98trPosb3V7WVXpNqfRFAICAmZgU3WS3CTLmzkXiHb6CWlNcSwrofWXqvAW5uoVEuRCSmqjcmi5sjEjjwiX"
    "QGvuhDXaP5SUUbOpXXv5DLfSH3Qtx+s5t8yV1r26PF7WS4gK1NVm5TETGMJ22enSZ9Tgfj3u2VcZHGlhfArplMddg2p4Z+XOQy8hbOq5zr8ZqPz2j8d/b5tq"
    "7aeHWNRwubV+N27G3+nV821HaHitPbW1b0NzbTZzuqj8YfqbhmNY9hmI2zEcSt4Q7ZbLezFgRWx20aZbARAKD80RFc/T2Pq9EUx0h2Gs1NWquTVVOZn1rmuT"
    "V+sYxTsjmIqICCBKach3pmZFt1KhL9ymPD5h5CRAVfm+Yf3wq1KtSeqrCAgIPEiOzKZOLKZobTglRxsh3CVEoFCyuTHLaHsmIi+G4HBHw03CW3d83n5v3yvP"
    "Qc59oPcZ2jmYaMcYdsi74+BalRrPlhj742C/bbXJIvmNSXbdIIf/ANFoXqrOeiY6uqlRcQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBieu/wBgzM/3J3H+auK9rncp/OGd39lV+Uv5sOHPU7Um38P2DQYGoV8jx2cQtrbLDN1eEGwpFbERERLwiv1N2a4V"
    "obnALVyqiM4h+Lu3HGuI2O1WqtUVzERVPr/GWafXZ1T+U3IPpl/8pekr4Vwunb+rj4PJTxvi826cXJ+L59dnVT5TMg+mX/ylXyRw33cfBXy1xT30/GT67Oqn"
    "ymZB9Mv/AJSeR+G+7j4Hlrinvp+Mn12dVPlMyD6Zf/KTyPw33cfA8tcU99Pxk+uzqp8pmQfTL/5SeR+G+7j4Hlrinvp+Mn12dVPlMyD6Zf8Ayk8j8N93HwPL"
    "XFPfT8ZPrs6qfKZkH0y/+Unkfhvu4+B5a4p76fjJ9dnVT5TMg+mX/wApPI/Dfdx8Dy1xT30/GT67OqnymZB9Mv8A5SeR+G+7j4JjjfFM/tp+Mvv12dVOXP65"
    "uQfTL/5SmOD8Nn/tx8GlHHeK99V+sn4rratWdWwxW7A1nN+dCpRupKK+Pc2PGXLb4vhXGu8J4dE/s4+DbT8c4rOnrnvJ+MrT9dnVT5TMg+mX/wApciOD8Nx+"
    "zj4OJ5b4p76fjJ9dnVT5TMg+mX/yk8j8N93HwPLXFPfT8ZPrs6qfKZkH0y/+Unkfhvu4+B5a4p76fjJ9dnVT5TMg+mX/AMpPI/Dfdx8Dy1xT30/GT67Oqnym"
    "ZB9Mv/lJ5H4b7uPgeWuKe+n4yfXZ1U+UzIPpl/8AKTyPw33cfA8tcU99Pxk+uzqp8pmQfTL/AOUnkfhvu4+B5a4p76fjJ9dnVT5TMg+mX/yk8j8N93HwPLXF"
    "PfT8ZPrs6qfKZkH0y/8AlJ5H4b7uPgeWuKe+n4yfXZ1U+UzIPpl/8pPI/Dfdx8E+W+Ke+n4y+/XZ1U5fZNyD6Zf/AClWrg/Dop5W4+C3l7isXIjfPxld8t1c"
    "1cJy294za/ROVpj0HbfHq9Udvod8JessLXCOHTV+zj4OVquOcViiPtz8Vn+uzqp8pmQfTL/5S5Pkfhvu4+DheW+Ke+n4yfXZ1U+UzIPpl/8AKTyPw33cfA8t"
    "cU99Pxk+uzqp8pmQfTL/AOUnkfhvu4+B5a4p76fjJ9dnVT5TMg+mX/yk8j8N93HwPLXFPfT8ZPrs6qfKZkH0y/8AlJ5H4b7uPgeWuKe+n4yfXZ1U+UzIPpl/"
    "8pPI/Dfdx8Dy1xT30/GT67OqnymZB9Mv/lJ5H4b7uPgeWuKe+n4yfXZ1U+UzIPpl/wDKTyPw33cfA8tcU99Pxk+uzqp8pmQfTL/5SeR+G+7j4Hlrinvp+Mn1"
    "2dVPlMyD6Zf/ACk8j8N93HwPLfFPfT8ZXXDNWdXByRkoucX6YfTd2xyvj1BL3IvjF6vm/erO5wjhsR+zj4ObpuO8VnH25+K1/XZ1U+U3IPpl/wDKSjhHDp/7"
    "cfBxbnHuLRMRvn4vn12dVPlMyD6Zf/KWnkfhvu4+Cnlvinvp+Mn12dVPlMyD6Zf/ACk8j8N93HwPLXFPfT8ZPrs6qfKZkH0y/wDlJ5H4b7uPgeWuKe+n4yfX"
    "Z1U+UzIPpl/8pPI/Dfdx8Dy1xT30/GT67OqnymZB9Mv/AJSeR+G+7j4Hlrinvp+Mn12dVPlMyD6Zf/KTyPw33cfBPlrinvp+MuqNG84zaXo/jMqVmF1dddgu"
    "1cccuDpEX6qf8xbl8L7Q6LQU9o9RRFMcser8H6E7K8U4jPZexVXVMzOfX+KhqpI1iyy32qPhOuV1xl6JfI8qdIbcJ7vkYN2+L4i8O/cPi83hXQRo9LmZiIeg"
    "scQ19NWaqpw1txcakaiW3MrK3bc8vUcDsNDIWbo8AlXvD/i8JL6p2G4bobukqmqiPg+Q9veK8RsXqdtc/Fqj67OqfP7JuQfTL/5S+geTOGUxTHdx8HzmON8X"
    "m3M95PxXc9XNXPqDBv6tr90fZYq9+9nHuZV6Q+5bd275y43kjh31iZ7uMfk58cd4t9TiIrn4ulOxmz7O8g44bZbr7ml2nRisFwqUeZcnXQIhbHkW0i5Lw/0j"
    "6PR2uzs12aIicx6vxh7/AOifX66/2lii7XMx+f4P2IX54frEQEBAQEBAQEAver+0gi2cW27XHbZe3hRkdpbeW5WqEpVBAQctdo/9mPhH/uoIf+TWQKJ6Jjq6"
    "5VFxAQEBAQEBAQEBAQEBAQEBAQEHLGn90ves3an6kZQ+0RY9o5pzasSs7m3wld7s77KXTaX2xjR7KNR9Xd85THVE9HSEp4o0V2QLJGQNkYtiO6tdo+WiuooW"
    "WKMS2ttjFoyRj1Hmx5+Ez8R+b51aq1StKUqrR1WzOMtsmBYfdc3yOWMe32i3OzJrxeUGmwqZF/BSq4ms1NGg01V2euP7OZoNLd13E6NNa61TEfGX4faC6+a3"
    "67dpfb9ScLuDbF5yzIX3Z3XhhICLa67icapvHwUCM1soQ7S8Pm8RbvnvZXh1fEdTf4xq+lXo/wBnu/pM7S6jgOt4f2U4bc5xE7/hExl+ti7259u3j1w41NVd"
    "MbrvX1qNNr0rdTd7j6Pm7qrfMUabHrllNNq7OY/hVlx46ORVMCtEJiMihUQfHCFsSIuXoV7UZUvThTjNOC3ueGlHT9Lm1WuTjorbjKqsmogICARiI1MipSnr"
    "bkoiZn8Vaqot04q6IZeHe7IPfGrtqI7ee3534K59MRFP8zgTFdNWZ9FMXArmcufREUQKKea0xvFIICAgpytu0Nw7vdBWtn0WN7qqLOr0m1PoigEBAQEjkel1"
    "UJXRccBum7q++2Q+r/7VyrcuHejE8n2H1Bb2yAr1aeYi9b9r5qXsR0LMZnmrLi43dXOztEVEGIa96w2DQPSG/auZLtKNZoNXRZJzZ13vK01SvqkZ1Ef3yz1F"
    "65b001UU5le3HD7lURqLm2Okz+bSnYg8O06Xb8u44tRYxv3/AD24yGrRKkD70bqkUh0P+Ne8O74rHziXN7L6zW6/h9N3URjGdvx5vN9o+zPBOzPE71Ogq3zX"
    "iap/s70hkNuuDttdmGdZLhyI4kPlHw8x5/hFu/fL1OYrec2bKk5RP2Gtc7o5ChAgII92t43a2vW8nqtdQfC4PmCtPFQh/BLarUq1FruDN0gtTmRMaH6rg8qj"
    "WhbSEv3yqskICAgJQLbBbh2/IJUNtwxcmiMvp7fB4doEQ/4G5XnoMV4pdGovERw351obKc6f1V4rOtjL27aTDzjJC26JeqQGQEJeqQrOeiY6rX2f+vt/4ouC"
    "rTPXfNoNIuQ5BiMRzKIdW9nd7s2HSnNbfV2SW3h2/BtVF25EBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEHN+F9p5w25twvaj8VUKt6j2n"
    "Su9XW05RZZVuoNy79DKlG2GWaF7qcqjkesegl7r3pofQVaiIbEc4n9P4/CixxgXCJcI+Mv4MzlIw3GB753Z6KMkGaBQuVXyoYhQKF4jrt5oLpoPrhjHEBorY"
    "9bsXtlygw71b6vuWy7sUZmW94K1bfiSApWtAfZdBxox512mBU5oMc4Y+JfI+JfE7XqI3w8ZXimOX3Ho94sV5yC5WpwJrL4NuNCLUOY86B1A6H4wHlyrTzehB"
    "k2hGtuLa/wCAlqJhsSazBC/3e0k3PbEXetb7jIgPV5DWvhq7GMh+bWnve8gwiDxv4SfEPZuGm/aTag2S+ZFKuLVhnXbHBbg3AIQVcffbcF0i6NB2e6VCg83W"
    "hryIxGoZBrvxCydF79imHWPSS/5nfMxnS49ptOPyoLB07vHKQ6ZnOksNDSgD8fnWvwIKWFcXWh+S6YWvVHLM0hYXHud0mWqlvza5RYD7FxiSnYsmEW52rZut"
    "PMuBXpmYlt3DUhrSqDOZWbYhEjzZ0nLbY0xbXQbuTzk9oQimYtkAuV3eAiFxshpX36GPxqILblOsulmG4Y9qDkGf2dmztQZMsZ1bk1sdajgRvEBbuR7BAufL"
    "3uSDCdF+NnQTX96xu6cZQEmBkeEWvJ7Tc3XmwZdjz33WGY+7f6JXUaICa98S8Pv+hBtWHcbdOkyYsOew+5DfozLbaeoRMHUBOgHSnlLYYFyr6pDX4UGDzeJL"
    "TC0an5fpllF8CyHhWOWu93y9XiQ1GgMxp7sxpn3Yy5DWhQXd27lTxB7+5BdLvrronYcJh6mXzWLF4eNXIxC3ZBJyCO3CkkW7bRt8j2HWu0ver6tUGKT+NHQG"
    "wa93Hh8yzPbbZLpAxey3tifebpHjRLizc5E9iM1GM3KdV3dbniqNKe8Qct3P0BbdR+NfDtMc3n4BetO8hfnRrtLt1vrFeg7Lk8xYXL2fSqUmnSpVlurVOt0v"
    "deXqeNBm5a/6MRZkSzZFqdjtnu8qBWZ7BXTIIrUxpqjIvnU2+r6jRUMq05iI+Lnt9KCFknEvpHY8XxfLrLlMXILfmWSR7JjcnHpjUtudJc6pH0zAthC02xId"
    "PkXhBg/hpyQaaY7W7hdk9njF7SWHByR7EJdQYbx9m2DW8jNrL7oUMo2/wug5QiLxbaNDV3ns8SDdgaxuScqYt0fCpBWM3IzUrKTvNvCLFffbMgjuN1f69Hd/"
    "dm9vT8RTGtu6m+ohA0C4rNEeI7F4d+09zq2nMk2sbhKx926RyuMBipcqHIYAyJqnvemvo8VEF5tOvmhN8xqTmli1pxOZZ4cysWXdouRRjisSKemrRuiewT+b"
    "WvNB8mcQGhUHBYeqU7WnFGMYnv0Zg5C/kUYIMhypENAB+p7DLmNacqF6tftIMK0l42tI9V7XKzBkHrHjcO2ypkjKL/cIUeEAx7rMtpjUuvvHc7DMxOo7CAg8"
    "W6u1BLvfGJphEzWHg+Jsycpcm2O23iNccfmRHYbsOZdQtoug8b4gex095CNa1qAls3ntCoZHpJrxp7rJjNoyjHriMU7332tttlwebCW+EWScd5wW6FWpCJh7"
    "489tDHny5oJuu/2DMz/cncf5q4tLX7Wn84Z3v2NX5S/mV0A+wRhf7k7f/Nm1+teyn/ILP5R/Z+HO3/8A1Zq//Of7yy5ein0ZeNjpIpSICAgICAgICifSk57Z"
    "x1Xizs3AsRvT0eYIMAUbvDJN8yd3GW3bX1dqxv8A71Q5mnx9VuTHTks63cMQEBAQEBAQEBASv0FvXC85ozcWXLX7ITge32Vg2drezY3t8IfO2/GXGt+nLl6v"
    "1LMuS4QgICAgICAgICC8YExPkZSw3bZlI7xNvbXib3iPuR8/D+DuFcfV/uzlaLnqcLOt/wCFxqudYpQICAgICieiY6us9FfsMYt/zc7/ADp9fA+0/wDzy/8A"
    "0fonspVb/RuxiPFj/EgzonIseNjrldnocWmYQq2MmSepvum13oAXQEi2l4/N4fjLoK/Rh6ixVX3c82E8Y3+3ayfufp/OX19a7BfuFf8AR8V+kT95t/1aiX0P"
    "1vm8ekvLjNw+t23I78HdPZowGP0/F1OkPj5/g+HauNT+3c2f3B072Jv+7utf7nrj/wCEK+f/AEmf9O1/nH/+ofSvoi/6qt/1/s/Z5fnDZD9d0iYmF4nAm2Vc"
    "wJiUiYwrSKFhAQEAver+0gjWgmStcco7dQAmR2iRbttFapWlJVI6LCkEHLXaP/Zj4R/7qCH/AJNZAonomOrrlUXEBAQEBAQEBAQEBAQEBAQEBB4cdaYbJ506"
    "CAjzIir6KUQcudlreHtTOH2/8TcmObZawamX/LYvUHaZW85ZQ7bu+d7HQ4I/vVaIZug8gFuRHYtpTKslJlBQdo+Ko0LfUfm7hAlrHRWpPVFhBxH23uvzuGaA"
    "Wvh5xWY77P6hXIGCjsV5H3JkwqdPD729wmQ2+sJHT4Kr55274jXa00aO31rw+ufRNwaq/wAVr4nd9C1EzmemZiY/+/NgvAL2ctw4SdRL1qjlmYQr3MudjjxL"
    "f3eGTRwiMt8kC3EW7xA0IkNfEIluEfKuz4fVOi4Jb0nqp/8Abw+s4Pe1vbbV8a1Fe/dMbf6cnVxV203ErW3dVT9nEvEUXBZEnqDvLxFtVq5ZWYxVl7WbcQEB"
    "BRdqL0ikcm6kNPGXxfmre3G2lxrv268qyxjlU5MY2YFAICAm3cTOFMhOQ5t8FWqeb4dxfFW/7Jx8zdlUWNdc3Jb00xRGJUSIoZVcIiq1Uv4n/tW8zFyHHmZs"
    "T4wrLCqnY5HKIzHUUJEBCOYomqY5LRDxI3bQ2ubfdKestrWdvouLeqt7ur2squVXotqare3qhZFfY+N2V++Sosp8GB593gxyeedLdtoIAPiIiIqK9unvU1VR"
    "apRMLzS15xbXpttiS4zsSYcSdDnM7Ho7wbdwHTxU8tRLcJEJCVCEla5b7tlZub6l4WMzhtjIm+ZTFOHh57plRsRIjPy/N+ctot45ywu3Nwy2Tbe0nCMvWIkq"
    "lFqMcx9nqFRwfOHpH8lLc803KJu097HUZc6zYudOo8/MJeqqXYytZr30/i9qnSGm2Ju74E3yiKq7t38HCPaU5FkXFdxKYB2d+ld32OzrozMyZ5sdwRyISIOq"
    "I+YWmeo/UfnB63l9Dwy1Ta0ly7MxNU9I/u+Vdrr9zivGbPC7WYojM1zzj8Yfp5pzgON6V4DZdNsQhjHtlitrMKCz9ppsBAf33hXJ09uNPa7u3Duc0xRMRzzj"
    "nnwT7yzKKJ3q3x2zkslvZ6g0/vjSpeXcPMVyaVKkltxt5sXmXRqBDzEhLcJJUUvSqCAgIIVvpLYuUuNIca6JFRyKI7RIRIfEPL8PcW75y0VpTVmsICAonoId"
    "yGQ3Khyo4t7RkbHic27tlRLy1L7+xXpVqTFVZzp2eeQuY3qVxBcME5gWTwPWKVdbS0I+ErXf2Gr22dP/APLlzw+bRoR+BUnqvHR0+oSICAgICAgICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAg4hxPsz9YMfzHEmHMpxCmKOyDumplrbckVeuV2t82fKsbrHuVBKgnNaJ8z2mPsZEEN47toZNlXBLxE6q8NOlnCBmGo"
    "FpxXFMMLGPqov2G3c3bncwtFtaMG2QlwCYAfZZiMfjodDjseIRI6gIZXw28N/ErwxXjKtPYGoNuzvBchzt6+w7zmF4qzfLazNiG5cGunDgBGfIrmPXDl0vBO"
    "f3V3NALoeuE3hBvvBhw4YfhGllot9zzGFYrDacuO+53eHra6EcWm5j0Xri/sKgdY2gFloXC2AZND4gC/8Hei+sXDzhP1r8xZxuVajyDKL05crbdJBvi9cMgm"
    "T4zIsnGAagMaTShnv3UdHaImPjQesB4fs0vOvGomu2t0uFWdeon1MYIzYZj4PWXGQETPlJHYbUuTKqch0mtu0WogUIiY3oNdcVHAReNQR07jafYzEzq24lkV"
    "yuN1s+qeq19bN+kmCUYKtTqNTZHgKtC6XhD3/h98MR0z4Cdc9BI1inYxpJpRnMePhl0xdvB8myKWzbcZgSrvKmhHhyTt0g5Ufu8hmK91WWjdCEyXi8iCwY5w"
    "Q5NYeJHSjhnkXmt9wq2acWK9a1s+x7oQp11x51tuwELhV8RPSN5m0W6pNWSPQ/D6DC7aX8APFDbh020jz2PgA4Rpm7mLAXuLfZkmfeo93jTWozlYLkMGmKh3"
    "oOqFZB+IS2lyoO4IWLdnTxQYfpXp7Gt7OnNMh04wnB7XFtcO/TAhXWXj16KWZE93ESYGTH5Fu6R1B46jWhiPVIOgtD9NOJrDrRq5nmUw8It+ZZ5lPs3jdrh3"
    "SXPtsEgslvgMtSniYYcP3WFUjJsKeEvD9pBojXPgM4suJTKbpr5mjuNYxkYZJisy34TjebyjjXGJZAu3IJFyO3iTThyLr3hrbGMQOBGIvERbAp6Z9nprlohm"
    "mOa6YlpzheSXSM3lQ3jTzKc+lvwYsm9SIDh3KNOO3FQZBBDIJAUiALlJT1RLcRdUMT0+7LfiT0SyW8T7XpvoxqHb8l0LiYC5Byi7TYjVh53i+T3orDfcH+ra"
    "mmrnFjg1vB0whh4WtooJVOyP4hrXa79YmtWcdvfffc4t4vE6WEqWNNMWsS68igsHtdOc3V8qUM/cS3biPwIM8znswrjmuPXaHdrFgky4XfXHAsvlTJzBnV22"
    "WSFY4suOZlHIidMIE4Gg8hBIpQjDedBDL9KeBzOcW4m7tqtmF0sDmLRMwybJcNs1vfe6sW5XWJAh0kuBVoQbMGW7mPgIufsiZe+RINO2Hsmdd7Fgn1qoed4c"
    "zjJ6NPMPWdl2TtPUI8a+p72Q39Dwwe6biqW3qk6e/pbkGxMm4TuLKTdMg01x2BgVcQyHWHF85kZHOyOWE5tm3P2Z2RBGGMMg6tStRVB2r20hcGhCFeZCGC53"
    "2Q+cZLw0ae6JYZlGKYheLDgmZ2HJsjslHQdedvkQwImiFkDebOTUHHt9WyLlupzKlEEeydlZqXld/tuV6h4hjNvkRc0wmRdLfN1Al3uPcbVYZciRs2Vt0dsS"
    "HrbY4EBV2kYmYjtEQzOvAprlpxxE3HiJwvCtOMxinmuTzrPhmSXiRCYgR7xGtAFOZdGC+LUyjsCWBhRuom1PdKju6pAYa7097JvWvBdLWcevOLabXV+3Y5Aa"
    "tOP2fKLhbIsGfEzC635k4r4wSJmjLU5gWt7Rh1WiA2ul4iDKcN7OzihkDGvme5Hg4TeVn6rMF7YdBiZyxkB9Y4tujsyJFYjZAUgWWurIruIRoROoM04HuA7W"
    "DhM1gk57kua2rLImSWOdFvMi6T3DmY8VLtKmRYtq/Uo07i41K92YMgrR6ODu52pltDP+NHis0p0ZwPJ8BzO15q7cJ+HTXo7mP6a3u7xhE2XQpvkwYbrLRcxr"
    "zEzGo05EW0SpVaWv2tP5wyvfsavyl/O/w+/YIwr9ylv/AJuC/WvZP/kFn8o/s/Dvb/8A6s1f/nP95ZgvRT6MvGx0kUpEBAQEBAQEBRPpSicbMT0XK2jaSxu5"
    "lKkbZQlH7m31CHf4i3+H1vRyWN/96oc2x+7XJnryW1buGICAgICAgICAgKKvRW8FzygLSJQPYeR1BK1s1ke6Ee17b4x8Xl/BXHt+nLl6v1LYuS4QgICAgICA"
    "gICC5YiNrcyBkb1I6cbpu9RzqEHp6RbfEPzuS42r/dnK0cf8TzW1cn+Fx6/TEVEBAQEBRPRMdXWeiv2GMW/5ud/nT6+B9p/+eX/6P0X2Uqrns3YzHii6wZJc"
    "sZtdnkW3Sl/LaycgjR3I8cQr3ADE90stwl4Q2/heNdBX6MPR2LVE0TmWs+Mb/btZP3P0/nL6+tdgv3C5/R8X+kP95t/1aiX0P1vm8emupDafqLBwZH6v9lCo"
    "TPUL9Z6Q8i5eXzbvEuNT+8OdV+4OoexN/wB3da/3PXH/AMIV8/8ApM/6dr/OP/8AUPpX0Rf9VW/6/wBnd3acacawXKys6uYnnDNtx7GsVuVJEiRnVwsn1PXQ"
    "yYKJfQCGBeyJsiDoDFPzk7QB3dU1+dLUxL9d1NY5BxlXS9cJ2pMy86k5RbdQ8mkOBZRj2+dGgC+zcItprYbZK27O9i8bUSRIa27Zcl10CIGh2bVW1JnDzodI"
    "xWDhetmJcXV4zrRzCMPvFnvrmnN41EkTJNttBwiAiO5MOv1diTJLT36ljPkW+PUfM6QEmnwREzl0HwR4XrW3wlSrPnGQZDj06+3K8yMJbvz3frtjNnkyHStj"
    "Mg5G8nZDLJNFsd37fC0e7YSxqjEtY6PvZyTMwd0FvlkzbUW/ZXMsWrGZ2Zm+ZNcCkzZEaHkE+Kz1T8NPC00A7REQHbtERHaKXEUt+LOjmsKAQEHw/LVBRtpS"
    "Ct7JSh5O9MeoO3ltJWqVpV1SOiwpBBy12j/2Y+Ef+6gh/wCTWQKJ6Jjq65VFxAQEBAQEBAQEBAQEBAQEBAQam46tUfrKcGGquqjL4tybJp/dpFtoRberM7o4"
    "MZqnzjeJsKfOKiCbwxaTW7QXhuwDRG0U5xsQwu2WZktu0iGNEaY51+dXZzr98lozZU+Dz2QMCUIKssxzcGQQ15i5UhGgj8Hl3IrUnIs+FWgjUq19FKelRVVt"
    "pmpammaqopfhJ2n/ABC3fij4w73LxJw3oVunt45izbdeZH0zIKnT8N4nDH5tQXzLhNFPaXtndv187Vr4dP8AL6T9IGvudhfoxt8PsVbdVqfDr1/w/UjQnAbt"
    "pbo3jWnt+ySZeJ9qs7EedcrhKN5yS9QPGW8yKu3du2j6o8h9Vekv0xTq59l5vgGn1djhVui/PPGZn82TzNpM9Fzd7oWzwqKIh2V6dtWFVUuNYjFORVWEBAUx"
    "1RM4hTj7iGrxOcxMuY/g+qtL07ZxDHTxuicqizr6NqPSFCRAUVfgUvEhw6CPTb3VIuX4Pzlraj1yxuT6nplltlujbY8hFVrr7zkvRRFvm+qkU7V5neK1NWyU"
    "cojE9VEiKHWpVIzAy/C2f+1bxEXHHiJsT4xKsuPMYlyImJET0fCMQ81VaKcq1ThSKURN7o8cz8XL7X+MtItRLGbsxLxKckeDnGAqdSm33T4f4q3s0Rt9Jlev"
    "Xd3oKvWkAQCUPnu8xCQ+FY1W43ek1pu3dvoLXmUjL3MXnt6ejbxvfRKlv9mCMIwu18IkewSqQj5to08Xl8KvZiLZcmblKBpLiV2w3ExtN8gxWpxyjfmSI9yO"
    "UUx4/Ebzhm0146lz8O3aI7RHw+Ea6ivejTUbZZOuNX1cqMR1eHnumQti3WtTL/7yWtu3lhdu45QMs9EdpOVMq+kiJTXXgot5e1jnLaYikU5wjPdz+Ci5tjvd"
    "4Mq8j2gXxR+KS2oiKurCunuq+8jorLC51bbu7j82N6wanY9oxpffdVMseoECx212W9uLbvIR8AD84y2gPziotbdmbkTEdXH4jxOzwmxF6vp63PvYqaO3/VHJ"
    "8x7QjVOzBW65XNfh4++81y2tb90lxv7QcxBmhfFaMf28ez3DOL6PXXKtTV9n1KdoeM9luNaLT6jh1rbcmJ3z+XKOr9Dl7KIjMTT6nipppqs7aauYrUrVLdYS"
    "jxXJNgZZMBhuU6fULduA/FQh+9u3D+9SopXFVBAQEFuuQRYt6hXRwXd5kcTwl6Np+Lxfvg8P4S0VpXFZrCAgKJ6CDkkVmVY3wd6u1seqPR8+4C3jt3et6Fel"
    "WpLjvNyo4SG6eF0aGP4NVVZz5ZRhaf8AaxTmW+TNNTtCmX6jy5C7KsN2IDL5xkze2h/Bj/NVJ6rx0dNKEiAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICDivWXKtdcHuPFprno3qPWHkOCtRitoZNJn3K1wrfGxaHcXGo1t70EZuQ7IrX3fZzp1SqXVpTpkGb5Vx/ZjjUi4X+ycNtxvuE4/l1qxW/ZczkkZh4Lr"
    "MlQ45k3CcHc5GZOa3Q3d4nuExBohHeg15r7xmat47pbmWVU4g8b07Zz3W4ME0dyTIKW9uPYbfCAWLrdXKyqdN93rQbw60D24C2RR2+PxBcNL+0P1b1lx7hry"
    "HRzELLlEPUS73Wy6kTGb4yyEW5W2FKGYMetBKhgMmI86Bj4XWgDb6HRJBMwDtO9WdU3MMiYDwUXKQ7qZZ7ncNPe/Z5CYblBbXQCZWYXTIoYe6hVogF4j3UoQ"
    "NIL3C7SO9Zrhf1zNG+Ge8ZFj9jwu3ZLnsh/IokKRaI8qJ3ykWMyW7vsoI9KmQb2mvEAi6RFWghWgdorNyzV/ItKME0YjvHb7O9Lxsr5mUe2zb4QwRmA+zCfC"
    "hnBMD2UlMk9UTEqG0FBIhDRWi/azam6T8NWlly4hccnZjqJqRhbeczG5VyhQodutEgWuk1HOFB8RGZH0WHQI9on1pXhEiDqXQDjBvXEhn8m0afaG3SLilvt9"
    "tk3HKb9cmojrRz7TFuUaOELaTpOiEsAdodQoBctpH4hANG6jaucV2G6UcRPGHZuKueUbR/JMjes+nN8x+zUsUqBaogSO6OPtwwnhV2m8Ot3kqiRiW0xHYQZJ"
    "l3avwrRUGrLw/wA4QDMp9huF2yjJo9ntUI4kG2zCF6c6BNsSXqXKgR2HdgulFf8AdQ2juDemH6mlxD2TPsUwu8XbEZuOX88fHILfWJJfaerb4cykpgX2n2C5"
    "DMEaC4Bjubr6K0qg0twjar6vYPgdjyvXjiFy/UuVnGr19wm1s3i02KEzaqQLreWGX6ex0CKR1Nm3AJ76n4y5iIeJBLvfHLrDkvE7h2lWkWi8eXj72o2QYrlk"
    "6435plyp26ILpPMjsKu2lD38vMWzZ4ee5Brjio7Ve9j2dNq4hdDbHcMYyTUnQa/Zzh1wkViyisbsCNDdoDrboG08X6sD3xqPgLmKC78RPaEa6XCDi+QcPelE"
    "22YPN4kcZwGTqDJucJx2bT6q49ouzVLe62RtxTIJcUZG8Xt/jEAHa6gzinag6YwOH3T3iLvund8atef4Hc8oj2+K809Jitw2mD7tyqQi464UgAGvMR3eYqUQ"
    "YnpDxear8PebP6HcTWnuaSshvN7x6Rbpt8zC2XNqNDvF2as8aO05Dixv6nOm92htkZVIi6ru7cIbJ1G44Mgser0rQXSrQWZleUN5RIscNty/sQIhOM2WBdjf"
    "eeMTq010pwteEDPeHl2luEMM9sozHUzC5UjQfhmvV1vFlwV2/wCdQ5GQQo72OfqifDGOxv3N3CR3i2zNobmmiBoCJweoIoMOwrtf7BhMzSDTjVrHvZubmFnw"
    "235JlVruIE9Gvl8Zhix1IbDHRaAnpbJGJPNOiDm8GDapQkG/uITUriKi8J83NtONLrxZM+k3CBFj4/DrBuU2K07dWY7zrfNzuzp0iG4+NCPbT1vLVBqO8cT+"
    "XWnhpzLPHuJXMLbkmmGSPWbLrDfNPbWd6kXeQzDdt9pbYj/qd03qS41WTZIxd76HM/CW0M017yvjdwDgywmmNMvXjVm4XfGoGZTsRs0R6sYHpTHsq9FjyjFg"
    "tjXXAKmW0edDLbShcgvWAcV+J4Pwh5dr3qtl+Q3CmmTF5LOq360RIl0jSIHUdeimxD9wJ3Zso30akLouNEJFvoVQ85llWsOnPBLbNU9Y9c64jkWOYszeM7vb"
    "WPRJdXnhjVN+I2wQgG6rxUAKBTcRCA051NBqrSPUjj21dzy1cMuouscDAs1sOj9vzbLLtasWiTO9TLrcbixFtxNP7m+nEagVGQbWwnXHR2G1T3w88JGs2X62"
    "cVulOsF0ulzgPajcNF0vOUY7GySbIsvslFutljhIixHXSZY8L0jaYAJmDvjIq+lB1Prv9gzM/wBydx/mri0tftafzhle/Y1flL+ZPh9+wRhX7lLf/NwX627J"
    "f8isfk/Dvb//AKs1f/nP95ZgvQ0/s3jY6SKUiAgICAgICApq9KSPQld7S823it3ZK2uuEZRtsgWxIWNpl5i9XmuPf/e6HMsfu1z+i0LdwxAQEBAQEBAQEBRX"
    "6UreC8ZhIZkOW2rNrci7LOwBdRmgdWtB84/GEvjLi6b0ZcvV+pZ1y3CEBAQEBAQEBAQXXCXm4+SMPvW12VQW3f1Oy2J1L3IvgL4vm/erj6v92cvR/vS1Lkfw"
    "uNV6YiogICAgKJ6Jjq6z0V+wxi3/ADc7/On18D7T/wDPL/8AR+i+ysXI7N2Mz4vOqcXVyVb7UOj9ys8WSF8YO8FemzIDt/i6wN7RL3Xybd3h8y6Cv0YeksVW"
    "4tzmGq+Mb/btZP3P0/nL6+tdgv3C5/R8X+kT95t/1aiX0P1vm0emvBSG/qBCLS1u7/Zgz750x2FTpD4N3v7vW2rjU/vDsK/3B092Jv8Au7rX+564/wDhCvn/"
    "ANJn/Ttf5x//AKh9I+iL/qq3/X+z9hsywPEdQYsG35pYmrhHt90jXKLHeItgy2D3sukIlyPYdKHQS3U3CJeYRX5vicP13tawm9n5wmXK7ZLdbppg7JHKxk+y"
    "lvkX6cUNg5MgJUl6LG6/SgPOyWgkG7HFoydAT3b/ABLXvaoNq0XfsyODvIsKPBcmwrI7k09k0PIJV2lajX32VkXCGBhEdcuIzBlmLIuH0mid6QFyMQ3iJKJv"
    "ybWbaecJ2iumF2sd+xe35C9Nxv2QpZ5l/wA6vF1cYpNoxSSJFNlOk7QqR2tonuFvbXZs3nur3tUm1lWnmmOD6U2mdY8AsfcItyv1xvM5vvTrvVnT5bsyU7zd"
    "IiHe884e0fCO7aIiIiKrOVl+QEBAQC96v7SCPaRJu2xxJ6jhC3T3QS50JWqVpSFVYQEHLXaP/Zj4R/7qCH/k1kCieiY6uuVRcQEBAQEBAQEBAQEBAQEBAQEH"
    "Kva8Qp+ZcOOHaIwBqVdQ9dMGscoQ9+sMb9Fmy/73dob3P5u5B0stKGaBbXGZF6uEhmYThNk0wTe3lQCoO79951eegnqvXEIroiqaaoaF7SbiO/QzcI2TZrbL"
    "x3S93GP7F44YlyOkx8SETD5wBvd/eLy3aviscJ4TciPSnlH9XuOw/BI432it0RGaY51flHNxB2cPBPw551h+Ja+XrKG71mWPXZyfMtsG7A61CdqRCw1JZ8RC"
    "6Oyjo+XxfGEV0PZOxe4Nwyd8c7vN2f0iRwLtj2qi5bqzGnxER6s4d4rtsbqJmfW4kTM43KTnUKU2AkO0RKpD633ltb5UMrk5qp/BVWWc1NpztkUAgItSpyyc"
    "Fkum3zrXbTb+2tqXGu9VQBERoIj6KehZT1bR0FCRAQFNEZRu20qLPTeerKEiIR9zH7XhLxLWqcQwt05qyrLFyBAQEiZhExmFHb3PyiZgbn7ez/8AUuTEd9yj"
    "q40z9V5z0fSeJ4jZjltqP9cIfQKpNuKOcdF4ud/y9b0MdsSo4Q7zoPLqF5lWbsepaLcw9rOebSIw8SN3g2kI+6CtrPosr3V7WdXpNqfRfHGweHpuN0KheqSU"
    "TtVqjKn0nGS3NluCg/re3/6FrVXE8mfdzD53mhCBMt1MjLlt27dv7fxVMWu75s5v97OIe2WRZAvFWpVLmREqVVbmtFvEPayaiAgVpuHaphE9FOKRE303HKEY"
    "egtq1vxTbxcnnEs7PdTRFfWr1w4g7VLNsq1y1MwPgB0q6ztyyi8MSL442JEABU9rIHt9UR6rx7vKIAX4PL4JruGxdrqqnNynpH5vDdu+E8Z4nTYtW6JizXnd"
    "V4Y6P0j0W0lxPQnSjH9HcFj1atWO2pmFD6nnOgDtqZ/GMy3EXziqu7t0znLK3bi1biiPUyhaVT4Limeogyu+M3yM8MoKRnWTacbIuRE54SDl8bw0NWp6K1Jy"
    "pPVYQEBBDyEZtLQ45b3gB5shcEnCpQeVCEq7ql5fDzVqSpMEhIRKhc6VSooFUEBB8Km4aj8ZBFsJSitMcZ2zrAO1zp7du6nh9X0K89FaXN/GmzJw/jM4UdZo"
    "pELQalXjELlt9Zi64/NMN3ze8W6N++2rKpZ1eqtBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBp1nhNt0/P8AVy75Zmsi6YprDbmo1+w9yCDV"
    "GCpbGLa6YSQLqcjjs0pt9UiqVKoNKa19mln+qnEDJyWBqDaLVgtzziwZNcLVDuV5ZcckWs4Z06kAJncJUgzhs070bY7R2c2jNpp0Q6CxbhS0OxG/YbktrxAi"
    "mYDY7hasXOVNdeGGzOcZOWe0yKhvOVYDc6XM+RHTd4y5hHvvB3oPfMgDKvqcn265M5x9VrEyx32XANq6lCCE66NY7gcgdjhsda8ju8yMSIyqglYFwp6Iaalh"
    "LuF4iUUtPLXcbdie+a6fc4842jkh4i8e+rLfpLnt2+hBh9+7OThfvtvhWRux5HbLaxjMXHbla7DmtygRr3ao27oxLg2w+IzAETcD3XcRA6YEVQMhQXRzgb4f"
    "ZGsETWSZaL3Jn265SLnabTIyiadqt09+E5CemR4FXe7svFHddDcIf10ypShFUkEe+cAvDvdMZw/FLNEybHaYLi1MYx+4YjmFwtUtuz0Boe4OPxXgN9n3FqvI"
    "6lUSHcNRLnVBsLTnR7ANJjuh4BYqwfZh+K7cB7y451KxoUeEz5yrt2x4zIff2bq+KtaoNd3Ts/eG2+57c86yK15LcG7vkns/ccXl5tczsMq48wKkh22dfujp"
    "bmwPabZDuAS27hpVBiHFT2b2Eay4zdntI8mv2FZRfcimXK7XyxZdNgd8pNZisTWXxa30fYdCHF3McgrXo+5OxzLq0DeGlOi2nOilsuFm00xsbXFuc9qZKjMO"
    "nUOqESPDDbQirsEWYjAbaejwc/hqgx3KOEnRbK9LGtIXrXcoNsi5VIyOBKs98lQ5sG6vTn57kpmS0YutHV+S9X0Ft2ukHLZXagsmJ8A/Ddg1ts1txSwX6E7Y"
    "c0kZZDurOX3Ck1y6yQ2S33pPX6kikilS6zbpEDu6u4aoMVc7JzgqlWmTjtywa9yrS5jd1x6BZJWZXJ2DabVcCApUKEwT+yK0VWg5C1QdtAER5CI0oF8yHs5O"
    "FnKdQWtQrnjl+AmM8h5qzYImXXBmzhkMaU3KC50gA8Mfrk83Qzrs5OERkVKkZFULBduy14Zo2J3ez4PYpbUh/Fb3ZMZgZNe7jdbNYY9zH9UMMW85QttxyOgE"
    "TTdWy2gIgYUEeQYXoZ2Wb1LtleScV+aPZHOv1gsVogFaczv0mTbvYyYc5iWzPnyjksOjJJpxoWiHpVa3VN0zM0G8dKuC3QfR3JG8xxKy3aRexvUy7vXq+ZJN"
    "uEqROlRGIj7zjsl0yOpMxmQ5VrtEW6UGg0QWHK+zk4W8tjDBexu+2+O7bZlsu0ex5dcIIXm3yZb8x6FNow8Peo/WkyCoB7tveHRHkLh0qEF3swOEJ272u4N4"
    "pfo8S0Xqw3iLY4uX3Fu2ncrMEULdNdii/wBJ15oIMYN5iW8Wh3bq+JBsCfwv6TXTQ4eHi4MXuVjzbwvx3pOUTnbixICX3xp8JxvVki63IoJgfU5hsGlPCPJB"
    "a8P4KtAcIi2xm349cJj9vziuXyLhd71JmSrne+6lFCbMeeMjlGDNQEKHWot9BnYI9INoS9VOE3R3WK4XS9Zwxfjm3Ny1vhMg5VPjHbZNuN84sqF0naUhvCUh"
    "yhGzQCdEtrm8fQgrY/wqaLWDSY9FWselTbHKvgXe60u11flyLpOGWEwn5T7pk5II3mw37yrQhHZy2eFBfNVdGdPNbbZabHqdYK3OBZsig3yLBceKjJzYbvXj"
    "G4A15O0aeEHaAXMd7QFy8NEGO658JmkXERdomRZzS/wbjEtcm1ldMVyibaJMm3vkNXoTzsN1s3WSIBLYVfCVOY7a1qgtmG8I1lwPiPtGtmO5TSFYcZ0zdwrE"
    "sEgWZhiJaYbsiE8ZA6PjP0wGREK+EaVqgzfXf7BmZ/uTuP8ANXFpa/a0/nDO9+xq/KX8yfD79gjCv3KW/wDm4L9bdkv+RWPyfhzt/wD9Wav/AM5/vLMF6Cn9"
    "m8bHSRWSICAgICAgICmr0pI9CV4s/spXEbz3Ume7bo3eupz3+ctmz4PNu3LjXv3uhzLH7tc/os65DhiAgICAgICAgICiv0pW8F5zP2U6ls9lSYIvYWP3fu4l"
    "+tbS27t3rfGXF03oy5er9SzLluEICAgICAgICAgu+C+yX1UR/YcmRf6b23vG7Zt6R8/L83cuPq/3Zy9J+8rQt/4XHq9MUqCAgICAonomOrrPRX7DGLf83O/z"
    "p9fA+0//ADy//R+ieykW47N2MT4rbrrYcFv1nsTOeakT8ZZYyaK/BkW+7DFrNl0E9kU6kJbwPcW4PW2/NXQV+jD0+nm53c4hrvjG/wBu1k/c/T+cvr612C/c"
    "K/6Pi30ifvNv+rUS+h+t82j0l5L2T+t03uKP3H2aPb4S6nV6Q/vdu1can9u7Cuf+AdO9ib/u7rX+564/+EK+f/Sb/wBO1/nH94fSPoi/6qt/1/s/Z5fm9+vR"
    "Tsj1go3UpzIrRNJmRVQICAgIBe9X9pBHs/R9i4/dN/T6I7ep7/JWqVpSFVYQEHLXaP8A2Y+Ef+6gh/5NZAonomOrrlUXEBAQEBAQEBAQEBAQEBAQEBBznxez"
    "Yt24seGXAnBEzLPr5fibKnOhNw8cuLHPl812eyX4Qig3wtKGaDYQMWHpDkIGTemOmQiNR3ci20Ku74wiKvPQ6pypXOJiPApzTVFD8ke3G4kIeoOv8DRO2c37"
    "Np7bymXlsXNoOznhEtm77wdEN3q1dMV8s49M8e7W2dB/DR1/u+ycGvR2C+j7VcduR9q5GKfHw/8Afwbp7J3QmZpDwtxsvyKOIXnOZRXmd4eVRZLwxwL954/m"
    "9Wq9zxObO2LVH8D472D0Vy1oatdqJma7szM5/N06upzmjD3sxO/l6MqTZNlKd27uYDQS/wAZcieVDCOdUqq41E5qlyq8bafxFKogItSpvbSkNtb9peI9vxhp"
    "/wD9W9LjXeqosJ6t46CAgIPEhyrbNSb279vh3fG9VXsxljqKttWHpsem2I/FU3ZwvTH2cvqzXEBAQ6KLxdZysNsiGpDzIh9UVvbnZyno49yO/wCnUbb7jybZ"
    "brVsi+N5P/arV195OZ6q27fccoVlxZomJ5uTFWYFPRKlJ27Q57v1wVrZ9Bje6qqzq9JtT6IoHh9wm+Qtt76kXLb8VaUUZ5s668clMo/dSrMbI6l6eoPm3rbv"
    "O8cbue6nKq24Lg0cbLcJekSWFcbJcq3Xuh6VFxAQEFh1EzzG9KcPu+omXShjWu1W16ZOkEXlBsd238IvKPxi5CuVboquaaKKecx/7dbrNXp+FxVejpjMy5n7"
    "GfTjL+IbWDOO0O1XYoTk+4v27FhcHmQc9vUqHxRaa6TNCH54+r4lns7p9FxurVUzmuccv6Os0f0h6/tJ2Wp0s29tMTPPlnq/R5emdIKduQSeogZM4zFtvslI"
    "jm73R4HRFsuReEtpF/FIlanorUnqs9VhQCAg8SGW5EdyO8O4HG6gQ/GoStSVo+Pyo82yxJEXfsrHptFwtxDyHbtr85KihLVQQEBBbsfZixO+wYu/3O4OG51P"
    "jHtdLl83xq89FaWke0aKDadG8Pz2dQQ+pjWzBrj3j/7Ha+qKBHkH/wBnkPj+CVVlUs6TVWggICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgICAgICAgICAgIMT13+wZmf7k7j/NXFairZXFXgrXTvomnxfjhwO9iLq9rHwXaSatWrWjHIcbKdNLFdo8WRBkEbASIDLogRD6Nwie"
    "1fZuD/SZpOHcOpszZmcREZz/ALvz72h+iXWcZ7QXdVF/EVTM4x4820af6n81v5/Z4xn6Pk/5l2UfSzpJsbe4n4x/l0EfQlrat0d/8n39L+a3fLzjH0fJ/wAy"
    "mPpZ0eP3efjH+VvMjrff/I/S/mt3y84x9Hyf8yednR/d5+Mf5PMjrff/ACP0v5rd8vOMfR8n/MnnZ0f3efjH+TzI633/AMj9L+a3fLzjH0fJ/wAyednR/d5+"
    "Mf5PMjrff/I/S/mt3y84x9Hyf8yednR/d5+Mf5PMjrff/I/S/mt3y84x9Hyf8yednR/d5+Mf5PMjrff/ACP0v5rd8vOMfR8n/MnnZ0f3efjH+TzI633/AMj9"
    "L+a3fLzjH0fJ/wAyednR/d5+Mf5PMjrff/J8/S/mt3y84z9Hyf8AMon6WdJP/Yn4x/lHmQ1sT+3+Suz2CWtMWE9bC18sO6ZtqPThyOn4PT4/D9/0KnnV0c8+"
    "4n4/7t6PoV1tMft/ko/pfzW75ecY+j5P+ZaednR/d5+Mf5ZT9COt9/8AI/S/mt3y84x9Hyf8yednR/d5+Mf5R5kdb7/5H6X81u+XnGPo+T/mTzs6P7vPxj/J"
    "5kdb7/5H6X81u+XnGPo+T/mTzs6P7vPxj/J5kdb7/wCR+l/Nbvl5xj6Pk/5k87Oj+7z8Y/yeZHW+/wDkfpfzW75ecY+j5P8AmTzs6P7vPxj/ACeZHW+/+R+l"
    "/Nbvl5xj6Pk/5k87Oj+7z8Y/yeZHW+/+R+l/Nbvl5xj6Pk/5k87Oj+7z8Y/yeZHW+/8AkfpfzW75ecY+j5P+ZPOzo/u8/GP8nmR1vv8A5H6X81u+XnGPo+T/"
    "AJk87Oj+7z8Y/wAnmR1vv/k+fpfzW/5ecZ+j5P8AmVavpY0k/wDYn4x/laj6EtbH/f8AkrzOwT1puxNE3r7YS7uwMcu9Q5FfEHxPR5PtKI+lbRx/2J+P+61z"
    "6E9bP/f+Sj+l/Nbvl5xj6Pk/5lfzs6P7vPxj/LPzI633/wAj9L+a3fLzjH0fJ/zJ52dH93n4x/k8yOt9/wDI/S/mt3y84x9Hyf8AMnnZ0f3efjH+TzI633/y"
    "P0v5rd8vOMfR8n/MnnZ0f3efjH+TzI633/yP0v5rd8vOMfR8n/MnnZ0f3efjH+TzI633/wAj9L+a3fLzjH0fJ/zJ52dH93n4x/k8yOt9/wDI/S/mt3y84x9H"
    "yf8AMnnZ0f3efjH+TzI633/yP0v5rd8vOMfR8n/MnnZ0f3efjH+TzI633/yP0v5rd8vOMfR8n/MnnZ0f3efjH+TzI633/wAj9L+a3fLzjH0fJ/zJ52dH93n4"
    "x/k8yOt9/wDJWgdglrTZpQ3F3XuwjQRId0SC/Rz0jt9HOn3/AE/N5qJ+lfR1TjuJ+Mf5Xo+hPXU//uPkofpfzW75ecZ+j5P+ZRH0saOP+xPxj/JX9Cetn/v/"
    "ACff0v5rd8vOMfR8n/MrednR/d5+Mf5U8yOt9/8AI/S/mt3y84x9Hyf8yednR/d5+Mf5PMjrff8AyP0v5rd8vOMfR8n/ADJ52dH93n4x/k8yOt9/8j9L+a3f"
    "LzjH0fJ/zJ52dH93n4x/k8yOt9/8j9L+a3fLzjH0fJ/zJ52dH93n4x/k8yOt9/8AI/S/mt3y84x9Hyf8yT9LOjmP3efjH+Ux9COtz+3+TauGdl5qLg2HW3B5"
    "mp1mfetUc2nHm4ro0MidN3dTd+Gvk/HO3Oi1PFLlcW8dH13gn0dajRcHsWe86Z9TXfFpwb2PSXG8SuWrUB3LI12zy3Wu0x7HjZzzgT3hd6Mt0SIek0Gw9zvi"
    "27x8PiXEt9qLNde6KXaVdjNXp4zFzr+C96u9jDqtrTeoV9tOsWPwwgW+kNxt+E8VSr1Tc3U5f8YP8Ve37J/SRo9BZnNmZ/q+f9rfol1nEa90X/kxL9L+a31/"
    "/PzjP0fJ/wAy9V52tHNzPcT8Y/y8ZH0H62KP3j5K9ewU1pK3DYPr+WHdR7vGzucjo8tvL3uXPf8A+iin6VNPFM19zPxj/LkT9COr7mI+sdPwbi4B+ya1M4Re"
    "IiLrPlGqtkusWNbJUUokCI8DhVcDlSvM/R6F5rtZ2/scf4VOni1MTy+UvVdiPo01nZ/jsa2q/mI/D8Hdq+YdX2wQEBAQEBAQEHw/LVBQtJE5bY7hsi1zbp7m"
    "I7dqtUrSkKqwgIOWu0f+zHwj/wB1BD/yayBRPRMdXXKouICAgICAgICAgIjMCZMYESICAq7gTcFOVaehTHRGYco60THcl7YDRPFxLwY1olnN6cHn5qybhj8M"
    "C/e0E/49Upn7asumlvSitBx3pFZY7jMwpAODVwXiHbu5lu/+slUlCya3ap2HRHSPI9WMldoMKwWl6Y7Qi5byAfQA/OrXlQfv1our4rrqNDwy5qrnSmOTtODc"
    "Pr4rxa1p7f8AFL8mez7zDR7WHXTJo/ExjQ5Hl+q9X5MKNKs9JMXu4um+9UvN0uZtV2+jwix5h8O7532T01/V0XOLXeszyl9P+kvjnZ+3rNL2Vx6Ec6f6Zh+k"
    "Ntt8Gz29i02uKEeNEZBqOy2PKgAI7RGg/FERXqZmKpzLy1q3FqmKaYxEKypiMtpqnCnH6m5wnCpX3Tw/NFb3ejj2OaouPR1cijORSCAoj01p9FS27pRETfvN"
    "jtL/AO/+8uTc6OPT6aquO2EBAQUZHTJ9lgufiLn/AALa1GKXGu86lZY1c6nIp5UiJEBRMZI5PLjgMt1ccLlQR5ktKbWVK7mHyO24237o5vqXpJTd8FLUYe1l"
    "RMxOW1WJURbKGQtst7myLxeLyf8AtXJ+zfpzPVxZmqxViOit5vSNFhid2J6OTG3GY6vEjdTZtL+uDuWtr0ZUuxze1lVGamscqXh5xwR2sDvP4u7y/OJaW6dr"
    "jV17hlltnmQ+mplzIiS5Xha3amer2qUziWtylTbIm5BsuP7t/jbH4tPWV55sM7ZVFk5AgItM5EnlBTDivtbNScszccS4LNLRF285tco7tybFzbXu/V2MtV+K"
    "JO7jIi8osfOXH03HLOi19u3VGaq8uv7RdjuM9o+z97uLkU00Y3dPF+g3DlofifDhojjOieFRQbg2C1txycEfTIe27nni+cbpGZfOJe2xVEb6ucvEafT2NBYi"
    "1RHo9fxZsrNRAQUpzbr0F5ll7pGbJUFz7VdvmQeba4T1tYeJ4XCNkKk42XMS8PmFWqVpV1SOiwpBASgQcd78MB0LhIBxwJj1BJshr4OqWwfD623arz0E5UBA"
    "QEESDSSNznC8IdInANkh28y8AjXn++p6y0Vpc29s4NwY7LXW/ILMXKZYcHkXuGXxXoBBMAv40cVms6ns9xjXi0xLxDr7lKjg+3+CY7qf/Ss2iUgIFa0p6aoj"
    "GREiBWtKemqBStK+miAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgxPXf7BmZ/uTuP81cQaW7Jr/etuG/"
    "/oJxT/yeKr7s/ZiGU/qqt0ugVMxNEYwvVi3G7ApxPgjMCYnwMx4CYnwMx4CYnwMx4CYnwMx4CYnwMx4CYnwMx4CYnwMx4CrEc+iYqifUoyBkd+YJtzkHj6g7"
    "vN4fQrROPUrVMR6lZMT4JzAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAq0zM+pMzEepHt4yB63eHN/6oPb4t22nq0V"
    "pjHqV3xPqSExPgnMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeAmJ8DMeChchdKGfd3dh+HaW7b6yimmZ9Rn8FdKqZj1GfGBT"
    "ifAzHgJifAzHgJifAzHgJifAzHgJifAzHgKJifBMTGWJ3n/ZmX/xg/4orymszVeuVT+D0Wj7yqmmM8oa04i73xK2Gx469ww4Tjt9nyMwhsZQzkko2Qi2Son3"
    "mQztcDc8JUb2D4h8VfCSmxXa7uZmpa5cm5imW4MP/qKR/wAo/wDQV2vC5jbiIdZrqZ6yu67TZic4dZ3kTGMI4jI9lCLre5d3/W93rbvNyUzTRFqKVcVRVzSK"
    "8/gqoxbtVRT4r3Ij0IFKOggICAgICAgIPh+WqClaxeC3sjIc3u9Mdxbue4v21apWlWVVhAQctdo/9mPhH/uoIf8Ak1kCieiY6uuVRcQEBAQEBAQECteVOfJR"
    "PQc/cY161nsN3xqxaG6+3Ky5XmV6j2nHccraLdIh0oG52bcHhfjG+TTMYHDKguhQiFoKEJO03a0REspllPGzl+oeAcJWeZXpXlvsHk0PHna2S91hNSO5ySqI"
    "A90nRID21LntIa0UW6c1LTVhz/lXGFxI5JYdDcdxm4w8eygsutga5DHiMyGwBm9x7FMtzPVEtgyJz7xg7TkXShntLmXNcibXKZZTXzZw/wBqHpHZ85v2LZTj"
    "b0eFabZeZke4229wp/evY0ak81sYcrtMxEtgiR+Kmw6tn4Fn3PJffKDhHFTkvD9EyPCNbMKzb2VhY83lEGbm2WWl1k4bkpiAzCCW10goYGQ7xMSOpH6DfJ0N"
    "zusyb5V9Pu05x/VnIrNp5propeLtlN0yC7WiVbW7tFCLAet8SBMddOSReJoo9xj1Gohv37gqAkNUmziOcm+WT6X8fOC6l8QVNAm8Zow/LlXKPablFyGFNCQ5"
    "BIutQxjuFs5iBlTbU9u2oO9I/AqVWuWTfLD3eKV+PxzZLZtQdb7rien+OMjFsr0qFAjWK7T40EpF0gvyZDRPd4ZB6NIEwcaGoMvAO7pPK8UYoRE5epjI3bto"
    "6XIa0qFn4YOm2X/LMiEi/mYLKmn7Szph5wm2TcEeZCNa7VpSVIuNCTePQQKHRivdQ3Rx3eDw+XxelKijm4M7cXWufcLBiHB3gss3LxmVzalXOIzurXuwO7GA"
    "qI+kt73p5f8AyK/NXzXt5rbt6LegtxP255vsn0V8N09ijUcVuTGLVM7fzmJ9Xw+bVfZG6ATIedZxxAZJidxttI7g43jMe9QyZkhGaECdOoF5TIaMiRD63VEf"
    "WXqqbVnhnBLegtRyxzl8U4P9c7Rdr9ZxnW0zumfs5/Dk71bsd6ebo8zapJhUeYkLJciXAjT1zD6BXrLVHKZh9+p+/f2Glf8AZyV40teUfW7WOcwidwlW8enK"
    "hvM1Iq12vDyr5la9yNLVFUZgXGo6uR0kUggKI9NafQUWdveHi6m7xUpt+L4VybnRx6fTVlx2wgICCn7pWUPipsFvy/fW8cqHHq51Kiwj0pcifREBAQU5Q7tj"
    "dG+dDc8X+N/6Leirk49dPNUWGct4jAkckinM7sx0RyxieqiIlDOgttj0fTu8Xk/9q3+zfpxHVxoiqxVmXqQTZCBHu5dSm3aoo5UyvXXmR94h9yZASd+Lu8vz"
    "lFEc+ZXXy5PrbLbZEQj4jLmRKKqsLRbe1jPNtE7RSjOVKSO0mnhb5lRzl+CJLeiN0OPXSqrCerkR0EBFopW7LMqsWD4vcMyyidSLb7VDclzpBeUGwEiIv4tF"
    "amndVEOLqb/cWqq/CJlxz2VWnl24u+JLUvtCNSLfIegQ5D1vxO2vEW0XiaEfQXq9GP0w9HrP1L1V6HyRwudRb5Zrp9f5vmvAO03H+IW9VdqrmLN2cRH/AIy/"
    "TyMZOsNu1bqNTbGuzkuXGLVUznLsqaK6o3xzz63uonT3260/borZhO2rwfSEh81OSIfEHwh3DtL1kEHF3mXrDG7uybTQDVsW3C5kIgRD5v7ytUrSnqkdFhSC"
    "AlAg2UYzUiezFExKk6tXNxeuQAXo+b6VeegnKgICAgt0OkMconttk71jixzcEvJt3GNNv+F/3LRWlpvtRrD9VHZocQmPj55eiOVNt/h+xErl/hclnKz1kbmb"
    "5dwX4LlOC3rN63IMQttxGxae3C3RLned0Jr3IHJ/IBEKnR0qUMalQNvj3bCrRz6pr5M34QtS73qrwo4FqdmGWw7vdbpicORfLrGilHadmdIaSK9MxCrVaO0M"
    "SGohtIa+EfeorjE8iicuVMD15ybgjsGRZjrvXPb9nDmnFxyIHJWoblzxXPyiGwb022mfW9iTEXg5RwZYDpPeEHhaEx3m3ExyZRXzdBYNxWalakZFecqxXSGz"
    "fW1x7Lbtj93ya5Zh3a4MO2156PMl90KN0u7BIjuB6X95BTq7eXhWc2+eGsVZaHu3af3LUHF3a5Xh1wwaJBuuK5JFyKB7NNszbCWR21mTyrMtkM3wNl7b7jR0"
    "HQMxr8Anr3CneLhnnHpmjma6ecSFuxiXGxSVpbnt4g4m1lZcruzFm461GfmgDXTjyQ60kelWjxMdUxqQlUwGIsxmYO8ZtwkcW9zzzXTL+Gq3x5l9uVlzvKJe"
    "QXe93V1pu2QG7q61GixN7Z97MNzIk0BA2w0YbjoRA2VK6IiMpor5sfzHiGw6z8dV91H1Uy/MbDgODE3jMPIwyZ2JYYt+bglcZkSZCbc2PhIizolWpDzda9aH"
    "VoCGpB1VNEzBXVzdjRpLMqOEmOW4HBoQFy5c6VWLVVQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBBieu/2D"
    "Mz/cncf5q4g0t2TX+9bcN/8A0E4p/wCTxVoz6ugUBAQEBAQEBAQEEWULJ3KMTjnI6b+mO3zeFWpEpVBAQEBAQEBAQEBBGtYsCMjovb90o6l4du0vioJKAgIC"
    "AgICAgICCPdhZK3mMhzYG4dxbd3rClAkJWCAgICAgJIxO8/7My/+MH/FFeT1c51k4ej0tVE6OMQ1dxNad55qRYcZg4DxDStOXrbmkG4XCdDZAyu8RsXd9sLe"
    "Q7Re503EPi8HlTTXtPRExXTla/TcnGJbmw/+opH/ACj/ANBXccJnNLreIruu0/jdZT6KKIs+zROdb3Xuo06e31d3mVqhKVQQEBAQEBAQEBAL3q/tIItnFlu1"
    "xxjObwFkdpbeW5WqVpSlVYQEHLXaP/Zj4R/7qCH/AJNZAonomOrrlUXEBAQEBAQEBARWlg+V4npFjGqVu19zWRHjX2kBrF7NcLjK5AyMyW1+p2RL0C7IepHG"
    "vLxH0mqeqrRmSp8vGhWLZNpRd9Hspvt9ulsvb8lyY/crsb0kRekE/wBIHS9IgFS2APqgIj8CiK5zlNWJY8zwaaGQ87yvUSJZprd1zHJLNe708NwKtO82t8JE"
    "YWhL0NNdYOqYD5zdOpeZX72qIwrFMOftWMT7OXSLM5mjeo3FTFav0C1XO3Y9p/meY17laGrrEqJxgjNUBzpGDwFQqmToDy2GNPQr0TKlUZUdFeFZvUfV/ILP"
    "xD6+M3LKY1iZtVshWHUCRdbpbHLfNhXLvNHnocUGSjuSbcVA6B1r3inWce3iNL1VzEcoKaG4sZ4SeG/hfyeTxJZLnF7pOg3OfdLrkuWZD1RORPiW+A+68RCI"
    "03N2+LSgjtES3baUpWgjj3lUxhfZSu2mfBBpRpXnNnzTH8oy59nGZU9/FMfuF/q5b7MMwTGQ0wztpUgr1C29UnCDygQj4UquVSbKWL6/8GPCzq3qI3hWompN"
    "5tcnNLp9UA4NDyBtqNeLnAoxX2SbjugZ9ZnZHIulUQLaPVA9xbpiudquIWi3P9PtfczeKM45Vnh1sXT6Y8//AMtXQqj/AH+QqtHOpOcOgLlfCGzuv+xM8N7Z"
    "0/WR5h4fNXxLWKYmVMzRm4wbX7W25aE8NeWatWjHJF3l4piT1xjC8FRalG01zpWtd27Zz8RfNoSvTZpqqiHX8W1Nzh3C6tVZ51YfldwZ9pzxJa78ZOGYbxEN"
    "2rOLVkGQtxIsWRj8aj1leeLYEmG6DdDaq1z5l6SoQbt3xh24lwPQ6i3uqo6dHzLsh9I/aazxanTXLkzRXM5p/CH6IaX6tYHrFirWYad5FGu8CrhsOSoLm8Ae"
    "DwmFfWEhL1SoPql6y8hqbF6xOyuevSH6E4TxfRcUsfWNLVHPr+C08WmJ6gP32zXGw4s1kcrJLXEs2DM0yWRCk43dfd3Hbi00AkBjRqtHTPmJCMTb6aHyXodH"
    "Tbm1h8+7QzqadXuimaszynPTxXPihyHTTL7njs256wYDWHZYV8jTouW3d5iG/PjHFF4eqyYCLzOx0CEiqQUdMthbS5Xoju655KcRu2tRVZqqmYiM8ufNlrGU"
    "/Vfpxh+YfUpKs1bni8SV7DyPS7C3tCXRLn6dwc9vp+Kuj4jZpmp7vs/qKquHwodb3QR6J+n1tvoXXU2qYh39u5VRciXzvXIal3d3w+rtU93Smb1T73jmVB6J"
    "+L1tvlTu6Ud9U896ptIu7uej5qU26e8hMXKu6qU48gus7+oXQ3OU8W33/CK2qt093DjWrlXfVKvePCRd3c9HzfMse7pcjvqjvHk/U7np+b5fwk7uk76o7z5v"
    "1O56Pm+ZO7pO+qO8+X9Tuen5vlTu6TvqlMZA96c9xc3UbH1fN4iWk26cMouVZVO8eES7u56fm+VZTbpy1i5UUkeIh7s56Pm+ZT3dJ31T53qm0S7u56fmpspO"
    "+qeu8eKo9Fz0ett8yju6fsq13au7lQckfqhkum5Tz+57fEXlW9FunFTC9dq72hX63jq3Rk/CO7dt9Cx7ulyu8rw+d6Lbu7o75uW3b6U7uk7yt961Op0uifl5"
    "7tvoSbdOEzcqee8e57u6ueblt2qabUdVKrsx1QrhcCt+xmPH2h6K+IfL4l2VjT271GZ6ur1GouWrmKeiWy4I7XhjmVXPSRbVwa6IiuaZc+1V9jMPdJHiIe7O"
    "ej5vmWXd0t+8rwd48Il3dz0/N8qd3Sd5WUkeIh7s56Pm+ZO7p2om7VuUpkj9S7um6Hi823y+Ja2LdPNhqb1UQrd48VR6Lno9bb5lnNunLaL1WHzvXh3d2d97"
    "y7VGylbvqsvvW90Ieifo9bb6FFFvvK4iroiLt2aquXT5uC+097RbiU0H4jpmgeit/j49aLHbo1Zr3sUy+5dnHmaOVNyrwGJN+PbQKDy9Bbt27w9Bxjjeo0eu"
    "izpqcQ/RP0W/Rf2W7U9mfKHE47yu5ujGZjbiZj1f/n+jt/s4dR4WpXBLh2eW7TSBi7l4jz3ZsGwW0IsSkkJTwOyAbDaIdUxJ3aPx+VPRSi9xwjUXNdw+i9X1"
    "fm7t52d4d2Q7W6jhWjmJt0T9nH4xE/LOEfiiyJmz321ZXhebvDd7TfbCxkrTeYSWXYMZ+WzQOjAEqMSSk0JwK9TbupSu3eQUBRr6qutM84x/9DkdmdPcv2Zt"
    "amI21xOOUYmYzjdV1jE9MeK38TOrEeFebhrFj+bQb9i+NYHS7PWa06gzrc9IoMl6jjrVIlelIqfS6Ibzr7oOzlTfuVb1VdVcYnHLxX4Pw/vcaW5TtrqqmM7Y"
    "qxy9cTziPXM+HN0Y9falCbuHsPMr1GwLo0a8Q7qc/TTn8Hwrs7XO3h4jVWq9PdzE7oiZ6Po3Yiktx/YmZTqiNep0x5Bu9UvEtcQxUxvhFHOR7Bzx2ENOmTI7"
    "i/B8SYgnohYzepbkFlqZGkyDeec/VDYiQAPVPaJV3eYR5DX8FWqpZ09U4b4RMuvewk8ent8JMjzP9rxKkUrz0evZguoy37DzvduXi6I7Q8XreLwquxLyN7Ih"
    "eKlnmj0fV6I+PxbfD4k2j6V6L3GlbPO92/8Akj4PFt8Xi8KCLBvEqky5de1yyBqQNWdrI+KmwB2+bzbuZen1VoJJXohbZc9h53u3q9EdweL4fEoxA++zBdR5"
    "v2Hne4iVd3RptPl8XxeJMQPhXwhZae9hZ9eqVabeiO4P2/EqD6V4IZDzPsRMLpjWvU6I7T2/a8SC3t3icN6dnFjczu7kVsBIYo9XeJHz5+Ly7SHb+ESvNLNg"
    "HHJMGVwa6x2l61ydldK8hoTxNjsIfY1/4dyrNLRC0G0S081r4PtFpuc2+dSdadOrO7a7taLxKt82KT1sZbeEJMVxtwQMfCYbtpbR3U8IqlNWJTshsrS/QnSv"
    "RiK7b9MsTpaYbluhwKW9ma8cZqPFAgZBtkzIGvQZbqhSlTrXce4vSlVeTZDS2reh3Zv8OONBa9bnbTY7ZfrPMx6yWm/5PLcBqA9sdkwrZGN4u7NV6bRGEYAo"
    "Ithz5CIq/e1K7Yhl+N8O/B5rnVjXXDIMDJ7NkM4r0wVryJ+TY7hMJurNZ3cwd7ocjbuGruzfupWpV3+lR3tcJiEzD+Bzhowa5RbvaMGmvSrdEgQ7fIu2ST5x"
    "xIkKYzNiRmayHz6TLMhhowaHaA8q+HkRc1V2pOyEu7cF3DRerhcLjcNL41XLnW7d8BuW+DZeyZxTn7QE9gdY4UYz20pzMKl5jOpVi7WnZSvkDh40ftmS23Lb"
    "dhwx7nZ79crvb5jMp0DCVP3d8KtRLxg7U9xNlzDcIFt5gO1VcqRTSx/VPgp4dNZr7dL/AJ9hct52+ORnr7Ht+QTYUa5vxhEY0iSzHdBt95oQb2OmJGHSa2l7"
    "mHKab0xyKqG12wEBoA1r4acvTXmqLvaAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIMT13+wZmf7k7j/ADVx"
    "Bpbsmv8AetuG/wD6CcU/8nirRm6BQEBAQEBAQEBAQRZRtjcookzuKu/aW7y+FWpEpVBAQEBAQEBAQEBBFtZNmMjps7OUo6F4ue6vxkEpAQEBAQEBAQEBBGux"
    "Nt28yeZ303D4d231hSgSUrBAQEBAQEGJ3n/ZmX/xg/4oryetzOtnL0mlqrnRxmHP/H45wds4bgRcZ1vkyLUWqlophoxe+7gyPa/3Mi7mQlt29bz+5fG9VW0s"
    "36YnZTk1FNE4zLpXD/6ikf8AKP8A0FdtwnO113EIXddp/G6yn0UQSZ9mib6Pj7qNepu9Xd5eStUJaqCAgICAgICAgIBe9X9pBFs5NuWthxlnYFWR2t7ue1Wq"
    "VpSlVYQEHLXaP/Zj4R/7qCH/AJNZAonomOrrlUXEBAQEBAQEBAVahorj/u0DGdH8dyq8vE1b7RqtiM+5SRZI6R4zN7iG46VBEq7QAakVftCtrfNnXMuYu0E4"
    "1dQMiu7194UdZptqt+MYI7e7FMjSJdGMpu3eHQ9j4sZiC6Ux5mjLZOtOHUBCSBdKlNztNrdFPrZzXLddi1I4nblxXtcMUwbk1awv31cPZdWP7i5i5seC00d2"
    "7e8UulTa2c93dGufrc1WuKI6L0VNg6yQpsni80ZmxojpsR4uS94eFutRa3RI+3dX1eazp6StX0cs6o8YmsNhw+wYBadQ7yzndt1H1EbzFhu2l1YlsiWzI37b"
    "WRXpbQar07abJ15C7RoCEjESXIi3EsouTDAM71X1H1Q4bWbXpzrbm+preVaQ0uuq8G/wOYWC60k23uotiLAdyeMzlh3Om7e011dvh6hzTTTnnCPtMsxzXPjx"
    "uHE3dByHUu2Wq5hnWQ26Xgjl2uLzzdgYpMKG8FtC1EywJRwjSG7gUrY6Z0Aj8fRBNNvEYPttscIcvXOA/wAN98z3U/MMjmagaVz79nZZDtqDNxK32cwaABaE"
    "YoCbr3Jr0cy3VruLdVY1RGJXtTlf40gIXbI3yBUuVbhw12x4fndG/wA0S/8AGH+Msbc/aaV0uj5w9SG82Te/c2VNvxvD5VfbUiIiumUO2RIN2xSPAuNoarFk"
    "wBakQX294VbqHKrVaF5qcq8tpK2aqebj27Vu9YqouxmlzrlPZj6Caf4vl2S8IGA2rBNQ73a32LdkxFIlUhVdpWhiwDzpBF3jUw3tCNQE/R5dq4XGKuJavh00"
    "aavFfqOy/AOy/B+PU6m/p80z6sz0nq/Pns8Y934LeMK9aO8QGXXXDnpTNYfsHIAPY+bK3e5Gbhegabae5Oh5t23fy8JeA0na3XUR5O4vb210ziKn1G79FGh0"
    "lVfHOB3/APha4zNuIzj/AN9X6eaJcYGjOuGL0v8ApVeYl9j2qUcGQ/Fd9LLwEQV9BDzES27hLykJbh3CvY1Xa9NREx0eG0lfD+KTVTYuxVPr5eDK7hn2KXZk"
    "Y10wiPJaCR1xbkNgdBd+PyKnm+clHE4lzb3Abdy3urmJx0jCzZXlbOXSm5jMSrItjVvYRc619JLh6y7VVLseH6SnTaPELUuJmrDssRVgU4qRtpExWbaRIirv"
    "IRE09zUox/DIf903e6Urt+L4RW1W7u4ce1tm7UrLHFbk7aRMVm2kTFZtpE+2iaaXj3TvVfe5dP8A71pO7DKNr2sp3ZaxtFOK07aRMVm2kUTu+yV0UzblTe8L"
    "zLnT3eIqbvi+H/8AUuRb3Yqce9TT3tEqiwnc5GaRMVmaRJiqSNqk89QSFlvzn5fm/OqtaIqYXZjPJ5cbJpsBL3WvUpuIv21rRXXjNLKq1RnFT6RFDKrzjhVa"
    "rt8O3yf+1VqqnUTiOpn6rzlWXHxXDlxVTMCfaTmkSN2CaYypzPE2LZN7xJwabf3y1sbsuNqKaVRZzFWXIimnAoxUnbSJE4pimnqTTavVzOcfg5n7Uq36M2DR"
    "OPr5qLwzWLO7nj06NDpJn3KVAcaiukQUo45EMKvALpN0oB+H3QuXJcvSaLTa/VbNTTGY9f8A+HD4n2+7U/R/w+7qOGXZponETEYmOfLPOJx/R1pwxZnj+Z8J"
    "uF53jWMQbXbp2FxZTNntDXTjRaFHEqsN09URrzEV3uli3a01MUdHk9Tr6+LVTrL9U1V1c5mes5/wzg8MwS8XO3ZVdsKtz91gR6BCnvwm3H4tOXvA4Q8x/eqa"
    "9NFNybkpt6y/a022mqdvg8y9O9N5xW8puA2h6tocq5aSctzRdzOtee5r0eCvP0+jkop0sdWtHENdRTVVvmmJ/GefqXl1yjp89vJa0fZ5OFTt51VdIeFbcgTc"
    "T0WzEaRfYJp2C2YNG46Yi4XMt1XSKv8AhVWtVSlPVc1SKlp6CrvSJvBBBtJRSmXByPv3d8Gj273twtAPo/e7VoJyz3Am8EBBbYrcUssmPNumTwwWAcb2+ER3"
    "OlTxfxleambV/aIXhvHez/10yFxzZSFo7k8gi+KIWqQW7/uVZqaMz4SoVbXwqaZWxwdtY2n1laIftVGCzRZYw0bDTA5x1H1U0u4bOL67ascRd2jY/Yr9gdtt"
    "+MZteY1QttucjSpjkyC7MKnThm73iMY9Qg6+zaO6rPJaRiqnDKapc48VnE3rVk2cQrhw4ZVTTrTR3EJN1xC6XYbnjfs5fKXCVSSbbLVplHPrUaR3mopgHehl"
    "EYi/u9y5FFunazmqW4MQ1I4xW9c8c4bs3ekyZGTybTms3K7bb3mYNrs7MRv2VtY1fbE6FW5tNNA06NHe63KvPkTBcs6oo6rYqav1Gv8Axc4ZwzaO5MWud56e"
    "oUcZupWWZjlj1lbtUnuIORYQSIcCRW3MmfV5l0x3mw2Bu7ndpzTTRmTFTq3gdyfVfMuF7F8h1myVi83yU1IqV2jQpLHfI1JLox3SGTFiuVImaN8z6DQufrgj"
    "QTosrm3PJpTU26I1pXnVZboXelIICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgxPXf7BmZ/uTuP81cTlHVM"
    "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"
    "eK6uIqrh7SNlNePWKPWqIdRRXM7lqYjYplupMAup4TbrTb8YlyetDizyrVFx5cgQEBBTl9Sre9nbuEhr4vi+t/grWzzpcfURipUVJ5VNqJ+yKqwgJnJjAkdS"
    "eiiPKO90yoAtl+t/B4vireqYmHHjlPN7ce6bgNg2REfxfVH4yrFHrWmv1DLPRHaTpHWpcyIkqrwi1bzOXtY5y5ExtUpHkD3Pd7pRbWfRYXuqqs6vSbU+iJTV"
    "sPs4xKiVe57i8ZhUv29i3iIuOP8AasVZnoqvOdMd3hqXqju81VnTExLS7VGOTzHbJsdzg0318Tm37arcnCLcZ6vaq1EBVmaonMFNNu3P2lm1IxNnPNO79g72"
    "3ZeLPJhFu/8AmNEH/wBZcjS13tNdi5R1dZxix9Z0Vy1a5xMT/Zyn/qeHMHp2nuc6RzopkVnyRm6EJFt6BOM9L3vtkTf+CvRbYjiF67V1qx/Z817J3Lsdm7Fm"
    "7mdk185/GZfpat42W6Nng7yKYqvU0z+IpTPUQFWoQMmkMx7HIF4TIXh6G1svGVTLZ6P4y1ogTWWRjsgy3X0AI0FRUPSqCAg+EQCNSLyirUlaBi8aLHsccoZG"
    "Tb26QJODtr7qRH4v4yVFC4KoICAgg2nu7kye9Heqe6Zyc3D5SEAHaP8AArVK0tCdqlj45xwZ3XStwaVDNswxXGHBLykFwyC3Q6j+DteJVWdOrNoICAgICAgI"
    "CAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICDE9b9S2NFtF8u1jlWo5zWJYxPvL0Ft7YUgYsdx8mhKtK7alQKjz5fCg5QyPtWXAtFrteQacVtlyu31H"
    "XODKwTM4l4ZCHd8ltlp7vLedh0GK7znjWrRN7nWgf6ToGG8Az/VHjU1p061iPRe88PVrd54peshuF2xrUMZMmy2mHHdqzOkR34AAHeHxBhoNx8zI67TBh2oh"
    "E4Vu0rwfXPXZjhivWK3W03auklizO1327zGXAvLcqFHkSmfcmgAX44yoxHtoInR+pAI0AqCGd8EvGJD4z8HyPPrVpvccaiWfL5FqtoXOSDp3KF3ePKh3ClBE"
    "ekEiLLjvC3XmQidKVrzQa/Z7UXC5Okuv2rbWlVx7poli93ya1N1ubWzMbLCbniM6I6NK0Bp2VbJzA893KjYOemjo0QZdojx7YPrla9MZmPYhNiy9QJ1zt13t"
    "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"
    "Qb2wB8DAKkzleIw6UVZnCRSCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIMT13+wZmf"
    "7k7j/NXEJcX9mP2dHAFnfZv6BZvmfBdpfdLxd9F8YmXS6T8HhPPy5Ltqjm484ZNbjMzIiqReIiJcn6zW480Rno3j7V32bv3B2kf8n9v/ADSfWazZHge1d9m7"
    "9wdpH/J/b/zSn6zV4p7ik9q77N37g7SP+T+3/mk+s1eJ3FJ7V32bv3B2kf8AJ/b/AM0n1mrxO4pPau+zd+4O0j/k/t/5pPrNXidxSe1d9m79wdpH/J/b/wA0"
    "n1mrxO4pPau+zd+4O0j/AJP7f+aT6zV4ncUntXfZu/cHaR/yf2/80n1mrxO4pPau+zd+4O0j/k/t/wCaT6zV4ncUntXfZu/cHaR/yf2/80n1mrxO4pPa1ezt"
    "g1pZ4/A3pWDE3xSGW8Fg0B3Z4h3j0uRbefh3eVPrNfidxSe1d9m79wdpH/J/b/zSfWavE7ik9q77N37g7SP+T+3/AJpPrNXidxSe1d9m79wdpH/J/b/zSfWa"
    "vE7ik9q77N37g7SP+T+3/mk+s1eJ3FJ7V32bv3B2kf8AJ/b/AM0n1mrxO4pPau+zd+4O0j/k/t/5pPrNXidxSe1d9m79wdpH/J/b/wA0n1mrxO4pPau+zd+4"
    "O0j/AJP7f+aT6zV4ncUntXfZu/cHaR/yf2/80n1mrxO4pPau+zd+4O0j/k/t/wCaT6zV4ncUntXfZu/cHaR/yf2/80n1mrxO4pB7Nfs7754rhwO6Vvd0/UrP"
    "eMHgHsaDyhTc14Rp8A+qn1mvxO4pPau+zd+4O0j/AJP7f+aT6zV4ncUntXfZu/cHaR/yf2/80n1mrxO4pPau+zd+4O0j/k/t/wCaT6zV4ncUntXfZu/cHaR/"
    "yf2/80n1mrxO4pPau+zd+4O0j/k/t/5pPrNXidxSe1d9m79wdpH/ACf2/wDNJ9Zq8TuKT2rvs3fuDtI/5P7f+aT6zV4ncUntXfZu/cHaR/yf2/8ANJ9Zq8Tu"
    "KT2rvs3fuDtI/wCT+3/mk+s1eJ3FJ7V32bv3B2kf8n9v/NJ9Zq8TuKQuzU7O3Hx9l7XwN6Vx3mvQL0fBYLRju8Jbai18UiFPrNfidxSe1d9m79wdpH/J/b/z"
    "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"
    "AgICAgICDE9d/sGZn+5O4/zVxUuVbKJq8IWop31xT4vy97PztWsw0x4EdF9OIekFtltWDSrH7a1Lcujgk6LFuZbE6js8O7avgXaX6Y7/AAPi17QRp87apjOe"
    "uJw+y8J+ivScQ0du93/pRnp4tvU7ZXO6U5fWTtf7fsw5+bXQ1fT1q6K6aY0vzdtX9DulpuzP1j5Htyud/Ila/phz82nn61f3b5o8zml9/wDI9uVzv5ErX9MO"
    "fm08/Wr+7fM8zml9/wDI9uVzv5ErX9MOfm08/Wr+7fM8zml9/wDI9uVzv5ErX9MOfm08/Wr+7fM8zml9/wDI9uVzv5ErX9MOfm08/Wr+7fM8zml9/wDI9uVz"
    "v5ErX9MOfm08/Wr+7fM8zml9/wDI9uVzv5ErX9MOfm08/Wr+7fM8zml9/wDI9uVzv5ErX9MOfm08/Wr+7fM8zml9/wDI9uVzr5ErX9MOfm0n6eNdVRP/AA/z"
    "J+h23Fcfr/kkR+2By+VbpN0c0XtFHYuzptldndx7y212+5rlWPp0186a5P1fw9bj3fohtRqbcd/4+pH9uVzv5ErX9MOfm1xfP1q/u3zcjzOaX3/yPblc7+RK"
    "1/TDn5tPP1q/u3zPM5pff/I9uVzv5ErX9MOfm08/Wr+7fM8zml9/8j25XO/kStf0w5+bTz9av7t8zzOaX3/yPblc7+RK1/TDn5tPP1q/u3zPM5pff/I9uVzv"
    "5ErX9MOfm08/Wr+7fM8zml9/8j25XO/kStf0w5+bTz9av7t8zzOaX3/yPblc7+RK1/TDn5tPP1q/u3zPM5pff/I9uVzv5ErX9MOfm08/Wr+7fM8zml9/8j25"
    "XO/kStf0w5+bU+frV/dvmnzOaX3/AMj25XO/kStf0w5+bSr6d9ddxjT/ADRP0O2qZn9f8ki49r/l1pJkYujFoc7xHB9zZdnfCZean62uZqvp019rb/w/zcWx"
    "9ENqYn9f8kf25XO/kStf0w5+bXC8/Wr+7fNyvM5pff8AyPblc7+RK1/TDn5tPP1q/u3zPM5pff8AyPblc7+RK1/TDn5tPP1q/u3zPM5pff8AyPblc7+RK1/T"
    "Dn5tPP1q/u3zPM5pff8AyPblc7+RK1/TDn5tPP1q/u3zPM5pff8AyPblc7+RK1/TDn5tPP1q/u3zPM5pff8AyPblc7+RK1/TDn5tPP1q/u3zPM5pff8AyPbl"
    "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"
    "4J8+s+N51pfH4inCwrPJmTyZthLE2OvHO8uSnXNsrq8y6TsmpDzHxUCg199a1Xc82cW2ValcNeV3vN7dqno7rE7h2TRsdCwXO4OWJq5MzreDtXQHoukPTdbc"
    "JwgMS5e6nQwcpt2xFzJNDEMQ4D8k021PxnN9P9f5AWvE8ejWSzWe+YxHnOR4wnvmGEjeGx6WfpddoG7wgI8hHard7lXu1qwLgIxTILvrRkOZYvcsYPULKqUs"
    "LLNxZeO0xIkgZbU2MIVcbZKRc+vcqhXnuq63R0KVEgo72eR3bKLdwbZHfsvi6m6163yMmyiLdrQ9Hmw8fZt8RmJbpLslqODAmdeZuPGRukZVr4dtAGnJR3kY"
    "5Ji3zbG0t0dtul2S5xksG8vy3M3yz2dlNutiNIrncYsPpBy8w7Yglzr8J1VJnLSIwzVVSICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA"
    "gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIPlac60r9pRjnlEzya90B4n9CuJ/S6us+hu"
    "ocS940MyXEcugibItPRnCB4Do6IkGwhr5qe96feWldueiIlM4feIDSLim0itWu+hGYt3/FL73j2Ju7LRthI6Mh2O5yExEqbXWTH009VVqp2FMs2GlKUVeqZl"
    "9pTlTlRJjJnLV2d8XugGm2sNv0My7MZEfIblJgRwBuzy3YkV6c4bUJmTKbaJiK7INswZB0wJ0h5DzV6aZxyTnDaAUrT31VEzhgugPEXovxQae01W0D1Ch5Lj"
    "x3GXB9k4O7Z3iO7VpwORCJegh9HxhISpzEqEpmjCIlnQjt+FQswTM+JLRTANZsR4fcwz6LAzDO2Zp4nZXRPqT6RG+q/trQdo7Q8XirTdy8PNW2TMImrDPFSO"
    "UJEmMozDG9OtVMG1WjXaTgt7pOCxX6VZboQsmHRnRj2PNeKlN20vWp6FaacdUbmSKm03KMuSxEjOS5Lm1toKmZfapT31ZNMsEuHE1onatJ8a1vmZuAYvmEqz"
    "xsduvdHdst25vssQR2bd49Vx9ofFSm3d4uStFE1Ey2CqYgzAmIMwKTMLXluT2PCsWuWY5JPpGt1pt7024SKhUukw0FTM+Q+mvIaVr6E25Mw+YbllhzzErXm2"
    "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"
    "FFuU8IRDjxHXWSB8YdZDu6k0BQWRqvGNmmm92vumHFVmGX2DI9VY1n0/LOsxs2L3C7W+BEn99fjTbdaQqDUmaHhHoPGUWB1gHa7tQbis3G3i2l3AB9fOse7T"
    "L1ZykWMbRkt5rcJDt6ZuTtrIXpkdsutEpLbP9Wg3yKOHV2eog1pwgcQuY6n4Norq5L4kJma5B9ebMMFy+VHinDgXmG5Iu8hj9RkNBA2WoEA2T9Jgz1QIy6ho"
    "O1of2QZv/NrP+Oa0n0IUjqvizXEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEEe5BU7e+2A8yJkqDT7fhU0KV84WPE8mxuNjUGNKvcVtxuKAONuPDQhKg+mlaK9dMzPJWmmPWuP1XYn/Z+F/2gVXEpxT4FcwxWvv5BC/7QKYwnMT6j"
    "6sMVp72QQv8AtAphOYhb+el/1U/Vxysfs17H9w9mOmz3ruu/qd363n6W/wAeznt3eJRtkysTGm/DXDjTIcPTzCG2LlcGZ1wYbs0QRlS2ZZTGZDo7PG6EkzkC"
    "ZeIXSqY+Ou5MSbmIap8K/CrrTq3Z9UdTcTw+9hbIF0ZmWW8WKFLjXKTN9jB72+LoFvebatbDQHXmWzw8+Q0omJNzM2sC4dYuQQcpj4JhjdztrjDlvuTdqiDI"
    "imxHfisVbd27gq1HlSWQ218Lch4B5C4Q1bZNygWlXC8TF0ilpnghN3uNJj3putjhbZ7MmQcqS09TZ7qDsh518xPnQ3HTMuZFWqYk3Ltlds0azli2Rc2s+N3h"
    "uz3Ri42lu6RmZAwZjPPoyWROhdJ4Oddjg8iHd4apiTctt+044asp1At+reS4BhNxyq1tiFsyWdaIj1whjTnyFqSYVcbp6S8pU9+qYNy4Wiz6M4+zZWbHZ8ah"
    "BjkSsPHxhxI7Y2uPURCrMbbT3ACEAHYHKnIKU9VNsm55kWDQ6VFrbpNgxd2OV8C8lHcgxyArkDovhM27eXeKOiJ0d8+8aV3bqJg3MYx7hz4KsTuHsvjGgOl9"
    "tlFJbld5t2J29l3rNvi+07uBqhdQXgB0S9+hjQqemnNMSbnyNw58FMW82vIYugOmLVwsQRgsk5vFLdR63hGaBqPSOfS3NUabAADZy2CAjTlSlFO03LxlOmnD"
    "RnGJhp9menOD3ewNXM7i3ZbpZYciGEs3DdORRlwKhR0nXXTq5y3VJ0y58yJRg3LdatA+DuyX+zZXZ9EdN4l0xtkWsducXGIDb9rboRHQIzgt7mB3GddoVp6T"
    "r8ZMSbkTBdDdC7BpZd9K88HHsug5HkFwvWUN3y2xnI91mTJZyjJxg6EBCNSAAEt20GgHdXam2TcvuPaacM+IMw42J6dYRbG7dZ5NotzVussRikW3STB2REbo"
    "AU2R3TaaM2h8Bk0BVpWtKJg3II6G8IdMsh54WjOnPs5b2YjUG9/U5A71HCJUKxgbe2bwozUA6YjX3PYO3ltop2m5Wy3SrQDOs3m5xmUa2XVy7Yx9T96tlxeb"
    "fgXSCL/XbCTGPc29Vs6vbCrTw0kvD66jEm5SuWh/CHetP4Wkt30a06l4pbJHXtuMyscgHb4rvi8bUYm+mBeIvSI8/FVMSbkvJtMOGTM4Mq2Zjptg12jT7ZFt"
    "05i42SI+3IhxnDONGMTCtDaaN10gbLwhUzqNKbqpg3K9qwPh5sOOy8PseEYbDtE+2hbptri2qK3GkQwoYhGNoRoBtDRxylG60203l4fFVTtNy2fWK4Pfrb/W"
    "d+stpz9SHfO9fUr9TUH2N6//AAvdun0t/wA7bzUbZNzLbFO08xezRMcxh+z223wGBYg2+CTTLMdoKchbAA5CA0p6KUp6ExJuSpOR4dNYOJMvUB1pwCFxtx8C"
    "odK+/So/DRNsm5i56b8NbuVWrOS0+wmt7sVt9jrLeK2iJ3q3w9tR7uw9s3tNbalTphWg8ir6Ewbnq9aecOOR6cRNHMmwTDrniUKOzHh4vcrZGkW5hpmg0ZCj"
    "Dgk2IhQabfR4eXoTbJuSsZxjQnCZ7F2w7F8VtMqJZmbRFk2yBGjutW5pwjaiAQDQqRwMzIWqeAanWtKeJNpuWlzSrQE9Qsd1Kjx7dGmYsN0Oxw4ckWYceTcn"
    "RdmTO7hyApLhdT3etN/6pken3c9zbJuZNY5kO6ZpOnW58H2hgstk62XOnPcdeXNXmeWFIhkazaRyESICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKOgjO2y2unV163sGVfMRs0rWqnf+KMRB7DWn+xkb/qKK2UbT2GtP8AYyN/1FEy"
    "bT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8A"
    "YyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEy"
    "bT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8A"
    "YyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybT2GtP8AYyN/1FEybVViOxGb6TDINj8UB5UVc5TiFREiAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC"
    "AgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg//2Q=="
)

if __name__ == "__main__":
    main()