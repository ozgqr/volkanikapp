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
gezici araç simülasyonu (2D Simulation).
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
            show(base_map(bbox, rough, "MOLA pürüzlülük bileşimi (0,6 / 2,4 / 9,6 km)")[0])
    with c2:
        if rough is not None:
            fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")
            for ch, col, lab in ((2, "#2b6cb0", "0,6 km (mavi kanal)"), (1, "#3a9a5c", "2,4 km (yeşil)"), (0, "#c0392b", "9,6 km (kırmızı)")):
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
    return {"ad": place, "body": "earth", "bbox": bbox, "dem": dem, "px": px, "reg": reg}


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
SIM = "2D Simulation"
SIM_LABEL = ":red-background[**🎮 2D SIMULATION**]"
PLACES = list(REGIONS) + [MARS]


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
        name = st.radio("Mekân seçin", PLACES + [COMPARE, SIM], index=0,
                        format_func=lambda o: SIM_LABEL if o == SIM else o)
        place_a = place_b = None
        if name == COMPARE:
            place_a = st.selectbox("1. mekân", PLACES, index=None, placeholder="Mekân seçin")
            place_b = st.selectbox("2. mekân", [p for p in PLACES if p != place_a], index=None, placeholder="Mekân seçin",
                                   help="İlk kutuda seçilen mekân burada listelenmez; aynı mekân kendisiyle karşılaştırılamaz.")
        mars_region = None
        if name == MARS or MARS in (place_a, place_b):
            mars_region = st.selectbox("Mars bölgesi", list(MARS_REGIONS), index=0)
        st.divider()
        this_year = date.today().year
        year = st.selectbox("Uydu verisi yılı (Dünya)", list(range(this_year - 1, 2018, -1)), index=0,
                            help="Sentinel-2 ve Landsat için yaz sahnesi, MODIS için tüm yıl kullanılır. Mars verisini etkilemez.")
        n = st.select_slider("Izgara çözünürlüğü (uzun kenar, piksel)", [300, 450, 600, 800], value=450,
                             help="Yüksek değer daha ayrıntılı ama daha yavaştır.")
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

    if name == SIM:
        sim_page()
        return
    if name == COMPARE:
        if place_a and place_b:
            compare_view(place_a, place_b, mars_region, n, year)
        else:
            st.title("Karşılaştırma")
            st.info("Sol menüdeki iki kutudan **iki farklı mekân** seçin. Beş volkanik alan ve Mars arasından istediğiniz "
                    "ikiliyi karşılaştırabilirsiniz; Mars seçildiğinde hangi Mars bölgesinin kullanılacağını da seçersiniz.")
        return
    if name == MARS:
        mars_page(mars_region, n)
        return

    set_body("earth")
    reg = REGIONS[name]
    bbox = tuple(reg["bbox"])
    W, H, px = grid_shape(bbox, n)
    st.title(name)
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

if __name__ == "__main__":
    main()