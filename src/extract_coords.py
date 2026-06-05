"""Phase B, step 1 — extract each patch's real-world location.

LESSON: our RGB jpegs have no location info, but EuroSAT's multispectral
GeoTIFFs are *georeferenced*: TIFF metadata tags record where on Earth the
patch sits. We read three tags straight out of the zip (never extracting
2.8GB to disk):
  - ModelTiepointTag : the map coordinate (easting/northing) of pixel (0,0)
  - ModelPixelScaleTag : meters per pixel (10m for Sentinel-2)
  - GeoKeyDirectoryTag : contains the EPSG code naming the UTM zone
Patch center = tiepoint + 32px * scale. Output: a tiny coords.csv keyed by
the same filenames as our RGB data. The 2GB zip is deleted afterwards.

Run:  .venv/bin/python src/extract_coords.py
"""

import csv
import io
import zipfile
from pathlib import Path

import tifffile

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = ROOT / "data" / "EuroSAT_MS.zip"
OUT_CSV = ROOT / "data" / "coords.csv"
HALF_PATCH_PX = 32  # patches are 64x64; center is 32px in


def epsg_from_geokeys(geokeys) -> int:
    """GeoKeyDirectoryTag is a flat list of 4-tuples; key 3072 = EPSG code."""
    vals = list(geokeys)
    for i in range(4, len(vals), 4):  # skip 4-value header
        key_id, _, _, value = vals[i:i + 4]
        if key_id == 3072:  # ProjectedCSTypeGeoKey
            return int(value)
    return 0


def main() -> None:
    rows, skipped = [], 0
    with zipfile.ZipFile(ZIP_PATH) as z:
        tifs = [n for n in z.namelist() if n.endswith(".tif")]
        print(f"{len(tifs)} GeoTIFFs in zip")
        for i, name in enumerate(tifs):
            if i % 5000 == 0:
                print(f"  {i}/{len(tifs)}")
            try:
                with z.open(name) as f:
                    tif = tifffile.TiffFile(io.BytesIO(f.read()))
                    tags = tif.pages[0].tags
                    tie = tags["ModelTiepointTag"].value      # (i,j,k, X,Y,Z)
                    scale = tags["ModelPixelScaleTag"].value  # (sx, sy, sz)
                    epsg = epsg_from_geokeys(tags["GeoKeyDirectoryTag"].value)
                easting = tie[3] + HALF_PATCH_PX * scale[0]
                northing = tie[4] - HALF_PATCH_PX * scale[1]  # y decreases downward
                stem = Path(name).stem            # e.g. AnnualCrop_1
                label = stem.rsplit("_", 1)[0]    # e.g. AnnualCrop
                rows.append([f"{stem}.jpg", label, epsg,
                             round(easting, 1), round(northing, 1)])
            except Exception:  # unreadable/missing tags — count and move on
                skipped += 1

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "epsg", "easting", "northing"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT_CSV}  (skipped {skipped})")


if __name__ == "__main__":
    main()
