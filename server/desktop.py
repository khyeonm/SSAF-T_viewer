#!/usr/bin/env python
"""The SSAF-T viewer server, for the desktop app: the site's viewer (app.py) over the user's own store.

    python desktop.py --web ../web --cache ~/.cache/ssaft --port 0

app.py is the site's viewer server, copied in unchanged (scripts/sync-viewer.sh).  Its catalog mode is
what the site's /viewer runs: a manifest names each dataset's layers, and the server cuts their tiles,
solid surfaces and gene index and serves them by id.  This module runs that same server with an EMPTY
catalog and adds one thing the site does not have, `/api/open?path=`: the SSAF-T store (.zarr) the user
picked is read, a manifest record is written for it (`manifest_for`) the way the published SSAF-T datasets
are written, and it becomes the catalog's one dataset (opening another replaces it).  From there the page
and the server are the site's own, unchanged.

What differs from the site, and why:
  - layers are cut when they are first asked for, not all at start-up: a laptop opening one store should
    not spend an hour on outlines nobody ticks.  The layers a dataset opens with are queued at once.
  - the gene index child process is this program again (`--build-genes`), which in a PyInstaller build
    is the executable itself, not `python app.py`.
  - the cache lives in the user's application-data folder; the page and its modules in `--web`.
  - large volumes are worked on in slices (see "memory" below): a laptop is not the site's server.

The app (electron/main.js) starts this with `--port 0 --token-file <f> --watch-stdin` and reads the port
from the `SSAFT_SERVER_READY <port>` line; the token keeps other local programs from reading files through
it, and stdin closing (the app quitting, or crashing) ends it.
"""
import argparse
import functools
import hashlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

# a big numpy allocation can stall for minutes while the kernel compacts huge pages (viewer/run.sh)
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
# pyarrow's own allocator (jemalloc / mimalloc) crashed the PyInstaller build with a segfault when parquet was
# read in one thread while another decoded zarr (a store's cell outlines being indexed as the next store's image
# was cut: every time, and never from source).  The system allocator does not; it must be set before pyarrow loads.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np  # noqa: E402

import app  # noqa: E402

FROZEN = getattr(sys, "frozen", False)
TMP = Path.home() / ".cache" / "ssaft-viewer" / "tmp"     # file-backed large arrays (_big); set in main
WEB = Path(__file__).resolve().parent.parent / "web"
IMAGE_BUDGET = 300e6    # source pixels of an image level the viewer resamples at once (see image_level)


# ----------------------------------------------------------------- the manifest of a user's store
PLATFORM_LABEL = {"xenium": "Xenium", "cosmx": "CosMx", "merfish": "MERFISH", "visiumhd": "Visium HD",
                  "stereoseq": "Stereo-seq"}
IMAGE_LABEL = {"xenium": "Nuclei (DAPI)", "merfish": "Nuclei (DAPI)", "cosmx": "Composite image",
               "visiumhd": "H&E image", "stereoseq": "H&E image"}


def guess_platform(images, shapes, points):
    """The platform, from the element names spatialdata-io writes (only the side panel and the gene
    panel's wording use it)."""
    names = " ".join(images + shapes + points).lower()
    if "morphology_focus" in names or "morphology_mip" in names or "morphology_zstack" in names:
        return "xenium"
    if "vizgen" in names or "dapi_zstack" in names or "merscope" in names:
        return "merfish"
    if "composite" in names or "cosmx" in names:
        return "cosmx"
    if "visium" in names or "square_0" in names or "enact" in names:
        return "visiumhd"
    if "stereo" in names or "ssdna" in names or "bin1" in names:
        return "stereoseq"
    return ""


def map_z(g, element, upu):
    """(bottom, top) in µm of a label map, by the same placement app._align gives its volume.

    The published manifests stand every flat layer on the map: the image as a 1 µm sheet just below
    its bottom (z0 = bottom - 1), the 2D outlines just above its top (z = top + 0.05)."""
    node = g["labels"][element]
    lv = app._levels(node)[0]
    a = node[lv]
    ax = app._axes(app._multiscale(node)) or (["z", "y", "x"] if len(a.shape) == 3 else ["y", "x"])
    Z = a.shape[ax.index("z")] if "z" in ax else 1
    M = app._element_matrix(node, lv)
    z0, dz = float(M[2, 3]), float(M[2, 2])
    if Z == 1:                                   # a sheet: app._align gives it z0 and a thickness
        return z0, z0 + (abs(dz) or 1.0)
    if dz < 0:
        z0 += dz * Z
    return z0, z0 + abs(dz) * Z


def image_level(node, upu, budget=None):
    """(level path, µm per pixel) of an image: the finest level of at most IMAGE_BUDGET pixels.

    The image is resampled onto the µm lattice in one piece (app.rgb_tiles_for), so its size is memory.
    The published Xenium sections were cut at s0 up to about 5e8 pixels and at s1 above; a desktop has
    less to spare than the server, hence the lower budget."""
    ax = app._axes(app._multiscale(node))
    best = None
    for lv in app._levels(node):
        a = node[lv]
        shp = dict(zip(ax, a.shape)) if ax else dict(zip(("y", "x"), a.shape[-2:]))
        M = app._element_matrix(node, lv)
        um = float(np.sqrt(abs(np.linalg.det(M[:2, :2])))) * upu
        best = (lv, round(um, 9))               # exact: rgb_tiles_for can then copy pixels, not resample
        if shp.get("y", 1) * shp.get("x", 1) <= (budget or IMAGE_BUDGET):
            return best
    return best


# A store's label elements are all offered, as the site's viewer offers a folder a visitor opens
# (viewer/local/layers.js): by their own names, and with nothing in their metadata to say which is the
# cell-type map and which are cell ids, by their values -- a map's values index the store's palette, cell
# ids run far past it.
def _shown(name):
    """An element's name for the panel: SpatialData names have no spaces, so `_` stands for one."""
    return name.replace("_", " ")


def _label_peek(node):
    """The largest value of a label element's middle plane, at its coarsest level read at most 1024
    pixels a side: enough to tell a map's class ids from cell ids."""
    lv = app._levels(node)[-1]
    a = node[lv]
    ax = app._axes(app._multiscale(node)) or (["z", "y", "x"] if len(a.shape) == 3 else ["y", "x"])
    sel = []
    for i, n in enumerate(ax):
        if n == "z":
            sel.append(a.shape[i] // 2)
        elif n in ("y", "x"):
            sel.append(slice(None, None, max(1, -(-a.shape[i] // 1024))))
        else:
            sel.append(0)
    return int(np.asarray(a[tuple(sel)]).max(initial=0))


def _region_table(g, tables, element):
    """The table that annotates a label element (its `region`) with a cell type per cell, or None."""
    for t in tables:
        tb = g["tables"][t]
        region = dict(tb.attrs).get("region")
        if element not in (region if isinstance(region, list) else [region]):
            continue
        if "obs" in tb and "cellman_annotation" in tb["obs"] and dict(tb.attrs).get("instance_key"):
            return t
    return None


def _stem(p):
    n = p.name
    return n[:-5] if n.lower().endswith(".zarr") else n


def title_for(path):
    """`XEN040` for `.../XEN040/ssaft_sdata.zarr`, the folder's own name otherwise."""
    st = _stem(path)
    if st.lower() in ("ssaft_sdata", "sdata", "store", "ssaft") and path.parent.name:
        return f"{path.parent.name} / {path.name}"
    return path.name


def bins_um_per_unit(g, store, shapes):
    """µm per `global` unit of a Visium HD store, from its bins: `<sample>_square_002um` are squares of 2 µm,
    so their side in global units gives the scale.  app._um_per_unit reads it from the transcripts, which an
    array platform's store does not have; there `global` is the full-resolution image's pixels (0.27 µm), and
    taking it for µm put the SSAF-T signal samples (in µm) at a quarter of the tissue's size and every scale
    bar 3.7x off.  None when the store has no such bins."""
    import pyarrow.parquet as pq
    import shapely
    best = None
    for name in shapes:
        m = re.search(r"square_(\d+)um$", name)
        if not m:
            continue
        um = int(m.group(1))
        if best is None or um < best[0]:
            best = (um, name)
    if best is None:
        return None
    um, name = best
    try:
        f = Path(store) / "shapes" / name / "shapes.parquet"
        # the first 2000 bins are enough for a median side; reading the whole column was 200 MB per open
        batch = next(pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["geometry"]))
        geoms = shapely.from_wkb(batch.column("geometry").to_numpy(zero_copy_only=False))
        b = shapely.bounds(geoms)
        S = app._shape_matrix(g["shapes"][name])
        side = float(np.median(b[:, 2] - b[:, 0])) * float(np.hypot(S[0, 0], S[1, 0]))
        return um / side if side > 0 else None
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        return None


def manifest_for(store):
    """A catalog record for one SSAF-T store, as the published records are written (viewer/datasets.json).

    The site's records were written by hand per dataset; their every value is either fixed or follows
    from the store's own metadata, and that is what is reproduced here:
      celltype_map       labels/ssaft_celltype_map; solid view, sigma 1 µm
      nuclei             the image (one), as picture tiles at its level's own pixel size, one channel of
                         a fluorescence scan and the middle plane of a z-stack, a 1 µm sheet under the map
      ssaft_cells        shapes/ssaft_cell_boundaries_2d coloured by tables/ssaft_cells, above the map;
                         with labels/ssaft_cell_labels_3d and shapes/ssaft_cell_boundaries_3d it is one
                         Show box: the 3D cells in 3D, the 2D outlines in the 2D slice
      vendor outlines    the platform's nucleus / cell boundaries, off until ticked
    um_per_unit is left to app._um_per_unit (the transcripts' transform), as the site does.
    The record has no `meta`: the app shows one store, named in the window title, so the side panel's
    dataset section is not shown."""
    g = app._open(str(store))
    keys = {grp: (sorted(g[grp].group_keys()) if grp in g else []) for grp in
            ("images", "labels", "shapes", "points", "tables")}
    st = str(store)
    platform = guess_platform(keys["images"], keys["shapes"], keys["points"])
    peek = {}
    for name in keys["labels"]:
        try:
            peek[name] = _label_peek(g["labels"][name])
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
    # the map shown at the start: SSAF-T's own by its name, otherwise (a store whose elements were renamed)
    # the first label element whose values fit the store's palette, as the site's viewer picks it
    pal = (app.palette_from_tables(st, 0)[0] or []) if keys["tables"] else []
    cmap = "ssaft_celltype_map" if "ssaft_celltype_map" in keys["labels"] else next(
        (n for n in keys["labels"] if peek.get(n) and peek[n] <= (len(pal) if pal else 254)
         and not _region_table(g, keys["tables"], n)), None)
    upu_given = None if keys["points"] else bins_um_per_unit(g, store, keys["shapes"])
    upu = upu_given or app._um_per_unit(g, {})
    bottom, top = map_z(g, cmap, upu) if cmap else (0.0, 1.0)
    has_table = "ssaft_cells" in keys["tables"]
    three_d = has_table and "ssaft_cell_labels_3d" in keys["labels"]
    layers = {}

    # the image: one is shown, the store's others are offered. Not offered: SSAF-T's own images (the
    # cell-type map's RGB rendering, the mask image: the map layer shows them) and a z-stack when the store
    # has a 2D image too (the z-stack is the same stain, plane by plane, and the 2D image is its summary).
    # Nor Space Ranger's hires / lowres copies of the full image: they are placed only in their own
    # downscaled coordinate systems, so read in the shared frame they would land at the wrong size.
    full = any(n.endswith("_full_image") for n in keys["images"])
    pref = ("morphology_focus", "morphology_mip", "composite", "dapi_zstack", "staining", "he_image")
    ranked = sorted(keys["images"], key=lambda n: next((i for i, p in enumerate(pref) if p in n.lower()), len(pref)))
    shown = []
    for name in ranked:
        node = g["images"][name]
        if name.startswith("ssaft_") or not app._levels(node):
            continue
        if full and re.search(r"_(hires|lowres)_image$", name):
            continue
        ax = app._axes(app._multiscale(node))
        a = node[app._levels(node)[0]]
        stack = "z" in ax and a.shape[ax.index("z")] > 1
        rgb = "c" in ax and a.shape[ax.index("c")] == 3 and str(a.dtype) == "uint8"
        shown.append((name, stack, rgb))
    if any(not stack for _, stack, _ in shown):
        shown = [s for s in shown if not s[1]]
    n_rgb = sum(rgb for _, _, rgb in shown)
    for i, (name, _, rgb) in enumerate(shown):
        node = g["images"][name]
        lv, um = image_level(node, upu)
        ax = app._axes(app._multiscale(node))
        a = node[lv]
        if i == 0:
            label = IMAGE_LABEL.get(platform, "Image")
        elif rgb:                                    # a colour slide image beside the stain: the H&E
            label = "H&E image" if n_rgb == 1 else f"H&E image: {name}"
        else:
            label = f"Image: {name}"
        spec = dict(label=label,
                    type="sdata_image", store=st, element=name, level=lv, um_per_px=um, as_image=True,
                    z0_um=bottom - 1.0, z_um=1.0, show=i == 0)
        if "c" in ax:
            nc = a.shape[ax.index("c")]
            if not (nc == 3 and str(a.dtype) == "uint8"):
                spec["channel"] = 0                  # a fluorescence scan: its first channel (DAPI)
        spec["_mono"] = not rgb                      # a grey picture: one channel on the GPU (see rgb_tiles_for)
        if "z" in ax:
            spec["plane"] = a.shape[ax.index("z")] // 2
        layers["nuclei" if i == 0 else f"image_{name}"] = spec

    if cmap:
        layers["celltype_map"] = dict(label="SSAF-T cell-type map" if cmap == "ssaft_celltype_map" else _shown(cmap),
                                      type="sdata_labels", store=st, element=cmap,
                                      solid=dict(sigma_xy_um=1.0, sigma_z_um=1.0))
    # every other label element, off until ticked: cell ids drawn by their cell's type through the table that
    # annotates them, a class map like the cell-type map, or cell ids no table names, in repeating colours
    # (label_volume) and with no solid view (a surface per colour of 255 is more than a desktop can make)
    for name in keys["labels"]:
        if name == cmap or (three_d and name == "ssaft_cell_labels_3d") or name not in peek:
            continue
        node = g["labels"][name]
        spec = dict(label=_shown(name), type="sdata_labels", store=st, element=name, show=False, group=name)
        tb = _region_table(g, keys["tables"], name)
        if tb:
            spec.update(table=tb, solid=dict(sigma_xy_um=1.5, sigma_z_um=1.0))
        elif peek[name] <= 254:
            spec.update(solid=dict(sigma_xy_um=1.0, sigma_z_um=1.0))
        else:
            spec.update(solid=False)
            if "z" not in (app._axes(app._multiscale(node)) or []):   # a flat segmentation at full
                spec["level"] = image_level(node, upu, 100e6)[0]   # resolution is a 4 GB int32 read
        layers[f"labels_{name}"] = spec

    cells = dict(label="SSAF-T cell segmentation", store=st, show=False)
    grp = dict(group="ssaft_cells") if three_d else {}
    if "ssaft_cell_boundaries_2d" in keys["shapes"]:
        layers["ssaft_cells"] = dict(cells, type="sdata_shapes", element="ssaft_cell_boundaries_2d",
                                     z_um=top + 0.05, color="#ffffff", **grp,
                                     **(dict(table="ssaft_cells", table_key="cell_id") if has_table else {}),
                                     **(dict(view="2d") if three_d else {}))
    if three_d:
        layers["ssaft_cells_3d"] = dict(cells, type="sdata_labels", element="ssaft_cell_labels_3d",
                                        table="ssaft_cells", group="ssaft_cells", view="3d",
                                        solid=dict(sigma_xy_um=1.5, sigma_z_um=1.0))
        if "ssaft_cell_boundaries_3d" in keys["shapes"]:
            layers["ssaft_cells_3d_boundaries"] = dict(
                cells, type="sdata_shapes", element="ssaft_cell_boundaries_3d", table="ssaft_cells",
                table_key="cell_id", id_column="cell_id", z_column="z_um", z_step_um=1.0, fill=False,
                color="#ffffff", group="ssaft_cells", view="3d")

    # the vendor's own segmentations, and any other outlines the store carries
    who = PLATFORM_LABEL.get(platform, "Vendor")
    for name in keys["shapes"]:
        if name.startswith("ssaft_"):
            continue
        if "nucle" in name:
            label, color = f"{who} nucleus segmentation", "#63d471"
        elif name == "cell_boundaries":
            label, color = f"{who} cell segmentation", "#ffd166"
        else:
            label, color = f"Outlines: {name}", "#63d471"
        layers[f"shapes_{name}"] = dict(label=label, type="sdata_shapes", store=st, element=name,
                                        z_um=top + 0.05, color=color, show=False,
                                        # outlines named as a label element are its outlines: one entry
                                        **(dict(group=name) if f"labels_{name}" in layers else {}))

    if upu_given:                                # every layer read in that one frame
        for v in layers.values():
            v["um_per_unit"] = upu_given
    if not layers:
        raise ValueError("nothing to show: the store has no SSAF-T cell-type map, image or outlines "
                         f"(images {keys['images']}, labels {keys['labels']}, shapes {keys['shapes']})")
    return dict(id=None, meta=dict(title=title_for(store), platform=platform), layers=layers)


# ----------------------------------------------------------------- opening a store
def find_store(p):
    """The zarr store at, in or (one level) under `p`."""
    def is_store(d):
        return (d / "zarr.json").is_file() or (d / ".zgroup").is_file()
    if is_store(p):
        return p
    kids = [c for c in p.iterdir() if c.is_dir() and is_store(c)] if p.is_dir() else []
    zarrs = [c for c in kids if c.name.endswith(".zarr")]
    if len(zarrs) == 1:
        return zarrs[0]
    if len(kids) == 1:
        return kids[0]
    raise ValueError(f"not a zarr store (no zarr.json in it): {p}")


OPEN_LOCK = threading.Lock()
CACHE_ROOT = None       # <app data>/cache: a folder per store under it (open_store), see prune_cache
CACHE_LIMIT = float(os.environ.get("SSAFT_CACHE_LIMIT_GB", "20")) * 1e9
OPENED = ".opened"      # the marker in a store's folder: its time is when the store was last opened


def _size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prune_cache(keep):
    """Keep the cache under CACHE_LIMIT: whole store folders go, the least recently opened first, never the
    one open now.  A store's prepared data is all in its own folder, so a folder goes or stays as a unit and
    no layer is ever left with half of its files.  What an older version kept loose in the cache's root
    (one folder for every store) is removed outright: nothing reads it any more."""
    try:
        for e in os.scandir(CACHE_ROOT):
            if e.is_file():
                try:
                    os.unlink(e.path)
                except OSError:
                    pass
        stores = []
        for e in os.scandir(CACHE_ROOT):
            if e.is_dir() and e.name != keep and os.path.exists(os.path.join(e.path, OPENED)):
                stores.append((os.path.getmtime(os.path.join(e.path, OPENED)), e.path))
        total = _size(CACHE_ROOT)
        for _, path in sorted(stores):
            if total <= CACHE_LIMIT:
                break
            freed = _size(path)
            shutil.rmtree(path, ignore_errors=True)
            total -= freed
            print(f"cache: removed {Path(path).name} ({freed / 1e9:.1f} GB), the least recently opened store", flush=True)
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()


def open_store(path):
    """Read the store at `path` and make it the catalog's one dataset (the one before is dropped, and its
    queued builds with it), queue the layers it opens with.  Returns its id."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise ValueError(f"not a .zarr folder: {p}")
    store = find_store(p.resolve())
    did = re.sub(r"[^a-z0-9]+", "-", _stem(store).lower()).strip("-")[:40] + "-" + \
        hashlib.sha1(str(store).encode()).hexdigest()[:8]
    with OPEN_LOCK:
        if did in app.CATALOG:
            return did
        d = manifest_for(store)
        for old in list(app.CATALOG):
            app.CATALOG.pop(old)
            app.GENES.pop(old, None)
            CELLS.pop(old, None)
        for reg in (app.TILES, app.SOLID):
            for k in [k for k in reg if k[0] != did]:
                reg.pop(k)
        d["id"] = did
        if CACHE_ROOT is not None:                   # this store's prepared data, in a folder of its own
            app.CACHE = CACHE_ROOT / did
            app.CACHE.mkdir(parents=True, exist_ok=True)
            (app.CACHE / OPENED).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            threading.Thread(target=prune_cache, args=(did,), daemon=True, name="prune").start()
        for k, spec in d["layers"].items():
            if spec["type"] == "sdata_image":
                spec["_rgb"] = app.sdata_is_rgb(spec)
        app.CATALOG[did] = d
        for k in d["layers"]:
            if app._solid_spec(d, k) is not None:
                app.SOLID[(did, k)] = dict(status="building", done=0, total=0)
            if app._tiled_spec(d, k):
                app.TILES[(did, k)] = dict(status="building", done=0, total=0)
        # what the dataset opens with is cut now, in the order the page asks for it; the rest on request
        for k, spec in d["layers"].items():
            if spec.get("show", True) is not False:
                JOBS.add(("tiles", did, k), prio=10)
        for k, spec in d["layers"].items():
            if spec.get("show", True) is not False:
                JOBS.add(("solid", did, k), prio=20)
    print(f"opened {did}: {store}  layers {list(d['layers'])}", flush=True)
    # the transcripts' gene index starts now, beside the layers, rather than when the page asks for it after
    # its first layers have arrived.  It is built in its own process (app._genes_build), so it neither waits
    # for nor holds up the build queue; a store whose genes are drawn as a grid has no index to build.
    try:
        if app._array_genes(d) is None:
            app.genes_status(d)
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()                                    # the page asks again, and reports it
    # the cells' outline index for the hover's location (a few seconds for a 3D store), ready before the first
    # hover asks for it
    threading.Thread(target=cell_index, args=(did,), daemon=True, name="cells").start()
    return did


# ----------------------------------------------------------------- building, one job at a time
class Jobs:
    """The builds, one at a time (each can take gigabytes), the one the page waits for first.

    A job is ("tiles" | "solid", dataset id, layer key), the same work app.build_tiles / build_solids do
    for every catalog layer at the site's start-up."""

    def __init__(self):
        self.cv = threading.Condition()
        self.queue = {}          # job -> (prio, seq)
        self.seen = set()        # queued or done
        self.seq = 0

    def add(self, job, prio):
        with self.cv:
            if job in self.seen and job not in self.queue:
                return           # built (or failed) already
            self.seq += 1
            old = self.queue.get(job)
            if old is None or prio < old[0]:
                self.queue[job] = (prio, self.seq if old is None else old[1])
            self.seen.add(job)
            self.cv.notify()

    def ask(self, job):
        """The page is waiting for this one: to the front."""
        self.add(job, prio=0)

    def run(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                job = min(self.queue, key=self.queue.get)
                del self.queue[job]
            try:
                self.build(*job)
            except Exception:                                    # noqa: BLE001
                traceback.print_exc()

    @staticmethod
    def build(kind, did, key):
        d = app.CATALOG.get(did)
        if d is None:
            return
        spec = d["layers"][key]
        if kind == "tiles":
            st = app.TILES.get((did, key))
            if st is None or st["status"] != "building":
                return
            t0 = time.time()
            try:
                build = {"sdata_image": app.rgb_tiles_for, "sdata_shapes": app.shapes_for}.get(spec["type"],
                                                                                            app.tiles_for)
                meta = build(spec, progress=lambda i, n, st=st: st.update(done=i, total=n))
                st.update(status="ready", meta=meta)
                print(f"ready: {did}/{key} ({meta['kind']}, {meta.get('bytes', 0) / 1e6:.1f} MB, "
                      f"{time.time() - t0:.0f} s)", flush=True)
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                st.update(status="error", error=f"{type(e).__name__}: {e}")
        else:
            st = app.SOLID.get((did, key))
            sp = app._solid_spec(d, key)
            if st is None or sp is None or st["status"] != "building":
                return
            t0 = time.time()
            try:
                build = app.instance_solid_tiles_for if spec.get("table") else app.solid_tiles_for
                meta = build(spec, progress=lambda i, n, st=st: st.update(done=i, total=n), **sp)
                st.update(status="ready", meta=meta)
                print(f"solid ready: {did}/{key}, {len(meta['classes'])} cell types "
                      f"({time.time() - t0:.0f} s)", flush=True)
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                st.update(status="error", error=f"{type(e).__name__}: {e}")


JOBS = Jobs()


def _genes_subprocess(store, element, st):
    """app._genes_subprocess, with this program as the child: `python desktop.py --build-genes`, or the
    frozen executable itself, which has no script to name."""
    cmd = ([sys.executable] if FROZEN else [sys.executable, str(Path(__file__).resolve())]) + \
        ["--build-genes", str(store), str(element), "--cache", str(app.CACHE)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)           # Windows: no console window for it
    # stdin=DEVNULL: with --watch-stdin a thread sits in a blocking read on our stdin (the app's pipe).  Left
    # to inherit, Popen duplicates that handle, and on Windows duplicating a pipe handle with a synchronous
    # read pending in another thread blocks until the read returns -- which is when the app quits.  The
    # child was then never started: the genes stayed "building" with nothing in the log.
    print(f"genes: indexing {element} of {store} in a child process", flush=True)
    pr = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", creationflags=flags)
    tail = []
    for line in pr.stdout:
        if line.startswith("progress "):
            _, i, n = line.split()
            st.update(done=int(i), total=int(n))
        else:
            tail.append(line.rstrip())
            del tail[:-20]
    if pr.wait() != 0:
        raise RuntimeError(f"the gene index could not be built (exit {pr.returncode}): "
                           + (" | ".join(tail[-3:]) or "killed, most likely out of memory"))
    return app.gene_index_for(store, element)                    # now a cache hit


app._genes_subprocess = _genes_subprocess


# ----------------------------------------------------------------- memory: what a laptop can hold
# The site's server has hundreds of gigabytes; a laptop has 8 to 32.  A 3D Xenium map of 10,000 x 10,000
# x 32 voxels is 3 GB as uint8, and two of app.py's steps multiply that many times over on the CPU:
#   - np.bincount of a uint8 volume first converts it to 8-byte integers (24 GB for that map);
#   - the solid view smooths every class over the WHOLE volume in float32 (three 12 GB arrays at once),
#     where on a GPU it goes in y strips with a halo wider than the filter reaches.
# Both are replaced here by the same arithmetic in slices: bincount summed over slices of the volume,
# and the GPU's strip loop run on the CPU (app._gpu_pairs / _gpu_slots / _lead_by_class_impl, with numpy
# and scipy in place of cupy).  The strips' halos cover the filters' reach, so the numbers are those of
# the whole-volume pass; only the peak memory changes.
SLICE = 1 << 27                 # elements per bincount slice
STRIP = 3e7                     # voxels per smoothing strip (about 20 bytes each while it is worked on)


class _Numpy:
    """numpy for app.py, with bincount taken in slices."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def bincount(x, weights=None, minlength=0):
        x = np.asarray(x)
        if weights is not None or x.size <= SLICE:
            return np.bincount(x, weights=weights, minlength=minlength)
        flat = x.reshape(-1)
        out = np.zeros(minlength, np.intp)
        for i in range(0, flat.size, SLICE):
            c = np.bincount(flat[i:i + SLICE], minlength=minlength)
            if len(c) > len(out):
                out = np.concatenate([out, np.zeros(len(c) - len(out), np.intp)])
            out[:len(c)] += c
        return out


app.np = _Numpy()


BIG = 1 << 30                   # bytes: an array larger than this is kept in a file, not in RAM


def _big(shape, dtype):
    """np.zeros, file-backed when large: a 3 GB volume and the two or three of its size a solid view writes
    are more than a laptop holds at once.  The file lives in the cache's tmp folder; the OS pages it in and
    out as it is used, rather than the job being killed.  (On POSIX it is unlinked at once and disappears
    with the array; Windows cannot unlink a mapped file, so the folder is emptied at the next start.)"""
    import tempfile
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if nbytes <= BIG:
        return np.zeros(shape, dtype)
    tmp = TMP
    tmp.mkdir(parents=True, exist_ok=True)
    fh = tempfile.NamedTemporaryFile(dir=tmp, suffix=".npy", delete=False)
    fh.close()
    a = np.lib.format.open_memmap(fh.name, mode="w+", dtype=dtype, shape=tuple(shape))
    try:
        os.unlink(fh.name)
    except OSError:
        pass
    return a


class _CPU:
    """Stands where app.py expects cupy: tells app._gpu_or_cpu to take the strip path."""


# The classes of a strip are smoothed in parallel threads (scipy.ndimage runs without the GIL), and their
# results are folded in class order exactly as the serial loop does, so the numbers do not change.  Each
# thread holds a few float32 copies of the strip, so the strips are cut smaller to keep the peak memory
# where the serial pass had it.
WORKERS = max(1, min(os.cpu_count() or 1, 8))


def _rows(Z, X, per_worker=0):
    strip = STRIP * 20.0 / (20.0 + per_worker * WORKERS) if per_worker and WORKERS > 1 else STRIP
    return max(16, int(strip // max(1, Z * X)))


def _in_order(fn, items):
    """fn over items in WORKERS threads, yielded in the items' order; at most WORKERS results held at once."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if WORKERS == 1:
        for it in items:
            yield it, fn(it)
        return
    with ThreadPoolExecutor(WORKERS) as ex:
        for i in range(0, len(items), WORKERS):
            batch = items[i:i + WORKERS]
            for it, r in zip(batch, ex.map(fn, batch)):
                yield it, r


def _lead_by_class_strips(vol, present, sig, progress, total, cp):
    """app._lead_by_class_impl's GPU loop, on the CPU."""
    from scipy.ndimage import gaussian_filter
    Z, Y, X = vol.shape
    lead = _big(vol.shape, np.uint8)
    val = _big(vol.shape, np.uint8)
    halo = int(4.0 * sig[1] + 0.5) + 2
    strips = list(app._strips(Y, _rows(Z, X, per_worker=8), halo))
    for n, (y0, y1, a0, a1) in enumerate(strips):
        v = vol[:, a0:a1]
        best = np.zeros(v.shape, np.float32)
        ld = np.zeros(v.shape, np.uint8)
        smooth = lambda k, v=v: gaussian_filter((v == k).astype(np.float32), sig, mode="nearest")  # noqa: E731
        for k, occ in _in_order(smooth, present):
            win = occ > best
            best[win] = occ[win]
            ld[win] = k
            del occ, win
        c = np.s_[:, y0 - a0:y1 - a0]
        lead[:, y0:y1] = ld[c]
        val[:, y0:y1] = np.clip(np.rint(best[c] * 255.0), 0, 255).astype(np.uint8)
        del v, best, ld
        if progress:
            progress(int(len(present) * (n + 1) / len(strips)), total)
    return lead, val


def _pairs_strips(cp, ids, reach, big):
    """app._gpu_pairs, on the CPU."""
    from scipy.ndimage import maximum_filter, minimum_filter
    Z, Y, X = ids.shape
    halo = reach[1] // 2 + 1
    out = []
    for y0, y1, a0, a1 in app._strips(Y, _rows(Z, X), halo):
        a = ids[:, a0:a1]
        for filt, fill in ((maximum_filter, 0), (minimum_filter, big)):
            src = a if fill == 0 else np.where(a > 0, a, big).astype(np.int32)
            b = filt(src, size=reach, mode="constant", cval=fill)[:, y0 - a0:y1 - a0]
            c = a[:, y0 - a0:y1 - a0]
            m = (c > 0) & (b != c) & (b > 0) & (b != big)
            if m.any():
                lo_ = np.minimum(c[m], b[m]).astype(np.int64)
                hi_ = np.maximum(c[m], b[m]).astype(np.int64)
                out.append(np.unique((lo_ << 32) | hi_))
            del src, b, c, m
    return out


def _slots_strips(cp, ids, slot8, lut, K, sig, reach, progress):
    """app._gpu_slots, on the CPU."""
    from scipy.ndimage import gaussian_filter, maximum_filter
    Z, Y, X = ids.shape
    halo = max(int(4.0 * sig[1] + 0.5), reach[1] // 2) + 2
    lead = _big(ids.shape, np.uint8)
    val = _big(ids.shape, np.uint8)
    cls = _big(ids.shape, np.uint8)
    strips = list(app._strips(Y, _rows(Z, X, per_worker=16), halo))
    for n, (y0, y1, a0, a1) in enumerate(strips):
        a = ids[:, a0:a1]
        sv = slot8[a]
        best = np.zeros(a.shape, np.float32)
        ld = np.zeros(a.shape, np.uint8)
        cl = np.zeros(a.shape, np.uint8)

        def smooth(k, a=a, sv=sv):
            m = sv == k
            occ = gaussian_filter(m.astype(np.float32), sig, mode="nearest")
            # the one cell of this slot within reach (slot 0 is the empty background)
            near = maximum_filter(np.where(m, a, 0), size=reach, mode="constant", cval=0) if k > 0 else None
            return occ, near

        for k, (occ, near) in _in_order(smooth, range(K + 1)):
            win = occ > best
            best[win] = occ[win]
            ld[win] = k
            if near is not None:
                cl[win] = lut[near[win]]
            del occ, near, win
        c = np.s_[:, y0 - a0:y1 - a0]
        lead[:, y0:y1] = ld[c]
        val[:, y0:y1] = np.clip(np.rint(best[c] * 255.0), 0, 255).astype(np.uint8)
        cls[:, y0:y1] = cl[c]
        del a, sv, best, ld, cl
        if progress:
            progress(2 + int(16 * (n + 1) / len(strips)), 20)
    return lead, val, cls


class _Above0:
    """`a > 0`, one z plane at a time: the occupancy a solid view's tissue envelope is made from, without
    a full-size boolean copy of the volume."""

    def __init__(self, a):
        self.a = a

    def __getitem__(self, z):
        return self.a[z] > 0


def _read_blocks(a, fn, dtype):
    """A (z, y, x) zarr array, read and converted a band of chunk rows at a time: the SSAF-T maps are
    written as int32, four times the uint8 they are drawn as, and read whole they are 12 GB for a large
    section.  The bands follow the chunks (which span every z), so each chunk is decoded once."""
    out = _big(a.shape, dtype)
    step = int(a.chunks[1]) if len(getattr(a, "chunks", ())) == 3 else 1024
    for y0 in range(0, a.shape[1], step):
        out[:, y0:y0 + step] = fn(np.asarray(a[:, y0:y0 + step]))
    return out


def _read_rows(a, fn, dtype):
    """_read_blocks for a flat (y, x) label element: a (1, y, x) volume, read a band of rows at a time."""
    out = _big((1,) + tuple(a.shape), dtype)
    step = int(a.chunks[0]) if len(getattr(a, "chunks", ())) == 2 else 1024
    for y0 in range(0, a.shape[0], step):
        out[0, y0:y0 + step] = fn(np.asarray(a[y0:y0 + step]))
    return out


def _wrap_colour(k):
    """The k-th of 255 colours for cell ids no table names: hues a golden angle apart, so neighbouring ids
    (touching cells, often) differ."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((k * 0.381966) % 1.0, 0.55 + 0.35 * (k % 3) / 2, 0.95 - 0.25 * (k % 2))
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def label_volume(spec):
    """app.label_volume, reading a SpatialData label map plane by plane; the same volume."""
    if spec["type"] != "sdata_labels":
        return _label_volume(spec)
    g = app._open(str(spec["store"]))
    node = g["labels"][spec["element"]]
    lv = spec.get("level") or app._levels(node)[0]
    a = node[lv]
    if len(a.shape) not in (2, 3):
        return _label_volume(spec)
    read = _read_blocks if len(a.shape) == 3 else _read_rows
    top = [0]
    if spec.get("table"):
        # app sizes the lookup to the largest id in the map; an id past the table's cells maps to 0 either way
        lut = app._instance_lut(g, spec, 0)
        last = len(lut) - 1
        arr = read(a, lambda s: np.where(s > last, 0, lut[np.clip(s, 0, last)]), np.uint8)
    else:
        # values past a byte are cell ids no table names: drawn in repeating colours, as the site's viewer
        # draws them (local/layers.js), so that at least the cells read as separate; a map's are kept as they are
        def to8(s):
            top[0] = max(top[0], int(s.max(initial=0)))
            return np.where(s > 0, (s - 1) % 255 + 1, 0)
        arr = read(a, to8, np.uint8)
    if top[0] > 254:
        n = _shown(spec["element"])
        names, colors = [n] * 255, [_wrap_colour(k) for k in range(255)]
    else:
        names, colors = app._palette(spec)
    vol, vox, org = app._align(arr, app._element_matrix(node, lv), app._um_per_unit(g, spec), spec, resample=True)
    return app._z_trim(dict(vol=vol, names=names or [], colors=colors or [], voxel_um=vox, origin_um=org,
                            name=f"labels/{spec['element']}"), "vol", spec.get("z_trim"))


def instance_volume(spec):
    """app.instance_volume, reading the cell ids plane by plane; the same volume."""
    g = app._open(str(spec["store"]))
    node = g["labels"][spec["element"]]
    lv = spec.get("level") or app._levels(node)[0]
    a = node[lv]
    if len(a.shape) != 3:
        return _instance_volume(spec)
    arr = _read_blocks(a, lambda s: np.where(s > 0, s, 0), np.int32)
    lut = app._instance_lut(g, spec, int(arr.max(initial=0)))
    names, colors = app._palette(spec)
    ids, vox, org = app._align(arr, app._element_matrix(node, lv), app._um_per_unit(g, spec), spec, resample=True)
    return app._z_trim(dict(ids=ids, lut=lut, names=names or [], colors=colors or [], voxel_um=vox,
                            origin_um=org, name=f"labels/{spec['element']}"), "ids", spec.get("z_trim"))


def _write_solid_tiles(key, chans, occupied, src, sigma_xy_um, sigma_z_um, classes, tile, ov_factor):
    """app._write_solid_tiles, plane by plane and tile by tile; the same files, less the tissue envelope
    (a strongly smoothed shell of the whole section, which the app does not offer).

    The original holds a boolean of the whole volume for the empty-tile test, two padded copies of it for
    the overview and a float32 copy for the tissue envelope (12 GB for a 3 GB map)."""
    import gzip
    from scipy.ndimage import gaussian_filter
    CACHE = app.CACHE
    lead, val = chans[0], chans[1]
    Z, Y, X = lead.shape
    vx, vy, vz = src["voxel_um"]
    T, tiles, nbytes = int(tile), [], 0
    for ty in range(0, Y, T):
        for tx in range(0, X, T):
            sl = np.s_[:, ty:ty + T, tx:tx + T]
            surfaced = (lead[sl] > 0) & (val[sl] >= 128)
            if len(chans) > 2:
                surfaced &= chans[2][sl] > 0
            t = dict(x0=tx, y0=ty, w=int(lead[sl].shape[2]), h=int(lead[sl].shape[1]), empty=not bool(surfaced.any()))
            del surfaced
            if not t["empty"]:
                fn = f"{key}_s{ty // T}_{tx // T}.bin.gz"
                with gzip.open(CACHE / fn, "wb", compresslevel=6) as fh:
                    fh.write(np.ascontiguousarray(np.stack([c[sl] for c in chans], -1)).tobytes())
                t.update(file=f"cache/{fn}", bytes=(CACHE / fn).stat().st_size)
                nbytes += t["bytes"]
            tiles.append(t)
    # overview: in each block the voxel where a real key is strongest, background elsewhere
    f = int(ov_factor)
    by, bx = -(-Y // f), -(-X // f)
    ov = [np.zeros((Z, by, bx), np.uint8) for _ in chans]
    for z in range(Z):
        blocks = []
        for i, c in enumerate(chans):
            pad = np.zeros((by * f, bx * f), np.uint8)
            pad[:Y, :X] = np.where(lead[z] > 0, c[z], 0) if i == 1 else c[z]
            blocks.append(pad.reshape(by, f, bx, f).transpose(0, 2, 1, 3).reshape(by, bx, f * f))
        idx = blocks[1].argmax(-1)[..., None]
        for i, bl in enumerate(blocks):
            ov[i][z] = np.take_along_axis(bl, idx, -1)[..., 0]
    ov[0] = np.where(ov[1] >= 128, ov[0], 0).astype(np.uint8)
    ofn = f"{key}_sov.bin.gz"
    with gzip.open(CACHE / ofn, "wb", compresslevel=6) as fh:
        fh.write(np.ascontiguousarray(np.stack(ov, -1)).tobytes())
    del ov
    return dict(kind="solidtiles", channels=len(chans), name=src["name"], dims=dict(x=X, y=Y, z=Z),
                voxel_um=list(src["voxel_um"]), origin_um=list(src["origin_um"]), tile=T, tiles=tiles,
                classes=classes, overview=dict(file=f"cache/{ofn}", factor=f, w=bx, h=by),
                sigma_um=[float(sigma_xy_um), float(sigma_xy_um), float(sigma_z_um)], iso=0.5,
                bytes=nbytes + (CACHE / ofn).stat().st_size)


def solid_tiles_for(spec, sigma_xy_um=1.0, sigma_z_um=1.0, tile=2048, ov_factor=4, progress=None):
    """app.solid_tiles_for, handing the envelope the map's occupancy plane by plane; the same files."""
    tile = app._auto_tile(spec, 2, tile)
    key = app._key(**app._label_key(spec), sxy=float(sigma_xy_um), sz=float(sigma_z_um), tile=int(tile),
                   o=int(ov_factor), v="stiles1")
    cached = app.CACHE / f"{key}_stiles.json"
    if cached.exists():
        return json.loads(cached.read_text())
    src = app.label_volume(spec)
    vol, names, colors = src["vol"], src["names"], src["colors"]
    Z, Y, X = vol.shape
    vx, vy, vz = src["voxel_um"]
    counts = app.np.bincount(vol.ravel(), minlength=256)
    present = np.nonzero(counts)[0]
    total = len(present) + 3
    sig = (sigma_z_um / vz if Z > 1 else 0.0, sigma_xy_um / vy, sigma_xy_um / vx)
    lead, val = app._lead_by_class(vol, present, sig, progress, total)
    classes = [dict(id=k, value=k - 1, name=names[k - 1] if k - 1 < len(names) else f"id {k}",
                    color=colors[k - 1] if k - 1 < len(colors) else "#cccccc",
                    voxels=int(counts[k])) for k in range(1, 256) if counts[k] > 0]
    meta = _write_solid_tiles(key, [lead, val], _Above0(vol), src, sigma_xy_um, sigma_z_um, classes, tile, ov_factor)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(total, total)
    return meta


def _instance_finish(key, cached, spec, src, ids, lead, val, cls, counts, names, colors, sigma_xy_um,
                     sigma_z_um, tile, ov_factor, K, present, e, progress):
    """app._instance_finish, with the occupancy plane by plane; the same files."""
    classes = [dict(id=k, value=k - 1, name=names[k - 1] if k - 1 < len(names) else f"id {k}",
                    color=colors[k - 1] if k - 1 < len(colors) else "#cccccc",
                    voxels=int(counts[k])) for k in range(1, 256) if counts[k] > 0]
    meta = _write_solid_tiles(key, [lead, val, cls], _Above0(ids), src, sigma_xy_um, sigma_z_um, classes, tile,
                              ov_factor)
    meta.update(n_slots=K, n_cells=int(present.sum()), n_touching_pairs=int(len(e)))
    cached.write_text(json.dumps(meta))
    if progress:
        progress(20, 20)
    return meta


def _obs_column(g, table, column, key=None):
    """app._obs_column, also reading anndata's newer `nullable-string-array` columns (a group of values
    and a mask, which anndata >= 0.11 writes for strings) besides categoricals and plain string arrays."""
    import zarr
    tb = g["tables"][table]
    obs = tb["obs"]

    def strings(a):
        return np.array([str(x) for x in np.asarray(a).tolist()], dtype=object)

    def col(name):
        node = obs[name]
        if isinstance(node, zarr.Group):
            if "categories" in node:                         # categorical: codes + categories
                cats = strings(node["categories"][:])
                codes = np.asarray(node["codes"][:])
                return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
            vals = strings(node["values"][:])               # nullable: values + mask
            mask = np.asarray(node["mask"][:]).astype(bool) if "mask" in node else np.zeros(len(vals), bool)
            return np.where(mask, "", vals)
        return strings(node[:])
    return col(key or dict(tb.attrs)["instance_key"]), col(column)


app._obs_column = _obs_column


def _obs_palette(spec):
    """(names, colours) from a table that carries no SSAF-T palette: the categories of the column the layer
    is coloured by, with the `<column>_colors` AnnData keeps beside it in `uns` (as SSAF-T_web's viewer
    reads them).  A vendor's table, or one from another analysis, names its cells that way."""
    import zarr
    name = spec.get("palette_table") or spec.get("table")
    col = spec.get("color_key", "cellman_annotation")
    if not name:
        return None, None
    g = app._open(str(spec["store"]))
    if "tables" not in g or name not in g["tables"]:
        return None, None
    tb = g["tables"][name]
    if col not in tb["obs"]:
        return None, None
    node = tb["obs"][col]
    if not isinstance(node, zarr.Group) or "categories" not in node:
        return None, None                                    # a plain string column has no categories
    names = [str(x) for x in np.asarray(node["categories"][:])]
    uns = tb.get("uns")
    colors = None
    if uns is not None and f"{col}_colors" in set(uns.array_keys()):
        colors = [str(x) for x in np.asarray(uns[f"{col}_colors"][:])]
    return names, (colors[:len(names)] if colors and len(colors) >= len(names) else None)


_palette_app = app._palette


def _palette(spec):
    """app._palette, falling back to the table's own categorical column when it has no SSAF-T palette."""
    names, colors = _palette_app(spec)
    if names or spec.get("palette_store"):
        return names, colors
    return _obs_palette(spec)


app._palette = _palette


_rgb_tiles_for = app.rgb_tiles_for


def rgb_tiles_for(spec, tile=2048, ov_factor=8, progress=None):
    """The picture's tiles, and whether they are grey: a grey picture (a fluorescence channel, its tiles grey
    JPEGs) is uploaded by the page as one byte a pixel instead of four, so it takes a quarter of the GPU
    memory and more of it stays at full resolution.  Like `color`, a display choice: never in the cache."""
    return dict(_rgb_tiles(spec, tile=tile, ov_factor=ov_factor, progress=progress), mono=bool(spec.get("_mono")))


def _rgb_tiles(spec, tile=2048, ov_factor=8, progress=None):
    """app.rgb_tiles_for, without the resampling when there is nothing to resample.

    app resamples every picture onto the µm lattice through its transform (bilinear, through the inverse of
    the affine): right for a rotated or rescaled image, but for an image whose pixels already ARE that lattice
    (no rotation, one pixel = `um_per_px`) the bilinear lookup at each pixel's own centre returns the pixel
    itself.  That pass was 35 of the 45 s a Visium HD H&E took; here those pixels are copied (and flipped
    where the transform flips).  Anything else goes to app's own resampling."""
    import gzip  # noqa: F401
    from PIL import Image
    um = float(spec.get("um_per_px", 1.0))
    key = app._key(t="rgb", p=str(spec["store"]), n=spec["element"], l=spec.get("level"), um=um,
                   tile=int(tile), o=int(ov_factor), u=spec.get("um_per_unit"),
                   z0=spec.get("z0_um"), zu=spec.get("z_um"), bg=spec.get("background_key"),
                   **({} if spec.get("plane") is None else dict(pl=spec["plane"])),
                   **({} if spec.get("channel") is None else dict(c=spec["channel"])),
                   mt=app._mtime(spec), v=3)
    cached = app.CACHE / f"{key}_rgb.json"
    if cached.exists():
        return json.loads(cached.read_text())
    g = app._open(str(spec["store"]))
    node = g["images"][spec["element"]]
    lv = spec.get("level") or app._levels(node)[0]
    S = app._element_matrix(node, lv)
    S[:2, :] *= app._um_per_unit(g, spec)
    L = S[:2, :2]
    axis_aligned = abs(L[0, 1]) < 1e-9 * abs(L[0, 0]) and abs(L[1, 0]) < 1e-9 * abs(L[1, 1])
    if not (axis_aligned and abs(abs(L[0, 0]) - um) < 1e-6 * um and abs(abs(L[1, 1]) - um) < 1e-6 * um):
        return _rgb_tiles_for(spec, tile=tile, ov_factor=ov_factor, progress=progress)
    a = node[lv]
    axn = app._axes(app._multiscale(node))
    sel = [slice(None)] * len(axn)
    if "z" in axn and spec.get("plane") is not None:
        nz = a.shape[axn.index("z")]
        sel[axn.index("z")] = max(0, min(nz - 1, int(spec["plane"])))
    if "c" in axn and spec.get("channel") is not None:
        sel[axn.index("c")] = int(spec["channel"])
    arr = np.asarray(a[tuple(sel)])
    if arr.ndim == 2:
        arr = arr[None]
    nc, H, W = arr.shape
    if arr.dtype != np.uint8:                                # 16-bit fluorescence: stretched as app does
        u8 = np.empty(arr.shape, np.uint8)
        for c in range(nc):
            s8 = arr[c, ::8, ::8]
            lo_v, hi_v = float(np.percentile(s8, 1.0)), float(np.percentile(s8, 99.8))
            g8 = 255.0 / max(hi_v - lo_v, 1e-6)
            for r in range(0, H, 4096):
                u8[c, r:r + 4096] = np.clip((arr[c, r:r + 4096].astype(np.float32) - lo_v) * g8,
                                            0, 255).astype(np.uint8)
        arr = u8
    bg = spec.get("background_key") or ("white" if nc == 3 else "dark")
    out = np.moveaxis(arr, 0, -1)                            # (H, W, nc)
    if L[0, 0] < 0:
        out = out[:, ::-1]
    if L[1, 1] < 0:
        out = out[::-1]
    out = np.ascontiguousarray(out)
    corners = np.array([[x, y, 0, 1] for x in (0, W) for y in (0, H)], float)
    pts = (S @ corners.T).T[:, :2]
    x0, y0 = pts.min(0)
    if nc == 1:
        out = out[..., 0]
    OH, OW = out.shape[:2]
    n_tiles = (-(-OH // tile)) * (-(-OW // tile))
    total = n_tiles + 2
    if progress:
        progress(1, total)
    T, tiles, nbytes, i = int(tile), [], 0, 1
    for ty in range(0, OH, T):
        for tx in range(0, OW, T):
            blk = out[ty:ty + T, tx:tx + T]
            fn = f"{key}_t{ty // T}_{tx // T}.jpg"
            Image.fromarray(blk).save(app.CACHE / fn, quality=90)
            tiles.append(dict(x0=tx, y0=ty, w=int(blk.shape[1]), h=int(blk.shape[0]), empty=False,
                              file=f"cache/{fn}", bytes=(app.CACHE / fn).stat().st_size))
            nbytes += tiles[-1]["bytes"]
            i += 1
            if progress:
                progress(i, total)
    f = int(ov_factor)
    ow, oh = -(-OW // f), -(-OH // f)
    ofn = f"{key}_ov.jpg"
    Image.fromarray(out).resize((ow, oh), Image.BOX).save(app.CACHE / ofn, quality=88)
    meta = dict(kind="rgb", tiled=True, bg_key=bg, name=spec["element"], dims=dict(x=OW, y=OH, z=1),
                voxel_um=[um, um, float(spec.get("z_um", 1.0))],
                origin_um=[float(x0), float(y0), float(spec.get("z0_um", 0.0))],
                tile=T, tiles=tiles, overview=dict(file=f"cache/{ofn}", factor=f, w=ow, h=oh),
                bytes=nbytes + (app.CACHE / ofn).stat().st_size)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(total, total)
    return meta


app.rgb_tiles_for = rgb_tiles_for


ARRAY_FRAMES = {}


def array_frame(g, store):
    """2x3 affine from SSAF-T's sample coordinates on a Visium HD store to its `global` frame, or None.

    SSAF-T writes its signal samples (tables/ssaft_downsampled_vectors, obsm['spatial']) in the bin array's
    own µm: array_col x bin size, array_row x bin size; `global` is the full-resolution image's pixels, with y
    the other way up.  The bins table keeps both for every bin (obs array_row / array_col, obsm['spatial']),
    so the relation is fitted from it (least squares; residual well under a pixel)."""
    key = str(store)
    if key in ARRAY_FRAMES:
        return ARRAY_FRAMES[key]
    out = None
    try:
        for name in (sorted(g["tables"].group_keys()) if "tables" in g else []):
            m = re.search(r"square_(\d+)um$", name)
            t = g["tables"][name]
            if not m or "obs" not in t or "obsm" not in t or "spatial" not in t["obsm"]:
                continue
            obs = t["obs"]
            if "array_row" not in obs or "array_col" not in obs:
                continue
            bin_um = int(m.group(1))
            n = obs["array_row"].shape[0]
            pick = np.unique(np.linspace(0, n - 1, min(n, 200000)).astype(np.int64))
            r = np.asarray(obs["array_row"][:], float)[pick] * bin_um
            c = np.asarray(obs["array_col"][:], float)[pick] * bin_um
            xy = np.asarray(t["obsm"]["spatial"][:], float)[pick]
            A = np.c_[c, r, np.ones_like(r)]
            fx = np.linalg.lstsq(A, xy[:, 0], rcond=None)[0]
            fy = np.linalg.lstsq(A, xy[:, 1], rcond=None)[0]
            out = np.vstack([fx, fy])
            print(f"array frame of {name}: x = {fx.round(4).tolist()}, y = {fy.round(4).tolist()}", flush=True)
            break
    except Exception:                                            # noqa: BLE001
        traceback.print_exc()
        out = None
    ARRAY_FRAMES[key] = out
    return out


GRID_SETUP = {}


def _grid_setup(spec):
    """What every gene's image of one store shares, worked out once: the lattice (app._grid_lattice reads the
    whole cell-type map for it), and the lattice bin each signal sample falls in (-1 outside).  Per gene it
    was 0.2 s of the 0.2 s a gene took; what is left per gene is its own column."""
    import json as _json
    k = _json.dumps({a: spec.get(a) for a in ("store", "element", "um_per_unit", "z_trim")}, sort_keys=True)
    st = GRID_SETUP.get(k)
    if st is not None:
        return st
    g = app._open(str(spec["store"]))
    t = app._vector_table(g)
    if t is None:
        raise KeyError("this dataset has no binned expression")
    names = [str(x) for x in np.asarray(g[f"tables/{t}/var"]["gene"][:]).tolist()]
    lat = app._grid_lattice(spec)
    xy = np.asarray(g[f"tables/{t}/obsm/spatial"][:], np.float64)
    T = array_frame(g, spec["store"])
    if T is not None:                            # Visium HD: the bin array's µm -> global -> µm
        upu = app._um_per_unit(g, spec)
        p = (xy @ T[:, :2].T + T[:, 2]) * upu
        xs, ys = p[:, 0], p[:, 1]
    else:                                        # app's own reading: µm or global units, whichever fits
        k2 = app._sample_scale(g, spec, xy, lat)
        xs, ys = xy[:, 0] * k2, xy[:, 1] * k2
    ix = np.floor((xs - lat["ox"]) / lat["vx"]).astype(np.int64)
    iy = np.floor((ys - lat["oy"]) / lat["vy"]).astype(np.int64)
    inside = (ix >= 0) & (iy >= 0) & (ix < lat["w"]) & (iy < lat["h"])
    bins = np.where(inside, iy * lat["w"] + ix, -1)
    st = dict(names=names, index={n: i for i, n in enumerate(names)}, table=t, lat=lat, bins=bins,
              mt=app._mtime(spec))
    GRID_SETUP[k] = st
    return st


def gene_grid(spec, gene):
    """app.gene_grid: one gene's signal as a uint8 image on the cell-type map's lattice, the strongest sample
    in each bin, scaled to its 99.5th percentile -- the same image, with the store's shared work done once
    (_grid_setup), and its cached file keyed also by the µm per unit it was placed with."""
    st = _grid_setup(spec)
    if gene not in st["index"]:
        raise KeyError(gene)
    lat = st["lat"]
    key = app._key(t="ggrid", p=str(spec["store"]), n=spec["element"], g=gene, mt=st["mt"],
                   m=app.GRID_MAX, u=spec.get("um_per_unit"), v="array1")
    cached = app.CACHE / f"{key}.bin.gz"
    if cached.exists():
        return f"cache/{cached.name}", lat
    g = app._open(str(spec["store"]))
    col = np.asarray(g[f"tables/{st['table']}/X"][:, st["index"][gene]], np.float32)
    ok = (st["bins"] >= 0) & (col > 0)
    grid = np.zeros(lat["h"] * lat["w"], np.float32)
    np.maximum.at(grid, st["bins"][ok], col[ok])
    hi = float(np.percentile(grid[grid > 0], 99.5)) if (grid > 0).any() else 1.0
    out = np.clip(grid * (255.0 / max(hi, 1e-9)), 0, 255).astype(np.uint8)
    import gzip
    with gzip.open(cached, "wb", compresslevel=6) as fh:
        fh.write(out.tobytes())
    return f"cache/{cached.name}", lat


app.gene_grid = gene_grid
_label_volume, _instance_volume = app.label_volume, app.instance_volume
app.label_volume = label_volume
app.instance_volume = instance_volume
app._write_solid_tiles = _write_solid_tiles
app.solid_tiles_for = solid_tiles_for
app._instance_finish = _instance_finish
app._gpu = lambda: _CPU
app._lead_by_class_impl = _lead_by_class_strips
app._gpu_pairs = _pairs_strips
app._gpu_slots = _slots_strips


# ----------------------------------------------------------------- http
# ----------------------------------------------------------------- the cell under the cursor
# The page names the cell type under the cursor itself; where the store has SSAF-T cells, it then asks for
# that cell's location: the centre the cell table keeps in obsm['spatial'] (x, y, z µm).  The cells' own
# outlines are indexed once per store, in µm through the same transform the outline layers are drawn with:
# the 2D outlines (the 2D slice) and the plane-by-plane 3D outlines (the 3D view, the plane nearest z).
CELLS = {}
CELLS_LOCK = threading.Lock()


def _outline_index(g, spec):
    """(STRtree, cell ids, z per outline or None, plane step) of one SSAF-T outline element, in µm."""
    import geopandas as gpd
    import shapely
    node = g["shapes"][spec["element"]]
    df = gpd.read_parquet(Path(spec["store"]) / "shapes" / spec["element"] / "shapes.parquet")
    S = app._shape_matrix(node)
    S[:2, :] *= app._um_per_unit(g, spec)
    A, t = S[:2, :2], S[:2, 3]
    geoms = shapely.transform(np.asarray(df.geometry.values), lambda c: c @ A.T + t)
    idc = spec.get("id_column")
    ids = np.asarray((df[idc] if idc and idc in df.columns else df.index).astype(str))
    zc = spec.get("z_column")
    z = df[zc].to_numpy(float) if zc and zc in df.columns else None
    step = float(spec.get("z_step_um") or 1.0)
    return shapely.STRtree(geoms), ids, z, step


def cell_index(did):
    """The dataset's SSAF-T cells for cell_at, built on first use; None when it has none, or its cell table
    keeps no centre (obsm['spatial'])."""
    with CELLS_LOCK:
        if did in CELLS:
            return CELLS[did]
        d = app.CATALOG.get(did)
        out = None
        try:
            lay = d["layers"] if d else {}
            two = next((v for v in lay.values() if v.get("element") == "ssaft_cell_boundaries_2d"), None)
            three = next((v for v in lay.values() if v.get("element") == "ssaft_cell_boundaries_3d"), None)
            spec = two or three
            table = spec and spec.get("table")
            if spec and table:
                g = app._open(str(spec["store"]))
                tb = g["tables"][table]
                if "obsm" in tb and "spatial" in tb["obsm"]:
                    ids, types = app._obs_column(g, table, "cellman_annotation", key=spec.get("table_key") or "cell_id")
                    out = dict(row={str(i): k for k, i in enumerate(ids)}, types=types,
                               centre=np.asarray(tb["obsm"]["spatial"][:], dtype=float),
                               two=_outline_index(g, two) if two else None,
                               three=_outline_index(g, three) if three else None)
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
            out = None
        CELLS[did] = out
        return out


def _ring(geom, cap=256):
    """The outline the page tests the cursor against: the exterior of the (largest) polygon, in µm, thinned
    to at most `cap` vertices."""
    import shapely
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or geom.is_empty:
        return None
    xy = np.asarray(geom.exterior.coords)[:-1]
    if len(xy) > cap:
        xy = xy[np.linspace(0, len(xy) - 1, cap).astype(int)]
    return np.round(xy, 2).tolist()


def cell_at(did, x, y, z, view):
    """{cell_type, location, id, ring, z, step} of the SSAF-T cell at (x, y, z) µm, or None.

    `ring` is the cell's outline (in 3D, on the plane the cursor is at, `z`, `step` thick): while the cursor
    stays inside it the page answers itself, without asking again for every move within the same cell."""
    import shapely
    ix = cell_index(did)
    if ix is None:
        return None
    p = shapely.Point(x, y)
    found, ring, zc, dz = None, None, None, None
    for which in (("three", "two") if view == "3d" else ("two", "three")):
        o = ix[which]
        if o is None:
            continue
        tree, ids, zs, step = o
        hit = tree.query(p, predicate="intersects")
        if not len(hit):
            continue
        if zs is not None and z is not None:                     # the plane nearest the cursor's depth
            dz = np.abs(zs[hit] - z)
            hit = hit[dz <= max(step, float(dz.min()))]
            hit = hit[np.argsort(np.abs(zs[hit] - z), kind="stable")]
        found = str(ids[hit[0]])
        ring = _ring(tree.geometries[hit[0]])
        if zs is not None:
            zc, dz = float(zs[hit[0]]), step
        break
    k = ix["row"].get(found) if found else None
    if k is None:
        return None
    c = ix["centre"][k]
    return dict(cell_type=str(ix["types"][k]), location=[round(float(v), 1) for v in c[:3]], id=found,
                ring=ring, z=zc, step=dz)


class Handler(app.Handler):
    """The site's handler, plus /api/open and the page's desktop script; the cache is served from the
    application-data folder rather than from beside the page."""

    def translate_path(self, path):
        p = posixpath.normpath(unquote(urlparse(path).path))
        if p.startswith("/cache/") and "/" not in p[len("/cache/"):]:
            return str(app.CACHE / p[len("/cache/"):])
        return super().translate_path(path)

    def _static_allowed(self, path):
        p = posixpath.normpath(unquote(path))
        if p in ("/desktop.js", "/desktop.css"):
            return True
        return super()._static_allowed(path)

    def _page(self):
        """store.html with the desktop script after it: the page is the site's, unchanged."""
        html = (WEB / "store.html").read_text(encoding="utf-8")
        tag = '<link rel="stylesheet" href="desktop.css">\n<script src="desktop.js"></script>\n'
        html = html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag
        b = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self._gate():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/store.html"):
            return self._page()
        if u.path == "/api/config":
            return self._json(dict(only=None, min_level=0, catalog=True, root=None, desktop=True))
        if u.path == "/api/open":
            try:
                did = open_store(q.get("path", ""))
                return self._json(dict(id=did, title=app.CATALOG[did]["meta"]["title"]))
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                return self._json(dict(error=str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"),
                                  400)
        if u.path == "/api/cell_at":
            try:
                z = q.get("z")
                return self._json(dict(cell=cell_at(q.get("id", ""), float(q["x"]), float(q["y"]),
                                                    None if z in (None, "") else float(z), q.get("view", "2d"))))
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                return self._json(dict(error=f"{type(e).__name__}: {e}"), 400)
        # a layer the page asks for that is not cut yet goes to the front of the queue
        if u.path in ("/api/dataset_layer", "/api/dataset_solid"):
            kind = "tiles" if u.path == "/api/dataset_layer" else "solid"
            st = (app.TILES if kind == "tiles" else app.SOLID).get((q.get("id", ""), q.get("key", "")))
            if st is not None and st["status"] == "building":
                JOBS.ask((kind, q["id"], q["key"]))
        return super().do_GET()


def _watch_stdin():
    """The app holds our stdin open; when it goes (the app quit or crashed), so do we."""
    try:
        while sys.stdin.buffer.read(1 << 16):
            pass
    except Exception:                                            # noqa: BLE001
        pass
    os._exit(0)


def main():
    global WEB, CACHE_ROOT, TMP
    # a crash in native code (a segfault) otherwise ends the server with nothing in its log: this writes
    # every thread's Python stack to stderr, which the app keeps in logs/server.log
    import faulthandler
    faulthandler.enable(all_threads=True)
    for s in (sys.stdout, sys.stderr):                           # Windows consoles are not UTF-8
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:                                        # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=0, help="0: any free port (printed on the READY line)")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--web", default=None, help="the page: store.html, local/, desktop.js")
    ap.add_argument("--cache", default=None, help="derived tiles, solids and gene indexes")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--watch-stdin", action="store_true", help="exit when stdin closes")
    ap.add_argument("--open", action="append", default=[], help="open this store at start-up")
    ap.add_argument("--build-genes", nargs=2, metavar=("STORE", "ELEMENT"), default=None)
    a = ap.parse_args()
    # app.py imports these inside its functions; without them a store would fail only once opened.  A frozen
    # build carries them, and importing them all here would only slow its start (on Windows each DLL loaded is
    # scanned on a first launch), so only a Python run from source is checked.
    missing = []
    for mod in () if FROZEN else ("zarr", "scipy.ndimage", "pyarrow.dataset", "geopandas", "shapely", "PIL.Image"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod.split(".")[0])
    if missing:
        raise SystemExit(f"{sys.executable} lacks {', '.join(missing)}: run it in the ssaft-app environment "
                         "(environment.yml), or set SSAFT_PYTHON to that environment's python")

    cache = Path(a.cache or (Path.home() / ".cache" / "ssaft-viewer" / "cache")).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    app.CACHE = cache                            # the gene index child is handed its store's folder itself
    TMP = cache.parent / "tmp"
    if a.web:
        WEB = Path(a.web).resolve()
    elif FROZEN:                                                 # PyInstaller: data beside the code
        WEB = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "web"
    app.SERVE = WEB
    if a.build_genes:                                            # the child of _genes_subprocess
        app.gene_index_for(*a.build_genes, progress=lambda i, n: print(f"progress {i} {n}", flush=True))
        raise SystemExit(0)
    CACHE_ROOT = cache
    shutil.rmtree(TMP, ignore_errors=True)       # a previous run's file-backed arrays (Windows keeps them)
    if a.token_file:
        app.TOKEN = Path(a.token_file).read_text().strip()
        if len(app.TOKEN) < 24:
            raise SystemExit("--token-file: the secret must be at least 24 characters")
    app.CATALOG = {}
    app.ROOT = Path(Path.home().anchor or "/")
    threading.Thread(target=JOBS.run, daemon=True, name="build").start()
    if a.watch_stdin:
        threading.Thread(target=_watch_stdin, daemon=True, name="stdin").start()
    for p in a.open:
        open_store(p)
    handler = functools.partial(Handler, directory=str(WEB))
    with app.Server((a.bind, a.port), handler) as srv:
        port = srv.server_address[1]
        print(f"serving {WEB} (cache {cache}) at http://{a.bind}:{port}", flush=True)
        print(f"SSAFT_SERVER_READY {port}", flush=True)
        srv.serve_forever()


if __name__ == "__main__":
    main()
