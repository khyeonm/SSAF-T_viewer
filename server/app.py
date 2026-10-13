#!/usr/bin/env python
"""Backend for the 3D viewer: open a SpatialData zarr, list what is in it, render a choice.

    python app.py --port 8766 --root /bce/groups/pnucolab/analysis/l202255179/UNNAMED

Then forward the port and open http://localhost:8766/ .

Why a backend at all: the stores are GB-scale and live on this machine, so the browser is
given a converted copy of only the element that was asked for — a uint8 volume plus a small
JSON — instead of a zarr reader and the whole store.

Endpoints
    GET /api/browse?dir=...          directories and *.zarr stores under `dir`
    GET /api/inventory?path=...      elements of that store: images / labels / shapes /
                                     points / tables, with shape, dtype and the transform
    GET /api/layer?path=...&element=...&...   build (and cache) a renderable layer, return its meta
    GET /api/tif?file=...            the same for an OME-TIFF z-stack (3D nuclei)
    GET /cache/<name>                the layer's binary (gzip of a uint8 volume)
    GET /api/config                  {"only": <file> or null, "catalog": bool}
    GET /api/datasets                catalog mode: the published datasets, by id, no paths
    GET /api/dataset_layer?id=..&key=..   catalog mode: build (and cache) one layer of one dataset
    GET /api/dataset_solid?id=..&key=..   catalog mode: that label layer's solid-surface volumes
    (a label layer with stride 1 is served at its native lattice as tiles; /api/dataset_layer
     answers 202 with progress while those are being cut)
                                     (202 with progress while the background build runs)

Single-file mode (`--only FILE`) exposes exactly one image stack and nothing else: no directory
listing, no zarr inventory or layer, no static file but the viewer page and the cache, and
`/api/tif` refuses any other path.  It is what to run when a view is shared and the rest of
this machine's stores must not be reachable through it.

Catalog mode (`--datasets manifest.json`) is the public form of the same idea for many datasets:
the page asks for a dataset by id and a layer by key, the server looks both up in the manifest
and builds the layer itself.  No path is ever accepted from, or returned to, the browser — the
layer meta is stripped of its source, and only the cached binaries are served, not their JSON.

A layer is always the same thing: a uint8 volume `(z, y, x)` plus
`{dims, voxel_um, origin_um, kind}`. `kind="labels"` means the value is a class id and the
palette is in `classes`; `kind="intensity"` means the value is 0..255 of one channel. The
viewer places every layer in ONE µm frame, so a 2D sheet and a 3D stack line up.
"""
import argparse
import functools
import gzip
import hashlib
import http.server
import json
import os
import posixpath
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
SERVE = HERE             # the page, thumbnails and cache are served from here (see --serve-dir)
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
LOCK = threading.Lock()
ROOT = Path("/")
ONLY = None             # single-file mode: the one image stack this server may read
ONLY_MIN_LEVEL = 2      # level 0/1 of a Xenium stack is GBs of uint8; not on offer in that mode
CATALOG = None          # catalog mode: {id: dataset} from --datasets; the only data served
TOKEN = None            # --token-file: every request needs this secret (link ?t=..., then a cookie)
TOKEN_COOKIE = "ssaft_t"


PLATFORMS = {"xenium": "Xenium", "merfish": "MERFISH", "stereoseq": "Stereo-seq",
             "stomics": "Stereo-seq",                             # the older key, for stored records
             "visiumhd": "Visium HD", "cosmx": "CosMx"}


def _info_rows(m):
    """The short fact list of the viewer's side panel, DERIVED from the dataset's metadata, so
    the website and the viewer cannot disagree about a dataset: there is one record, not two."""
    pub = m.get("publication") or {}
    cite = (f"{pub.get('first_author', '')} et al., "
            f"{pub.get('journal_abbrev') or pub.get('journal', '')} {pub.get('year', '')}") if pub else ""
    rows = [["Platform", PLATFORMS.get(m.get("platform"), m.get("platform", ""))],
            ["Species", m.get("species")], ["Tissue", m.get("tissue")],
            ["Condition", m.get("condition")], ["Publication", cite],
            ["Gene panel", m.get("panel")], ["Morphology", m.get("morphology")]]
    rows += [list(r) for r in m.get("highlights", [])]
    return [r for r in rows if r[1]]


def _public(d):
    """A dataset as the browser may see it: what it is and which layers it has, never where.

    `meta` is public by construction — the manifest keeps everything a visitor may read under it
    and every path under `layers`, which is never returned."""
    m = d.get("meta") or {}
    st = _dl_store(d)
    dl = DOWNLOADS.get(str(st)) if st else None
    return dict(id=d["id"], title=m.get("title", d.get("title", d["id"])),
                download=dict(dl, name=f"{d['id']}_ssaft.zip") if dl else None,  # bytes = without the KDE samples
                platform=m.get("platform", ""), meta=m,
                info=_info_rows(m) if m else d.get("info", []),
                layers=[dict(key=k, label=v.get("label", k),
                             kind=_layer_kind(v),
                             solid=_solid_spec(d, k) is not None, show=v.get("show", True),
                             group=v.get("group", k),
                             # an image — a stained slide, a DAPI stack — belongs in the 2D view, where
                             # every pixel is drawn as it is rather than stretched through a volume
                             view=v.get("view") or ("2d" if _layer_kind(v) in ("rgb", "intensity") else None))
                        for k, v in d["layers"].items()])


def _layer_kind(spec):
    """What the viewer draws a layer as: a label map, one intensity channel, or a colour image."""
    if spec["type"] == "tif":
        return "intensity"
    if spec["type"] == "sdata_image":
        return "rgb" if spec.get("_rgb") else "intensity"
    if spec["type"] == "sdata_shapes":
        return "shapes"
    return "labels"


def layer_for(d, key):
    """Build one catalog layer from the manifest's own parameters."""
    spec = d["layers"][key]
    if spec["type"] == "tif":
        m = layer_from_tif(Path(spec["file"]), stride=int(spec.get("stride", 1)),
                           z_um=float(spec.get("z_um", 3.0)),
                           um_per_px=float(spec.get("um_per_px", 0.2125)),
                           z0_um=float(spec.get("z0_um", 0.0)), level=spec.get("level"))
    elif spec["type"] == "sdata_image":
        m = image_volume_for(spec)
    elif spec["type"] == "ssaft":
        m = layer_from_ssaft(Path(spec["store"]), spec["element"],
                             stride=int(spec.get("stride", 1)))
    else:
        raise ValueError(f"unknown layer type {spec['type']!r}")
    m = {k: v for k, v in m.items() if k != "source"}
    m.update(key=key, name=spec.get("label", key))
    return m


# ----------------------------------------------------------------- zarr inspection
def _open(path):
    """A store's group, reading the arrays' OWN metadata.  A store rewritten in place can keep a stale
    consolidated index (zarr.json at its root): STM1's staining was rewritten at 0.26 um per pixel while
    the index still said 0.382, and the image was then placed 1.5x too large."""
    import zarr
    try:
        return zarr.open_group(path, mode="r", use_consolidated=False)
    except TypeError:                                            # older zarr without the option
        return zarr.open_group(path, mode="r")


def _ome(node):
    at = dict(node.attrs)
    return at.get("ome", at)


def _multiscale(node):
    ms = _ome(node).get("multiscales")
    return ms[0] if ms else None


def _axes(ms):
    return [a["name"] for a in ms.get("axes", [])] if ms else []


def _to_global(ms):
    """(scale, translation) of the element's own -> `global` transform, per axis."""
    if not ms:
        return None, None
    for t in ms.get("coordinateTransformations", []) or []:
        if t["type"] == "scale":
            return list(t["scale"]), None
        if t["type"] == "translation":
            return None, list(t["translation"])
        if t["type"] == "affine":
            A = np.asarray(t["affine"], dtype=float)
            return list(np.abs(np.diag(A[:, :-1]))) or None, list(A[:, -1])
        if t["type"] == "identity":
            return None, None
    return None, None


def _levels(node):
    ms = _multiscale(node)
    if not ms:
        return []
    return [d["path"] for d in ms.get("datasets", [])]


# An ssaft Dataset store is a plain zarr, not SpatialData: the maps are (x, y, z) arrays next
# to the palette. It is where the UNFILTERED map lives, so both maps can be layered from it.
SSAFT_ARRAYS = {
    "filtered_celltype_maps": ("labels", "cell-type map, after filter_celltypemaps"),
    "celltype_maps": ("labels", "cell-type map, before filtering"),
    "max_probabilities": ("intensity", "LDA posterior of the winning class"),
    "celltype_marker_score": ("intensity", "own-marker score of the winning class"),
    "vf_norm": ("intensity", "KDE density (molecules per um^3)"),
}


def inventory_ssaft(path, g):
    off = (np.asarray(g["coordinate_offset"][:], float).tolist()
           if "coordinate_offset" in g else [0.0, 0.0, 0.0])
    names = [str(x) for x in g["celltype_names"][:]] if "celltype_names" in g else []
    sd = float(g.attrs.get("sampling_distance", 1.0))
    zst = float(g.attrs.get("map_z_step", sd))
    out = {"path": str(path), "store": "ssaft", "elements": [], "origin_um": off,
           "n_classes": len(names)}
    for k, (kind, note) in SSAFT_ARRAYS.items():
        if k not in g:
            continue
        a = g[k]
        out["elements"].append(dict(group="ssaft", name=k, shape=list(a.shape),
                                    dtype=str(a.dtype), axes=["x", "y", "z"], kind=kind,
                                    renderable=True, note=note, um_per_px_guess=sd,
                                    z_um=zst, origin_um=off, n_classes=len(names)))
    for k in sorted(g.array_keys()):
        if k not in SSAFT_ARRAYS:
            out["elements"].append(dict(group="ssaft", name=k, shape=list(g[k].shape),
                                        dtype=str(g[k].dtype), renderable=False,
                                        note="not a raster"))
    return out


def layer_from_ssaft(path, name, stride=1, zmin=None, zmax=None, um_per_px=None, z_um=None):
    """`celltype_maps` / `vf_norm` / ... of an ssaft store -> uint8 volume + meta.

    The arrays are (x, y, z) and the viewer wants (z, y, x); class ids are kept as
    `value + 1` so the palette in the store's own attrs indexes straight into them.
    """
    import zarr
    key = _key(p=str(path), n=name, s=stride, a=zmin, b=zmax, u=um_per_px, z=z_um, v=2)
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    g = _open(str(path))
    a = g[name]
    z0 = 0 if zmin is None else max(0, int(zmin))
    z1 = a.shape[2] if zmax is None else min(a.shape[2], int(zmax) + 1)
    arr = np.asarray(a[::stride, ::stride, z0:z1]).transpose(2, 1, 0)   # -> (z, y, x)
    off = (np.asarray(g["coordinate_offset"][:], float)
           if "coordinate_offset" in g else np.zeros(3))
    sd = float(um_per_px if um_per_px else g.attrs.get("sampling_distance", 1.0))
    zst = float(z_um if z_um else g.attrs.get("map_z_step", sd))
    kind = SSAFT_ARRAYS.get(name, ("intensity", ""))[0]
    if kind == "labels":
        names = [str(x) for x in g["celltype_names"][:]] if "celltype_names" in g else []
        colors = list(g.attrs.get("cellman_celltype_colors", []))
        ids = np.unique(arr)
        ids = ids[ids >= 0]
        classes = [dict(id=int(v) + 1, value=int(v),
                        name=names[int(v)] if int(v) < len(names) else f"id {int(v)}",
                        color=colors[int(v)] if int(v) < len(colors) else "#cccccc",
                        voxels=int((arr == v).sum())) for v in ids if int(v) + 1 < 256]
        vol = np.where(arr >= 0, arr + 1, 0).astype(np.uint8)
        meta = dict(kind="labels", classes=classes, name=f"ssaft/{name}")
    else:
        v = arr.astype(np.float32)
        good = v[v > 0]
        lo_v = 0.0
        hi_v = float(np.percentile(good[::max(1, good.size // 200000)], 99.5)) if good.size else 1.0
        vol = np.clip((v - lo_v) * (255.0 / max(hi_v - lo_v, 1e-9)), 0, 255).astype(np.uint8)
        meta = dict(kind="intensity", name=f"ssaft/{name}", lo=lo_v, hi=hi_v, color="#9fd0ff")
    meta.update(voxel_um=[sd * stride, sd * stride, zst],
                origin_um=[float(off[0]), float(off[1]), float(off[2]) + z0 * zst],
                source=dict(path=str(path), element=name, stride=stride, z=[z0, z1 - 1]))
    return _write_layer(key, vol, meta)


def inventory(path):
    import zarr
    g = _open(str(path))
    if any(k in g for k in SSAFT_ARRAYS):
        return inventory_ssaft(path, g)
    out = {"path": str(path), "store": "spatialdata", "elements": []}
    # the pixel size of the reference image, used to guess µm/px of elements whose transform
    # is recorded in pixel units (ssaft writes its map on a 1 µm lattice, but a Xenium store's
    # `global` frame is image pixels, so the two do not agree without saying so)
    ref = None
    for kind in ("images", "labels"):
        if kind not in g:
            continue
        for name in sorted(g[kind].group_keys()):
            node = g[kind][name]
            ms = _multiscale(node)
            lv = _levels(node)
            if not lv:
                continue
            a = node[lv[0]]
            ax = _axes(ms) or (["c", "y", "x"] if len(a.shape) == 3 else ["y", "x"])
            sc, tr = _to_global(ms)
            e = dict(group=kind, name=name, shape=list(a.shape), dtype=str(a.dtype),
                     axes=ax, levels=lv, scale=sc, translation=tr,
                     renderable=True, kind="labels" if kind == "labels" else "intensity")
            yx = [a.shape[ax.index("y")], a.shape[ax.index("x")]] if "y" in ax and "x" in ax else None
            e["yx"] = yx
            if ref is None and kind == "images" and yx:
                ref = yx
            out["elements"].append(e)
    for kind in ("shapes", "points"):
        if kind not in g:
            continue
        for name in sorted(g[kind].group_keys()):
            out["elements"].append(dict(group=kind, name=name, renderable=False,
                                        note="not rendered yet"))
    if "tables" in g:
        for name in sorted(g["tables"].group_keys()):
            t = g["tables"][name]
            n = None
            try:
                n = int(t["obs/_index"].shape[0])
            except Exception:
                for k in t["obs"].array_keys():
                    n = int(t["obs"][k].shape[0]); break
            out["elements"].append(dict(group="tables", name=name, renderable=False,
                                        n_obs=n, note="palette / labels source"))
    # a µm-per-pixel guess for every raster: the reference image is the finest grid
    for e in out["elements"]:
        if not e.get("renderable") or not e.get("yx"):
            continue
        e["um_per_px_guess"] = round(ref[0] * 0.2125 / e["yx"][0], 6) if ref else 1.0
    out["reference_yx"] = ref
    return out


def palette_from_tables(path, n_classes, table=None):
    """(names, colors) for a celltype map, from the table asked for, or from whichever table
    carries an ssaft palette — in NAME order, so a store that gains a table does not silently
    hand a layer a different palette than it had yesterday."""
    import zarr
    g = _open(str(path))
    if "tables" not in g:
        return None, None
    for name in ([table] if table else sorted(g["tables"].group_keys())):
        uns = g["tables"][name].get("uns")
        if uns is None:
            continue
        keys = set(uns.array_keys())
        nm = next((k for k in keys if k.endswith("celltype_names")), None)
        cl = next((k for k in keys if k.endswith("_colors")), None)
        if nm:
            names = [str(x) for x in uns[nm][:]]
            colors = [str(x) for x in uns[cl][:]] if cl else None
            if colors and len(colors) >= len(names):
                return names, colors[:len(names)]
            return names, None
    return None, None


def _palette_mtime(src):
    """When the palette store was last written: a layer's colours and names come from outside the
    SpatialData store, so an edit there has to reach the cache key too."""
    root = Path(src)
    fs = [root / "zarr.json", root / ".zattrs"] + [f for f in (root / "celltype_names").rglob("*") if f.is_file()]
    ts = [f.stat().st_mtime for f in fs if f.exists()]
    return round(max(ts), 3) if ts else 0


def _palette(spec):
    """(names, colours) for a label layer.

    A section can be mapped more than once — a second annotation of the same tissue is its own set
    of cell types — so a layer may name the plain ssaft store its labels came from, and the palette
    is read from there (`celltype_names` + `cellman_celltype_colors`) instead of from the
    SpatialData tables, which carry the FIRST mapping's palette."""
    src = spec.get("palette_store")
    if not src:
        # a layer read through a table is named by THAT table's palette; one that names a table
        # explicitly by that one; anything else by whichever table carries a palette
        return palette_from_tables(spec["store"], 0, spec.get("palette_table") or spec.get("table"))
    import zarr
    g = zarr.open_group(str(src), mode="r")
    names = [str(x) for x in g["celltype_names"][:]] if "celltype_names" in g else []
    colors = [str(c) for c in (g.attrs.get("cellman_celltype_colors") or [])]
    return (names or None), (colors[:len(names)] if colors and names else None)


# ----------------------------------------------------------------- SpatialData elements
# A SpatialData store records, per element, a transform from its array index to the `global`
# frame, and the viewer draws in µm.  Every element is therefore placed through its OWN transform
# (the level's scale, then the element's), times the µm per `global` unit of the store — never by
# a pixel size typed into the manifest.  Catalog layer types that read such a store:
#   sdata_labels  labels/<element>, 0 = no call, value k = the ssaft table's celltype_names[k - 1]
#   sdata_image   images/<element>: one channel (optionally a z-stack) or an RGB image
def _tx_matrix(t, axes):
    """4x4 matrix over (x, y, z, 1) of one NGFF coordinate transform, and its output axis names.
    `axes` names the input when the transform does not; channel axes are ignored."""
    ins = [a["name"] for a in t["input"]["axes"]] if t.get("input") else list(axes)
    outs = [a["name"] for a in t["output"]["axes"]] if t.get("output") else list(ins)
    sp = {"x": 0, "y": 1, "z": 2}
    typ = t["type"]
    M = np.eye(4)
    if typ == "identity":
        return M, outs
    if typ in ("scale", "translation"):
        for n, v in zip(ins, t[typ]):
            if n in sp:
                if typ == "scale":
                    M[sp[n], sp[n]] = v
                else:
                    M[sp[n], 3] = v
        return M, ins
    if typ == "affine":
        A = np.asarray(t["affine"], float)
        M = np.zeros((4, 4))
        M[3, 3] = 1.0
        for i, on in enumerate(outs):
            if on not in sp:
                continue
            for j, n in enumerate(ins):
                if n in sp:
                    M[sp[on], sp[n]] = A[i, j]
            M[sp[on], 3] = A[i, -1]
        for n, k in sp.items():                  # an axis the affine does not name passes through
            if n not in outs:
                M[k, k] = 1.0
        return M, outs
    if typ == "sequence":
        cur = ins
        for sub in t["transformations"]:
            Ms, cur = _tx_matrix(sub, cur)
            M = Ms @ M
        return M, outs
    raise ValueError(f"unsupported coordinate transform {typ!r}")


def _element_matrix(node, level):
    """Array index (x, y, z) of `level` -> the store's `global` frame."""
    ms = _multiscale(node)
    axes = _axes(ms)
    M = np.eye(4)
    for d in ms.get("datasets", []):
        if d["path"] == level:
            for t in d.get("coordinateTransformations", []) or []:
                M = _tx_matrix(t, axes)[0] @ M
    txs = ms.get("coordinateTransformations", []) or []
    glob = [t for t in txs if (t.get("output") or {}).get("name") == "global"] or txs[:1]
    for t in glob[:1]:
        M = _tx_matrix(t, axes)[0] @ M
    return M


def _um_per_unit(g, spec):
    """µm per `global` unit, in x and y.  spatialdata-io's Xenium reader keeps the image pixel grid
    as `global` and scales the µm-valued transcripts into it, so that scale is the answer; a store
    without such points is in µm.  Every element of the store is read in that one frame (the ssaft
    elements included, now that they are written into it like the vendor's)."""
    if spec.get("um_per_unit") is not None:
        return float(spec["um_per_unit"])
    if "points" not in g:
        return 1.0
    for n in g["points"].group_keys():
        for t in dict(g["points"][n].attrs).get("coordinateTransformations", []) or []:
            M = _tx_matrix(t, ["x", "y", "z"])[0]            # scale, affine or a sequence of them
            sx, sy = np.hypot(M[0, 0], M[0, 1]), np.hypot(M[1, 0], M[1, 1])
            # x and y scales agree only to the 7th digit in some stores, so compare them loosely
            if sx > 0 and abs(sx - sy) < 1e-4 * sx and abs(sx - 1.0) > 1e-6:
                return 1.0 / float(sx)
    return 1.0


def _rotate_onto_lattice(arr, S, vox):
    """A rotated (z, y, x) element read back onto the axis-aligned µm lattice, NEAREST neighbour.

    A label may not be interpolated — half of cell type 3 and half of 7 is not 5 — so every output
    voxel takes the single input voxel its centre falls in, and what falls outside the element is
    0 (no call).  Returns the volume and the µm of its (x, y) corner."""
    Z, H, W = arr.shape
    L, t = S[:2, :2], S[:2, 3]
    pts = np.array([[0., 0.], [W, 0.], [0., H], [W, H]]) @ L.T + t
    lo, hi = pts.min(0), pts.max(0)
    sx, sy = float(vox[0]), float(vox[1])
    OW, OH = int(np.ceil((hi[0] - lo[0]) / sx)), int(np.ceil((hi[1] - lo[1]) / sy))
    Li = np.linalg.inv(L)
    X = lo[0] + (np.arange(OW)[None, :] + 0.5) * sx
    out = np.zeros((Z, OH, OW), arr.dtype)
    for r0 in range(0, OH, 1024):                            # by strips: the index grids are large
        Y = lo[1] + (np.arange(r0, min(OH, r0 + 1024))[:, None] + 0.5) * sy
        dx, dy = X - t[0], Y - t[1]
        ci = np.rint(Li[0, 0] * dx + Li[0, 1] * dy - 0.5).astype(np.int64)
        ri = np.rint(Li[1, 0] * dx + Li[1, 1] * dy - 0.5).astype(np.int64)
        ok = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        np.clip(ci, 0, W - 1, out=ci)
        np.clip(ri, 0, H - 1, out=ri)
        for z in range(Z):
            out[z, r0:r0 + ri.shape[0]] = np.where(ok, arr[z][ri, ci], 0)
    return out, [float(lo[0]), float(lo[1])]


def _align(arr, M, upu, spec, resample=False):
    """(z, y, x) array + its index->global matrix -> the array on the viewer's axis-aligned µm
    lattice: axes swapped and flipped as the transform says, voxel size and corner origin in µm.

    An element can also be ROTATED in that frame — a Visium HD capture area sits on the slide at
    its own small angle, and the map is written on the bin lattice — which no swap or flip can
    place.  A label layer asks to be resampled onto the lattice instead (`resample`); an image
    goes through rgb_tiles_for, which resamples with interpolation."""
    S = M.copy()
    S[:2, :] *= upu
    if abs(S[0, 1]) > abs(S[0, 0]):              # x is driven by the row index: swap x and y
        arr = arr.transpose(0, 2, 1)
        S[:, [0, 1]] = S[:, [1, 0]]
    L = S[:3, :3]
    vox = [float(abs(L[0, 0])), float(abs(L[1, 1])), float(abs(L[2, 2]))]
    if max(abs(L[0, 1]), abs(L[1, 0])) > 1e-3 * min(abs(L[0, 0]), abs(L[1, 1])):
        if not resample:
            raise ValueError("the element is rotated in its frame; it has to be resampled")
        arr, (ox, oy) = _rotate_onto_lattice(arr, S, vox)
        org = [ox, oy, float(S[2, 3])]
        if L[2, 2] < 0:                          # the resampling leaves only z to flip
            arr = np.flip(arr, axis=0)
            org[2] += float(L[2, 2]) * arr.shape[0]
        if arr.shape[0] == 1:
            vox[2] = float(spec.get("z_um", vox[2] or 1.0))
            org[2] = float(spec.get("z0_um", org[2]))
        return np.ascontiguousarray(arr), vox, org
    org = [float(v) for v in S[:3, 3]]
    for ax, arr_ax in ((0, 2), (1, 1), (2, 0)):
        if L[ax, ax] < 0:
            arr = np.flip(arr, axis=arr_ax)
            org[ax] += float(L[ax, ax]) * arr.shape[arr_ax]
    if arr.shape[0] == 1:                        # a 2D element is a sheet: where, and how thick
        vox[2] = float(spec.get("z_um", vox[2] or 1.0))
        org[2] = float(spec.get("z0_um", org[2]))
    return np.ascontiguousarray(arr), vox, org


def sdata_is_rgb(spec):
    """Is this image served as picture tiles rather than one intensity volume?  A colour image is,
    and so is any single-plane image the manifest marks `as_image` — a whole-slide ssDNA or H&E scan
    is far too large for one GPU texture."""
    import zarr
    node = _open(str(spec["store"]))["images"][spec["element"]]
    ax = _axes(_multiscale(node))
    a = node[_levels(node)[0]]
    if "z" in ax and spec.get("plane") is None:   # a stack is a volume unless one plane is picked
        return False
    nc = a.shape[ax.index("c")] if "c" in ax else 1
    if spec.get("channel") is not None:           # one channel of a many-channel scan: a grey picture
        nc = 1
    return (nc == 3 and str(a.dtype) == "uint8") or (bool(spec.get("as_image")) and nc in (1, 3))


def image_volume_for(spec):
    """images/<element> of a SpatialData store, one channel -> uint8 intensity volume + meta.
    A z-stack keeps its planes (nuclei in 3D); `level` picks the pyramid level."""
    import zarr
    key = _key(t="sdimg", p=str(spec["store"]), n=spec["element"], l=spec.get("level"),
               c=spec.get("channel", 0), u=spec.get("um_per_unit"), z0=spec.get("z0_um"),
               zu=spec.get("z_um"),
               **({} if spec.get("plane") is None else dict(pl=spec["plane"])),
               mt=_mtime(spec), v=1)
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    g = _open(str(spec["store"]))
    node = g["images"][spec["element"]]
    lv = spec.get("level") or _levels(node)[0]
    ax = _axes(_multiscale(node))
    sel = [slice(None)] * len(ax)                # read one channel, and one plane if asked
    if "c" in ax:
        sel[ax.index("c")] = int(spec.get("channel", 0))
    if "z" in ax and spec.get("plane") is not None:
        nz = node[lv].shape[ax.index("z")]
        sel[ax.index("z")] = max(0, min(nz - 1, int(spec["plane"])))
    arr = np.asarray(node[lv][tuple(sel)])
    if arr.ndim == 2:
        arr = arr[None]
    sub = arr[:, ::4, ::4]
    lo_v, hi_v = float(np.percentile(sub, 1.0)), float(np.percentile(sub, 99.8))
    vol = np.clip((arr.astype(np.float32) - lo_v) * (255.0 / max(hi_v - lo_v, 1e-6)),
                  0, 255).astype(np.uint8)
    vol, vox, org = _align(vol, _element_matrix(node, lv), _um_per_unit(g, spec), spec)
    meta = dict(kind="intensity", name=spec["element"], color="#9fd0ff", lo=lo_v, hi=hi_v,
                voxel_um=vox, origin_um=org)
    return _write_layer(key, vol, meta)


def label_volume(spec):
    """A label layer as one (z, y, x) uint8 volume: 0 = no call, id k = names[k - 1]."""
    import zarr
    g = _open(str(spec["store"]))
    if spec["type"] == "ssaft":                  # plain ssaft Dataset store: (x, y, z), -1 = no call
        a = g[spec["element"]]
        X, Y, Z = a.shape
        off = (np.asarray(g["coordinate_offset"][:], float)
               if "coordinate_offset" in g else np.zeros(3))
        sd = float(g.attrs.get("sampling_distance", 1.0))
        zst = float(g.attrs.get("map_z_step", sd))
        names = [str(x) for x in g["celltype_names"][:]] if "celltype_names" in g else []
        if len(names) + 1 > 255:
            raise ValueError(f"{len(names)} classes do not fit a uint8 label texture")
        vol = np.empty((Z, Y, X), np.uint8)
        for z in range(Z):
            sl = np.asarray(a[:, :, z])
            vol[z] = np.where(sl >= 0, sl + 1, 0).astype(np.uint8).T
        return _z_trim(dict(vol=vol, names=names,
                            colors=list(g.attrs.get("cellman_celltype_colors", [])),
                            voxel_um=[sd, sd, zst], origin_um=[float(v) for v in off],
                            name=f"ssaft/{spec['element']}"), "vol", spec.get("z_trim"))
    node = g["labels"][spec["element"]]
    lv = spec.get("level") or _levels(node)[0]
    arr = np.asarray(node[lv][:])
    if arr.ndim == 2:
        arr = arr[None]
    if spec.get("table"):           # instance labels (one id per cell): each voxel takes its cell's type
        arr = _instance_lut(g, spec, int(arr.max(initial=0)))[np.clip(arr, 0, None)]
    top = int(arr.max()) if arr.size else 0
    if top > 254:
        raise ValueError(f"label value {top} does not fit a uint8 label texture")
    arr = np.where(arr > 0, arr, 0).astype(np.uint8)
    names, colors = _palette(spec)
    vol, vox, org = _align(arr, _element_matrix(node, lv), _um_per_unit(g, spec), spec, resample=True)
    return _z_trim(dict(vol=vol, names=names or [], colors=colors or [], voxel_um=vox, origin_um=org,
                        name=f"labels/{spec['element']}"), "vol", spec.get("z_trim"))


def _shape_matrix(node):
    """A shapes element's coordinates -> the store's `global` frame, over (x, y, z)."""
    txs = dict(node.attrs).get("coordinateTransformations", []) or []
    glob = [t for t in txs if (t.get("output") or {}).get("name") == "global"] or txs[:1]
    M = np.eye(4)
    for t in glob[:1]:
        M = _tx_matrix(t, ["x", "y"])[0] @ M
    return M


def _obs_column(g, table, column, key=None):
    """(ids, values) of one column of a table's obs, read straight from the zarr; the ids are the
    table's instance key unless another obs column is named."""
    import zarr
    tb = g["tables"][table]
    obs = tb["obs"]

    def strings(a):                                          # zarr v3 hands back numpy StringDType
        return np.array([str(x) for x in np.asarray(a).tolist()], dtype=object)

    def col(name):
        node = obs[name]
        if isinstance(node, zarr.Group):                     # categorical: codes + categories
            cats = strings(node["categories"][:])
            codes = np.asarray(node["codes"][:])
            return np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
        return strings(node[:])
    return col(key or dict(tb.attrs)["instance_key"]), col(column)


def shapes_for(spec, progress=None):
    """shapes/<element> of a SpatialData store (cell or nucleus boundaries) -> outline buffers.

    Every polygon ring becomes one closed line loop.  The polygons are placed through the element's
    own transform (µm), simplified by `simplify_um` (steps smaller than that are dropped; 0 keeps
    every vertex), and their vertices written as uint16 at `q` µm — 1/8 µm while the section fits
    in 65,535 steps — with one vertex count per ring.  With `table`, each cell carries its cell type
    (`color_key` of that table's obs, matched by the table's instance key), as the class id of the
    cell-type map, so the outlines can be coloured and filtered like the map."""
    import zarr
    import shapely
    import geopandas as gpd
    tol = float(spec.get("simplify_um", 0.25))
    ck = spec.get("color_key", "cellman_annotation")
    fill = bool(spec.get("fill", True))
    key = _key(t="shapes", p=str(spec["store"]), n=spec["element"], tol=tol, tb=spec.get("table"),
               ck=ck, u=spec.get("um_per_unit"), z=spec.get("z_um"), idc=spec.get("id_column"),
               tk=spec.get("table_key"), zc=spec.get("z_column"), zs=spec.get("z_step_um"), fill=fill,
               **({} if spec.get("z_trim") is None else dict(zt=spec["z_trim"])),
               mt=_mtime(spec), v=4)
    cached = CACHE / f"{key}_shapes.json"
    if cached.exists():                  # the colour is a display choice of the manifest, not of the cache
        return dict(json.loads(cached.read_text()), color=spec.get("color", "#ffffff"))
    g = _open(str(spec["store"]))
    node = g["shapes"][spec["element"]]
    df = gpd.read_parquet(Path(spec["store"]) / "shapes" / spec["element"] / "shapes.parquet")
    if spec.get("z_trim") and spec.get("z_column"):      # the same rim planes the solid view drops
        zc = df[spec["z_column"]].to_numpy(float)
        planes, inv = np.unique(zc, return_inverse=True)
        area = np.bincount(inv, weights=df.geometry.area.to_numpy(), minlength=len(planes))
        keep = np.nonzero(area >= float(spec["z_trim"]) * area.max())[0]
        lo, hi = float(planes[keep[0]]), float(planes[keep[-1]])
        if (zc < lo).any() or (zc > hi).any():
            df = df[(zc >= lo) & (zc <= hi)]
            print(f"  z trim: outlines kept between {lo:g} and {hi:g} um "
                  f"({len(df)} of {len(zc)} rings)", flush=True)
    if progress:
        progress(1, 5)
    S = _shape_matrix(node)
    S[:2, :] *= _um_per_unit(g, spec)
    A, t = S[:2, :2], S[:2, 3]
    geoms = shapely.transform(np.asarray(df.geometry.values), lambda c: c @ A.T + t)
    geoms, part_row = shapely.get_parts(geoms, return_index=True)   # a MultiPolygon cell -> its polygons
    if tol > 0:
        geoms = shapely.simplify(geoms, tol, preserve_topology=True)
    # an invalid polygon (a self-touching ring from the vendor's segmentation) cannot be triangulated:
    # it is replaced, before its rings are taken, by the largest polygon of its repaired geometry
    bad = np.nonzero(~shapely.is_valid(geoms))[0]
    for i in bad:
        fixed = shapely.make_valid(geoms[i])
        cand = [g for g in getattr(fixed, "geoms", [fixed]) if g.geom_type == "Polygon" and not g.is_empty]
        geoms[i] = max(cand, key=lambda g: g.area) if cand else shapely.Polygon()
    if progress:
        progress(2, 5)
    row_cls, classes = np.zeros(len(df), np.uint8), []
    if spec.get("table"):
        names, colors = _palette(spec)                   # the same palette the layer is labelled with
        names, colors = names or [], colors or []
        ids, vals = _obs_column(g, spec["table"], ck, key=spec.get("table_key"))
        cell_type = dict(zip(ids, vals))
        class_id = {n: k + 1 for k, n in enumerate(names) if k + 1 < 256}
        idc = spec.get("id_column")
        # geopandas restores a column stored as the pandas index AS the index, so the named id may be either
        row_ids = (df[idc] if idc and idc in df.columns else df.index).astype(str)
        row_cls = np.array([class_id.get(cell_type.get(i, ""), 0) for i in row_ids], np.uint8)
        cnt = np.bincount(row_cls, minlength=256)
        classes = [dict(id=k, name=names[k - 1], color=colors[k - 1] if k - 1 < len(colors) else "#cccccc",
                        voxels=int(cnt[k])) for k in range(1, 256) if cnt[k] > 0]
    poly_cls = row_cls[part_row]
    def ring_arrays(gs):
        rg, rp = shapely.get_rings(gs, return_index=True)
        xy, of = shapely.get_coordinates(rg, return_index=True)
        per = np.bincount(of, minlength=len(rg))
        keep = np.ones(len(xy), bool)
        keep[np.cumsum(per) - 1] = False                     # a ring repeats its first vertex last
        return rg, rp, xy[keep], per - 1
    rings, ring_poly, coords, counts = ring_arrays(geoms)
    if progress:
        progress(3, 5)
    # A cell left whole — the myocardium, the syncytium — can have an outline of more than 65,535
    # vertices, which the ring's uint16 vertex count cannot hold.  Those cells, and only those, are
    # simplified harder until their rings fit; everything else keeps the tolerance it was given.
    coarse = 0
    while len(counts) and counts.max() > 65535:
        over = np.unique(ring_poly[counts > 65535])
        coarse = max(coarse, tol) * 2 if coarse or tol else 0.5
        geoms[over] = shapely.simplify(geoms[over], coarse, preserve_topology=True)
        rings, ring_poly, coords, counts = ring_arrays(geoms)
        print(f"  {len(over)} outline(s) over 65,535 vertices: simplified at {coarse} um", flush=True)
        if coarse > 64:
            raise ValueError("a ring is still over 65,535 vertices at a 64 um tolerance")
    org = np.floor(coords.min(0))
    ext = coords.max(0) - org
    q = 0.125
    while ext.max() / q > 65535:
        q *= 2
    xy = np.round((coords - org) / q).astype("<u2")
    # the fill (the segmentation drawn as a map): each polygon's constrained Delaunay triangulation.
    # It adds no vertex, so every triangle corner is one of the polygon's own ring vertices, sent as a
    # uint16 index local to that polygon, with the triangle and ring counts per polygon.
    n_poly = len(geoms)
    rings_per_poly = np.bincount(ring_poly, minlength=n_poly)
    vert_poly = np.repeat(ring_poly, counts)                 # vertices are grouped by polygon, in order
    poly_start = np.zeros(n_poly + 1, np.int64)
    poly_start[1:] = np.cumsum(np.bincount(vert_poly, minlength=n_poly))
    def cdt(gs):                                             # isolate the rare polygon GEOS still refuses
        try:
            return shapely.constrained_delaunay_triangles(gs)
        except shapely.errors.GEOSException:
            if len(gs) == 1:
                return np.array([shapely.GeometryCollection()], dtype=object)
            h = len(gs) // 2
            return np.concatenate([cdt(gs[:h]), cdt(gs[h:])])
    if fill:
        tris = np.concatenate([cdt(geoms[i:i + 20000]) for i in range(0, max(n_poly, 1), 20000)])
        n_unfilled = int(((shapely.get_num_geometries(tris) == 0) & (shapely.area(geoms) > 0)).sum())
    else:                                                    # outlines only
        tris, n_unfilled = np.array([], dtype=object), 0
    tri_geoms, tri_poly = shapely.get_parts(tris, return_index=True)
    tri_xy = shapely.get_coordinates(tri_geoms)
    if len(tri_xy) != 4 * len(tri_geoms):
        raise ValueError("a triangulation part is not a triangle")
    tri_xy = tri_xy.reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    if progress:
        progress(4, 5)

    def vkey(poly, qxy):                                     # (polygon, quantized x, quantized y)
        return ((poly.astype(np.uint64) << np.uint64(32)) | (qxy[:, 0].astype(np.uint64) << np.uint64(16))
                | qxy[:, 1].astype(np.uint64))
    keys = vkey(vert_poly, xy)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    tkeys = vkey(np.repeat(tri_poly, 3), np.round((tri_xy - org) / q).astype("<u2"))
    pos = np.clip(np.searchsorted(sk, tkeys), 0, len(sk) - 1)
    ok = (sk[pos] == tkeys).reshape(-1, 3).all(1)            # every corner found among the ring vertices
    # A cell left whole — the syncytium, the myocardium — can have an outline of more than 65,535
    # vertices, which no uint16 index into its own vertices can reach.  That one cell is drawn as an
    # outline and left unfilled, rather than the layer failing.
    big = np.diff(poly_start) > 65535
    if big.any():
        ok &= ~big[tri_poly]
        n_unfilled += int(big.sum())
    local = (order[pos] - poly_start[np.repeat(tri_poly, 3)]).reshape(-1, 3)[ok]
    tris_per_poly = np.bincount(tri_poly[ok], minlength=n_poly)
    if len(local) and local.max() > 65535:
        raise ValueError("a polygon has more than 65,535 vertices")
    # per-plane outlines (a 3D segmentation drawn plane by plane): each ring's plane, drawn on its top face
    z_top, dz, ring_plane = float(spec.get("z_um", 0.0)), 0.0, None
    if spec.get("z_column"):
        row_z = df[spec["z_column"]].to_numpy(float)
        planes = np.unique(row_z)
        dz = float(spec.get("z_step_um") or (np.diff(planes).min() if len(planes) > 1 else 1.0))
        z_top = float(planes.min() + dz)
        ring_plane = np.rint((row_z[part_row][ring_poly] - planes.min()) / dz).astype(np.uint8)
    files, nbytes = {}, 0
    for tag, arr in (("xy", xy), ("n", counts.astype("<u2")), ("c", poly_cls[ring_poly]),
                     ("rp", rings_per_poly.astype("<u2")), ("tc", tris_per_poly.astype("<u2")),
                     ("ti", local.astype("<u2")), ("zr", ring_plane)):
        if (tag == "c" and not classes) or (tag in ("rp", "tc", "ti") and not fill) or arr is None:
            continue
        fn = f"{key}_{tag}.bin.gz"
        with gzip.open(CACHE / fn, "wb", compresslevel=6) as fh:
            fh.write(np.ascontiguousarray(arr).tobytes())
        files[tag] = f"cache/{fn}"
        nbytes += (CACHE / fn).stat().st_size
    meta = dict(kind="shapes", name=spec["element"],
                shapes=dict(xy=files["xy"], n=files["n"], c=files.get("c"), rp=files.get("rp"),
                            tc=files.get("tc"), ti=files.get("ti"), zr=files.get("zr")), q=q, dz=dz,
                n_cells=int(len(df)), n_rings=int(len(counts)), n_vertices=int(len(xy)),
                n_triangles=int(len(local)), n_triangles_unmatched=int((~ok).sum()),
                n_repaired=int(len(bad)), n_unfilled=n_unfilled,
                n_vertices_raw=int(shapely.get_num_coordinates(np.asarray(df.geometry.values)).sum()),
                origin_um=[float(org[0]), float(org[1]), z_top],
                voxel_um=[1.0, 1.0, 1.0], dims=dict(x=int(np.ceil(ext[0])) + 1, y=int(np.ceil(ext[1])) + 1, z=1),
                classes=classes, color=spec.get("color", "#ffffff"), simplify_um=tol, bytes=nbytes)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(5, 5)
    return meta


def _mtime(spec):
    """Newest modification time of the files that define an element (and the table it is coloured
    through), part of every cache key, so a store rewritten in place is never served stale."""
    root = Path(spec["store"])
    dirs = [root] if spec["type"] == "ssaft" else [root / grp / spec["element"] for grp in ("labels", "images", "shapes")]
    if spec.get("table"):
        dirs.append(root / "tables" / spec["table"] / "obs")
    ts = [f.stat().st_mtime for d in dirs if d.is_dir() for f in d.iterdir() if f.is_file()]
    ts += [f.stat().st_mtime for d in dirs if d.is_dir() for f in d.glob("*/zarr.json")]
    # The palette (uns names/colours) colours every layer and lives outside those dirs.  A layer
    # that says where its palette comes from watches only THAT one — otherwise a store that gains
    # or edits any table re-cuts every layer it holds, which is hours of tiles and solids.
    if spec.get("palette_store"):
        pass                                             # already in the key, as _palette_mtime
    elif spec.get("palette_table") or spec.get("table"):
        # a layer read THROUGH a table is coloured by that table (`_palette`), so that one table is
        # all it has to watch -- a store that gains a segmentation, and with it a table, must not
        # re-cut the layers of every earlier run
        t = spec.get("palette_table") or spec["table"]
        ts += [f.stat().st_mtime for f in (root / "tables" / t / "uns").rglob("*") if f.is_file()]
    else:
        ts += [f.stat().st_mtime for f in root.glob("tables/*/uns/**/*") if f.is_file()]
    return round(max(ts), 3) if ts else 0


def _instance_lut(g, spec, top):
    """cell instance id -> class id of the cell's type (0 = unannotated), through the spec's table."""
    names, _ = _palette(spec)                            # the ids have to index the SAME palette the
                                                         # layer is drawn and labelled with
    class_id = {n: k + 1 for k, n in enumerate(names or []) if k + 1 < 256}
    ids, vals = _obs_column(g, spec["table"], spec.get("color_key", "cellman_annotation"))
    inst = np.array([int(float(i)) for i in ids], np.int64)
    lut = np.zeros(int(max(inst.max(initial=0), top)) + 1, np.uint8)
    lut[inst] = [class_id.get(v, 0) for v in vals]
    return lut


def _z_trim(src, key, frac):
    """Cut the z planes at either end where the lattice runs past the tissue.

    A KDE map is written over the whole z lattice, and at the slab's rim the field is so thin that a
    plane carries a few per mille of the calls an inner plane does — labels decided on almost no
    transcripts, which the solid view then stands up as little balls above the surface.  A plane is
    kept while its labelled area reaches `frac` of the fullest plane's; the cut is taken from the two
    ends only, since a thin plane INSIDE the tissue is tissue (a lumen, a gap between tubules)."""
    vol = src[key]
    if not frac or vol.shape[0] < 3:
        return src
    area = (vol > 0).mean(axis=(1, 2))
    keep = np.nonzero(area >= float(frac) * area.max())[0]
    z0, z1 = int(keep[0]), int(keep[-1]) + 1
    if z1 - z0 == vol.shape[0]:
        return src
    src[key] = vol[z0:z1]
    src["origin_um"] = [src["origin_um"][0], src["origin_um"][1],
                        src["origin_um"][2] + z0 * src["voxel_um"][2]]
    print(f"  z trim: planes {z0}-{z1 - 1} of {vol.shape[0]} kept "
          f"(labelled area at least {frac:g} of the fullest plane's)", flush=True)
    return src


def instance_volume(spec):
    """A cell segmentation's instance labels as (z, y, x) int32 on the µm lattice, with the lookup
    from instance to cell type."""
    import zarr
    g = _open(str(spec["store"]))
    node = g["labels"][spec["element"]]
    lv = spec.get("level") or _levels(node)[0]
    arr = np.asarray(node[lv][:])
    if arr.ndim == 2:
        arr = arr[None]
    arr = np.where(arr > 0, arr, 0).astype(np.int32)
    lut = _instance_lut(g, spec, int(arr.max(initial=0)))
    names, colors = _palette(spec)
    ids, vox, org = _align(arr, _element_matrix(node, lv), _um_per_unit(g, spec), spec, resample=True)
    return _z_trim(dict(ids=ids, lut=lut, names=names or [], colors=colors or [], voxel_um=vox,
                        origin_um=org, name=f"labels/{spec['element']}"), "ids", spec.get("z_trim"))


TILE_BYTES = 256e6      # a full-resolution tile (all its z levels) stays under this many bytes on the GPU


def _auto_tile(spec, bytes_per_voxel, tile):
    """The tile edge for a label layer: `tile`, halved while one tile of all its z levels would pass
    TILE_BYTES.  The viewer holds about 1.2 GB of tiles; a deep map (80 z levels) cut at 2048 has
    tiles of 0.3-1 GB, so hardly one fits and the view falls back to the coarse overview."""
    import zarr
    try:
        g = _open(str(spec["store"]))
        node = g["labels"][spec["element"]]
        a = node[spec.get("level") or _levels(node)[0]]
        Z = a.shape[0] if len(a.shape) >= 3 else 1
    except Exception:                                            # noqa: BLE001
        return int(tile)
    T = int(tile)
    while T > 256 and T * T * Z * bytes_per_voxel > TILE_BYTES:
        T //= 2
    return T


def _label_key(spec):
    return dict(t=spec["type"], p=str(spec["store"]), n=spec["element"], l=spec.get("level"),
                z0=spec.get("z0_um"), zu=spec.get("z_um"), u=spec.get("um_per_unit"),
                **({} if spec.get("z_trim") is None else dict(zt=spec["z_trim"])),
                tb=spec.get("table"), ck=spec.get("color_key"),
                **({} if spec.get("palette_store") is None else
                   dict(pal=str(spec["palette_store"]), palmt=_palette_mtime(spec["palette_store"]))),
                **({} if spec.get("palette_table") is None else dict(palt=spec["palette_table"])),
                mt=_mtime(spec))


# ----------------------------------------------------------------- layer building
def _write_layer(key, vol, meta):
    f = CACHE / f"{key}.bin.gz"
    if not f.exists():
        with gzip.open(f, "wb", compresslevel=6) as fh:
            fh.write(np.ascontiguousarray(vol).tobytes())
    meta = dict(meta)
    meta.update(file=f"cache/{key}.bin.gz",
                dims=dict(x=int(vol.shape[2]), y=int(vol.shape[1]), z=int(vol.shape[0])),
                bytes=f.stat().st_size)
    (CACHE / f"{key}.json").write_text(json.dumps(meta))
    return meta


def _key(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:16]


def layer_from_zarr(path, group, name, stride=1, um_per_px=None, z_um=1.0, z0_um=0.0,
                    level=None, channel=0, lo=None, hi=None):
    """A raster element of the store -> uint8 volume + meta."""
    import zarr
    key = _key(p=str(path), g=group, n=name, s=stride, u=um_per_px, z=z_um, z0=z0_um,
               l=level, c=channel, lo=lo, hi=hi)
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    g = _open(str(path))
    node = g[group][name]
    ms = _multiscale(node)
    ax = _axes(ms)
    lv = _levels(node)
    level = level or lv[0]
    a = node[level]
    lvl_scale = 2 ** lv.index(level) if level in lv else 1
    arr = np.asarray(a[:])
    if "c" in ax:
        arr = arr[channel]
        ax = [x for x in ax if x != "c"]
    if arr.ndim == 2:
        arr = arr[None]                                   # a 2D element is a one-slice volume
    arr = arr[:, ::stride, ::stride]

    sc, tr = _to_global(ms)
    upp = (um_per_px or 1.0) * stride * lvl_scale
    ox = float(tr[0]) if tr else 0.0
    oy = float(tr[1]) if tr else 0.0

    if group == "labels":
        ids = np.unique(arr)
        ids = ids[ids >= 0]
        names, colors = palette_from_tables(path, len(ids))
        lut = np.zeros(int(arr.max()) + 2, np.uint8)
        classes = []
        for i, v in enumerate(ids[:254], start=1):
            lut[int(v)] = i
            nm = names[int(v)] if names and int(v) < len(names) else f"id {int(v)}"
            col = colors[int(v)] if colors and int(v) < len(colors) else "#cccccc"
            classes.append(dict(id=i, value=int(v), name=nm, color=col,
                                voxels=int((arr == v).sum())))
        vol = lut[np.clip(arr, 0, len(lut) - 1)]
        vol[arr < 0] = 0
        meta = dict(kind="labels", classes=classes, name=f"{group}/{name}")
    else:
        sub = arr[::max(1, arr.shape[0] // 3 or 1), ::8, ::8]
        lo_v = float(np.percentile(sub, 1.0)) if lo is None else float(lo)
        hi_v = float(np.percentile(sub, 99.5)) if hi is None else float(hi)
        sc_v = 255.0 / max(hi_v - lo_v, 1e-6)
        vol = np.clip((arr.astype(np.float32) - lo_v) * sc_v, 0, 255).astype(np.uint8)
        meta = dict(kind="intensity", name=f"{group}/{name}", lo=lo_v, hi=hi_v,
                    color="#9fd0ff")
    meta.update(voxel_um=[upp, upp, float(z_um)], origin_um=[ox, oy, float(z0_um)],
                source=dict(path=str(path), element=f"{group}/{name}", level=level,
                            stride=stride, um_per_px=upp))
    return _write_layer(key, vol, meta)


def layer_from_tif(file, stride=8, z_um=3.0, um_per_px=0.2125, z0_um=0.0, level=None,
                   lo=None, hi=None):
    """An OME-TIFF z-stack (Xenium morphology) -> uint8 intensity volume.

    The pyramid is used rather than the full grid: level 3 of this file is already 1.7 µm/px,
    which is the scale nuclei are read at.
    """
    import tifffile
    key = _key(f=str(file), s=stride, z=z_um, u=um_per_px, z0=z0_um, l=level, lo=lo, hi=hi)
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    with tifffile.TiffFile(str(file)) as t:
        s = t.series[0]
        nlv = len(s.levels)
        level = 0 if level is None else int(level)
        level = max(0, min(level, nlv - 1))
        arr = s.levels[level].asarray()
    if arr.ndim == 2:
        arr = arr[None]
    lvl_scale = arr.shape[-1] and (s.levels[0].shape[-1] / arr.shape[-1])
    arr = arr[:, ::stride, ::stride]
    sub = arr[:, ::4, ::4]
    lo_v = float(np.percentile(sub, 1.0)) if lo is None else float(lo)
    hi_v = float(np.percentile(sub, 99.8)) if hi is None else float(hi)
    vol = np.clip((arr.astype(np.float32) - lo_v) * (255.0 / max(hi_v - lo_v, 1e-6)),
                  0, 255).astype(np.uint8)
    upp = um_per_px * stride * lvl_scale
    meta = dict(kind="intensity", name=Path(file).name, color="#9fd0ff", lo=lo_v, hi=hi_v,
                voxel_um=[upp, upp, float(z_um)], origin_um=[0.0, 0.0, float(z0_um)],
                source=dict(path=str(file), level=level, stride=stride, um_per_px=upp))
    return _write_layer(key, vol, meta)


# ----------------------------------------------------------------- solid surfaces
# A label map cannot be interpolated — a class id halfway between two others means nothing — so
# the solid view is not drawn from the labels themselves.  For every cell type the server builds
# its OCCUPANCY: the fraction of each factor x factor xy block that carries the label, smoothed in
# 3D.  That field is continuous, so the viewer can put an iso-surface through it with a real
# normal and the voxel steps disappear.  It is the recipe of the DMD viewer (`run_3d.py solid`:
# 4x4 blocks, sigma 8 um in xy and 1.125 um in z), so the two views read alike.  The surface
# moves with the smoothing and the threshold: structure is readable in it, counts and single
# cells are not — those stay in the voxel view, which is drawn from the same labels.
SOLID = {}              # (dataset id, layer key) -> {status, done, total, meta | error}
TILES = {}              # same, for label layers served at full resolution


def _is_label_spec(spec):
    return (spec["type"] == "sdata_labels" or
            (spec["type"] == "ssaft" and SSAFT_ARRAYS.get(spec["element"], ("",))[0] == "labels"))


def _solid_spec(d, key):
    """Smoothing parameters for a label layer's solid view, or None if it has none."""
    spec = d["layers"][key]
    if not _is_label_spec(spec):
        return None
    sp = spec.get("solid", {})
    if sp is False:
        return None
    return dict(sigma_xy_um=float(sp.get("sigma_xy_um", 1.0)), sigma_z_um=float(sp.get("sigma_z_um", 1.0)))


def solid_for(spec, factor=4, sigma_xy_um=8.0, sigma_z_um=1.125, progress=None):
    """Per-cell-type smoothed occupancy volumes (+ the tissue envelope) of a label layer.

    One pass over the map, one z slab at a time: every labelled voxel adds 1 to its (class,
    block) count through a single bincount, so beyond the map itself memory is a uint8 count
    volume per class, never a boolean copy of the map per class.  A 2D map is a one-slab volume.
    """
    from scipy.ndimage import gaussian_filter
    key = _key(**_label_key(spec), f=factor, sxy=sigma_xy_um, sz=sigma_z_um, v="solid2")
    cached = CACHE / f"{key}_solid.json"
    if cached.exists():
        return json.loads(cached.read_text())
    src = label_volume(spec)
    vol, names, colors = src["vol"], src["names"], src["colors"]
    Z, Y, X = vol.shape
    C, f = max(len(names), int(vol.max())), int(factor)
    bx, by = -(-X // f), -(-Y // f)
    nb = bx * by
    bid = (np.arange(Y) // f)[:, None] * bx + (np.arange(X) // f)[None, :]   # (y, x) -> block
    counts = np.zeros((C + 1, Z, by, bx), np.uint8)          # row C: any label, the envelope
    total = Z + C + 1
    for z in range(Z):
        sl = vol[z]
        ok = sl > 0
        lab, b = sl[ok].astype(np.int64) - 1, bid[ok]
        counts[:C, z] = np.bincount(lab * nb + b, minlength=C * nb)[:C * nb].reshape(C, by, bx)
        counts[C, z] = np.bincount(b, minlength=nb).reshape(by, bx)
        if progress:
            progress(z + 1, total)
    vx, vy, vz = src["voxel_um"]
    sig = (sigma_z_um / vz, sigma_xy_um / (vy * f), sigma_xy_um / (vx * f))   # (z, y, x) voxels

    def one(row, tag):
        occ = gaussian_filter(counts[row].astype(np.float32) / float(f * f), sig)
        out = np.ascontiguousarray(np.clip(occ * 255.0, 0, 255).astype(np.uint8))   # (z, y, x)
        fn = f"{key}_{tag}.bin.gz"
        with gzip.open(CACHE / fn, "wb", compresslevel=6) as fh:
            fh.write(out.tobytes())
        return dict(file=f"cache/{fn}", peak=int(out.max()), bytes=(CACHE / fn).stat().st_size)

    classes = []
    for k in range(C):
        if counts[k].any():
            classes.append(dict(id=k + 1, name=names[k] if k < len(names) else f"id {k + 1}",
                                color=colors[k] if k < len(colors) else "#cccccc",
                                **one(k, f"c{k + 1}")))
        if progress:
            progress(Z + k + 1, total)
    tissue = one(C, "tissue")
    meta = dict(kind="solid", dims=dict(x=bx, y=by, z=Z), voxel_um=[vx * f, vy * f, vz],
                origin_um=list(src["origin_um"]), factor=f,
                sigma_um=[sigma_xy_um, sigma_xy_um, sigma_z_um], classes=classes, tissue=tissue)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(total, total)
    return meta


# ----------------------------------------------------------------- solid smoothing (GPU when there is one)
# The solid views gaussian-smooth one indicator per class (or per slot) over the whole map.  A 3D map of
# billions of voxels takes the CPU minutes per class, so on a CUDA machine the map is cut into y strips
# with a halo wider than any filter reaches (the gaussian's 4 sigma, the cell reach), each strip smoothed
# on the GPU, and only the strip's own rows kept: the same numbers as the whole-volume CPU pass.
def _gpu():
    """cupy on the CUDA device with the most free memory (the GPUs are shared with other jobs), or None."""
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        free = []
        for i in range(n):                       # a device too full to open even a context counts as 0
            try:
                cp.cuda.runtime.setDevice(i)
                free.append(cp.cuda.runtime.memGetInfo()[0])
            except Exception:                                    # noqa: BLE001
                free.append(0)
        if max(free) < 1e9:
            return None
        cp.cuda.runtime.setDevice(int(np.argmax(free)))
        return cp
    except Exception:                                            # noqa: BLE001
        return None


def _gpu_rows(cp, Z, X, bytes_per_voxel, cap):
    """Rows per strip: at most `cap` voxels, and at most half the device's free memory."""
    free = cp.cuda.runtime.memGetInfo()[0]
    return _strip_rows(Z, X, min(cap, 0.5 * free / bytes_per_voxel))


def _gpu_or_cpu(gpu_fn, cpu_fn):
    """Run on the GPU; if it runs out of memory there (other jobs share it), run on the CPU."""
    cp = _gpu()
    if cp is not None:
        try:
            return gpu_fn(cp)
        except Exception as e:                                   # noqa: BLE001
            if "OutOfMemory" not in type(e).__name__ and "out of memory" not in str(e).lower():
                raise
            print(f"GPU out of memory ({e}); falling back to the CPU", flush=True)
            cp.get_default_memory_pool().free_all_blocks()
    return cpu_fn()


def _strips(Y, rows, halo):
    for y0 in range(0, Y, rows):
        y1 = min(Y, y0 + rows)
        a0, a1 = max(0, y0 - halo), min(Y, y1 + halo)
        yield y0, y1, a0, a1


def _strip_rows(Z, X, budget):
    return max(16, int(budget // max(1, Z * X)))


def _lead_by_class(vol, present, sig, progress=None, total=None):
    """(lead class, its occupancy 0..255) per voxel: each class's indicator gaussian-smoothed."""
    return _gpu_or_cpu(lambda cp: _lead_by_class_impl(vol, present, sig, progress, total, cp),
                       lambda: _lead_by_class_impl(vol, present, sig, progress, total, None))


def _lead_by_class_impl(vol, present, sig, progress, total, cp):
    lead = np.zeros(vol.shape, np.uint8)
    val = np.zeros(vol.shape, np.uint8)
    if cp is None:
        from scipy.ndimage import gaussian_filter
        best = np.zeros(vol.shape, np.float32)
        for i, k in enumerate(present):
            occ = gaussian_filter((vol == k).astype(np.float32), sig, mode="nearest")
            win = occ > best
            best[win] = occ[win]
            lead[win] = k
            del occ, win
            if progress:
                progress(i + 1, total)
        val[:] = np.clip(np.rint(best * 255.0), 0, 255).astype(np.uint8)
        return lead, val
    from cupyx.scipy.ndimage import gaussian_filter as ggf
    Z, Y, X = vol.shape
    halo = int(4.0 * sig[1] + 0.5) + 2
    strips = list(_strips(Y, _gpu_rows(cp, Z, X, 20, 2.0e8), halo))
    for n, (y0, y1, a0, a1) in enumerate(strips):
        v = cp.asarray(vol[:, a0:a1])
        best = cp.zeros(v.shape, cp.float32)
        ld = cp.zeros(v.shape, cp.uint8)
        for k in present:
            occ = ggf((v == int(k)).astype(cp.float32), sig, mode="nearest")
            win = occ > best
            best = cp.where(win, occ, best)
            ld = cp.where(win, cp.uint8(k), ld)
            del occ, win
        c = np.s_[:, y0 - a0:y1 - a0]
        lead[:, y0:y1] = ld[c].get()
        val[:, y0:y1] = cp.clip(cp.rint(best[c] * 255.0), 0, 255).astype(cp.uint8).get()
        del v, best, ld
        cp.get_default_memory_pool().free_all_blocks()
        if progress:
            progress(int(len(present) * (n + 1) / len(strips)), total)
    return lead, val


def solid_tiles_for(spec, sigma_xy_um=1.0, sigma_z_um=1.0, tile=2048, ov_factor=4, progress=None):
    """The solid view of a label map, on the map's own lattice.

    Every class's indicator — the empty background included — is gaussian-smoothed, and each voxel
    keeps the class with the largest smoothed occupancy and that occupancy (two bytes).  The
    occupancies of all classes sum to 1, so a class's region {occupancy >= t} for any t >= 0.5 lies
    inside the voxels that class leads: the two bytes carry every class's surface, and the viewer
    draws each visible class's iso-surface from them.  With a sigma of about one voxel that surface
    is the voxel map with its steps rounded off — the same cells, nothing thinned away.  (The earlier
    view smoothed 4 µm blocks by 8 µm, a density picture that kept 70% of the voxels and painted a
    third of its area over other types.)  Served as tiles like the voxel view, with a mode-free
    overview and a coarse translucent tissue envelope."""
    from scipy.ndimage import gaussian_filter
    tile = _auto_tile(spec, 2, tile)
    key = _key(**_label_key(spec), sxy=float(sigma_xy_um), sz=float(sigma_z_um), tile=int(tile),
               o=int(ov_factor), v="stiles1")
    cached = CACHE / f"{key}_stiles.json"
    if cached.exists():
        return json.loads(cached.read_text())
    src = label_volume(spec)
    vol, names, colors = src["vol"], src["names"], src["colors"]
    Z, Y, X = vol.shape
    vx, vy, vz = src["voxel_um"]
    counts = np.bincount(vol.ravel(), minlength=256)
    present = np.nonzero(counts)[0]
    total = len(present) + 3
    sig = (sigma_z_um / vz if Z > 1 else 0.0, sigma_xy_um / vy, sigma_xy_um / vx)
    lead, val = _lead_by_class(vol, present, sig, progress, total)
    classes = [dict(id=k, value=k - 1, name=names[k - 1] if k - 1 < len(names) else f"id {k}",
                    color=colors[k - 1] if k - 1 < len(colors) else "#cccccc",
                    voxels=int(counts[k])) for k in range(1, 256) if counts[k] > 0]
    meta = _write_solid_tiles(key, [lead, val], vol > 0, src, sigma_xy_um, sigma_z_um, classes, tile, ov_factor)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(total, total)
    return meta


def _write_solid_tiles(key, chans, occupied, src, sigma_xy_um, sigma_z_um, classes, tile, ov_factor):
    """Tiles, overview and tissue envelope of a solid view whose voxels carry `chans`:
    [leading key, its occupancy] for a label map, [leading slot, its occupancy, its cell's class]
    for a cell segmentation."""
    from scipy.ndimage import gaussian_filter
    lead, val = chans[0], chans[1]
    Z, Y, X = lead.shape
    vx, vy, vz = src["voxel_um"]
    surfaced = (lead > 0) & (val >= 128)
    if len(chans) > 2:
        surfaced &= chans[2] > 0
    T, tiles, nbytes = int(tile), [], 0
    for ty in range(0, Y, T):
        for tx in range(0, X, T):
            sl = np.s_[:, ty:ty + T, tx:tx + T]
            t = dict(x0=tx, y0=ty, w=int(lead[sl].shape[2]), h=int(lead[sl].shape[1]), empty=not bool(surfaced[sl].any()))
            if not t["empty"]:
                fn = f"{key}_s{ty // T}_{tx // T}.bin.gz"
                with gzip.open(CACHE / fn, "wb", compresslevel=6) as fh:
                    fh.write(np.ascontiguousarray(np.stack([c[sl] for c in chans], -1)).tobytes())
                t.update(file=f"cache/{fn}", bytes=(CACHE / fn).stat().st_size)
                nbytes += t["bytes"]
            tiles.append(t)
    del surfaced
    # overview: in each block the voxel where a real key is strongest, background elsewhere
    f = int(ov_factor)
    by, bx = -(-Y // f), -(-X // f)
    blocks = []
    for i, c in enumerate(chans):
        pad = np.zeros((Z, by * f, bx * f), np.uint8)
        pad[:, :Y, :X] = np.where(lead > 0, c, 0) if i == 1 else c
        blocks.append(pad.reshape(Z, by, f, bx, f).transpose(0, 1, 3, 2, 4).reshape(Z, by, bx, f * f))
        del pad
    idx = blocks[1].argmax(-1)[..., None]
    ov = [np.take_along_axis(bl, idx, -1)[..., 0] for bl in blocks]
    ov[0] = np.where(ov[1] >= 128, ov[0], 0).astype(np.uint8)
    del blocks
    ofn = f"{key}_sov.bin.gz"
    with gzip.open(CACHE / ofn, "wb", compresslevel=6) as fh:
        fh.write(np.ascontiguousarray(np.stack(ov, -1)).tobytes())
    # the tissue envelope stays a coarse, strongly smoothed shell (4 µm blocks, sigma 8 µm)
    tf = 4
    ty_, tx_ = -(-Y // tf), -(-X // tf)
    tb = np.zeros((Z, ty_ * tf, tx_ * tf), np.float32)
    tb[:, :Y, :X] = occupied
    tocc = gaussian_filter(tb.reshape(Z, ty_, tf, tx_, tf).mean(axis=(2, 4)),
                           (1.125 / vz if Z > 1 else 0.0, 8.0 / (vy * tf), 8.0 / (vx * tf)))
    tfn = f"{key}_stissue.bin.gz"
    with gzip.open(CACHE / tfn, "wb", compresslevel=6) as fh:
        fh.write(np.ascontiguousarray(np.clip(tocc * 255.0, 0, 255).astype(np.uint8)).tobytes())
    return dict(kind="solidtiles", channels=len(chans), name=src["name"], dims=dict(x=X, y=Y, z=Z),
                voxel_um=list(src["voxel_um"]), origin_um=list(src["origin_um"]), tile=T, tiles=tiles,
                classes=classes, overview=dict(file=f"cache/{ofn}", factor=f, w=bx, h=by),
                sigma_um=[float(sigma_xy_um), float(sigma_xy_um), float(sigma_z_um)], iso=0.5,
                tissue=dict(file=f"cache/{tfn}", dims=dict(x=tx_, y=ty_, z=Z), voxel_um=[vx * tf, vy * tf, vz],
                            origin_um=list(src["origin_um"]), bytes=(CACHE / tfn).stat().st_size),
                bytes=nbytes + (CACHE / ofn).stat().st_size)


def instance_solid_tiles_for(spec, sigma_xy_um=1.5, sigma_z_um=1.0, tile=2048, ov_factor=4, progress=None):
    """The solid view of a cell segmentation: every cell its own smooth surface, coloured by its type.

    Smoothing each cell type, as the cell-type map's solid view does, would fuse touching cells of one
    type into a single blob, and smoothing each of ~10^5 cells alone is out of reach.  So the cells
    are given SLOTS — a greedy colouring of the graph of cells that come within the smoothing reach
    of one another — and each slot is smoothed like one class: cells sharing a slot never meet, so
    every surface of a slot is one cell.  Each voxel keeps the leading slot, its occupancy, and the
    cell type of that slot's cell there (three bytes); the viewer separates surfaces by slot and
    colours them by type."""
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
    tile = _auto_tile(spec, 3, tile)
    key = _key(**_label_key(spec), sxy=float(sigma_xy_um), sz=float(sigma_z_um), tile=int(tile),
               o=int(ov_factor), v="istiles1")
    cached = CACHE / f"{key}_stiles.json"
    if cached.exists():
        return json.loads(cached.read_text())
    src = instance_volume(spec)
    ids, lut = src["ids"], src["lut"]
    Z, Y, X = ids.shape
    vx, vy, vz = src["voxel_um"]
    rxy = max(1, int(np.ceil(1.5 * sigma_xy_um / min(vx, vy))))
    rz = max(1, int(np.ceil(1.5 * sigma_z_um / vz))) if Z > 1 else 0
    reach = (2 * rz + 1, 2 * rxy + 1, 2 * rxy + 1)
    big = np.iinfo(np.int32).max
    # cells within reach of each other: a voxel's own id against the largest and smallest id around it
    pairs = []
    gp = _gpu_or_cpu(lambda cp: _gpu_pairs(cp, ids, reach, big), lambda: None)
    if gp is not None:
        pairs = gp
    for filt, fill in (() if gp is not None else ((maximum_filter, 0), (minimum_filter, big))):
        src_ids = ids if fill == 0 else np.where(ids > 0, ids, big).astype(np.int32)
        around = filt(src_ids, size=reach, mode="constant", cval=fill)
        del src_ids
        for z in range(Z):
            a_, b_ = ids[z], around[z]
            m = (a_ > 0) & (b_ != a_) & (b_ > 0) & (b_ != big)
            if m.any():
                lo_, hi_ = np.minimum(a_[m], b_[m]).astype(np.int64), np.maximum(a_[m], b_[m]).astype(np.int64)
                pairs.append(np.unique((lo_ << 32) | hi_))
        del around
    if progress:
        progress(1, 20)
    n = int(ids.max(initial=0)) + 1
    e = np.unique(np.concatenate(pairs)) if pairs else np.zeros(0, np.int64)
    ea, eb = e >> 32, e & 0xFFFFFFFF
    frm, to = np.concatenate([ea, eb]), np.concatenate([eb, ea])
    nbr = to[np.argsort(frm, kind="stable")]
    indptr = np.zeros(n + 1, np.int64)
    indptr[1:] = np.cumsum(np.bincount(frm, minlength=n))
    present = np.bincount(ids.ravel(), minlength=n) > 0
    present[0] = False
    slot = np.zeros(n, np.int64)
    for c in np.argsort(-np.diff(indptr), kind="stable"):      # most crowded cells first
        if not present[c]:
            continue
        used = set(slot[nbr[indptr[c]:indptr[c + 1]]].tolist())
        k = 1
        while k in used:
            k += 1
        slot[c] = k
    K = int(slot.max(initial=0))
    if K > 255:
        raise ValueError(f"{K} slots do not fit a byte")
    sig = (sigma_z_um / vz if Z > 1 else 0.0, sigma_xy_um / vy, sigma_xy_um / vx)
    lead, val, cls = _gpu_or_cpu(lambda cp: _gpu_slots(cp, ids, slot.astype(np.uint8), lut, K, sig, reach, progress),
                                 lambda: _cpu_slots(ids, slot.astype(np.uint8), lut, K, sig, reach, progress))
    counts = np.bincount(lut[ids].ravel(), minlength=256)
    names, colors = src["names"], src["colors"]
    return _instance_finish(key, cached, spec, src, ids, lead, val, cls, counts, names, colors, sigma_xy_um,
                            sigma_z_um, tile, ov_factor, K, present, e, progress)


def _cpu_slots(ids, slot8, lut, K, sig, reach, progress):
    from scipy.ndimage import gaussian_filter, maximum_filter
    sv = slot8[ids]
    best = np.zeros(ids.shape, np.float32)
    lead = np.zeros(ids.shape, np.uint8)
    cls = np.zeros(ids.shape, np.uint8)
    for k in range(K + 1):                                       # slot 0: the empty background
        m = sv == k
        occ = gaussian_filter(m.astype(np.float32), sig, mode="nearest")
        win = occ > best
        best[win] = occ[win]
        lead[win] = k
        if k > 0:                                                # the one cell of this slot within reach
            near = maximum_filter(np.where(m, ids, 0), size=reach, mode="constant", cval=0)
            cls[win] = lut[near[win]]
            del near
        del occ, win, m
        if progress:
            progress(2 + int(16 * (k + 1) / (K + 1)), 20)
    val = np.clip(np.rint(best * 255.0), 0, 255).astype(np.uint8)
    return lead, val, cls


def _gpu_pairs(cp, ids, reach, big):
    """Unique (low id << 32 | high id) of cells within `reach` of each other, in GPU strips."""
    from cupyx.scipy.ndimage import maximum_filter as gmax, minimum_filter as gmin
    Z, Y, X = ids.shape
    halo = reach[1] // 2 + 1
    out = []
    for y0, y1, a0, a1 in _strips(Y, _gpu_rows(cp, Z, X, 24, 1.5e8), halo):
        a = cp.asarray(ids[:, a0:a1])
        for filt, fill in ((gmax, 0), (gmin, big)):
            src = a if fill == 0 else cp.where(a > 0, a, big).astype(cp.int32)
            b = filt(src, size=reach, mode="constant", cval=fill)[:, y0 - a0:y1 - a0]
            c = a[:, y0 - a0:y1 - a0]
            m = (c > 0) & (b != c) & (b > 0) & (b != big)
            lo_ = cp.minimum(c[m], b[m]).astype(cp.int64)
            hi_ = cp.maximum(c[m], b[m]).astype(cp.int64)
            if lo_.size:
                out.append(cp.unique((lo_ << 32) | hi_).get())
            del src, b, c, m, lo_, hi_
        del a
        cp.get_default_memory_pool().free_all_blocks()
    return out


def _gpu_slots(cp, ids, slot8, lut, K, sig, reach, progress):
    """Per voxel: the leading slot, its occupancy, and the cell type of that slot's cell there."""
    from cupyx.scipy.ndimage import gaussian_filter as ggf, maximum_filter as gmax
    Z, Y, X = ids.shape
    halo = max(int(4.0 * sig[1] + 0.5), reach[1] // 2) + 2
    lead = np.zeros(ids.shape, np.uint8)
    val = np.zeros(ids.shape, np.uint8)
    cls = np.zeros(ids.shape, np.uint8)
    slot_g, lut_g = cp.asarray(slot8), cp.asarray(lut)
    strips = list(_strips(Y, _gpu_rows(cp, Z, X, 32, 1.2e8), halo))
    for n, (y0, y1, a0, a1) in enumerate(strips):
        a = cp.asarray(ids[:, a0:a1])
        sv = slot_g[a]
        best = cp.zeros(a.shape, cp.float32)
        ld = cp.zeros(a.shape, cp.uint8)
        cl = cp.zeros(a.shape, cp.uint8)
        for k in range(K + 1):                                   # slot 0: the empty background
            m = sv == k
            occ = ggf(m.astype(cp.float32), sig, mode="nearest")
            win = occ > best
            best = cp.where(win, occ, best)
            ld = cp.where(win, cp.uint8(k), ld)
            if k > 0:                                            # the one cell of this slot within reach
                near = gmax(cp.where(m, a, 0), size=reach, mode="constant", cval=0)
                cl = cp.where(win, lut_g[near], cl)
                del near
            del m, occ, win
        c = np.s_[:, y0 - a0:y1 - a0]
        lead[:, y0:y1] = ld[c].get()
        val[:, y0:y1] = cp.clip(cp.rint(best[c] * 255.0), 0, 255).astype(cp.uint8).get()
        cls[:, y0:y1] = cl[c].get()
        del a, sv, best, ld, cl
        cp.get_default_memory_pool().free_all_blocks()
        if progress:
            progress(2 + int(16 * (n + 1) / len(strips)), 20)
    return lead, val, cls


def _instance_finish(key, cached, spec, src, ids, lead, val, cls, counts, names, colors, sigma_xy_um,
                     sigma_z_um, tile, ov_factor, K, present, e, progress):
    classes = [dict(id=k, value=k - 1, name=names[k - 1] if k - 1 < len(names) else f"id {k}",
                    color=colors[k - 1] if k - 1 < len(colors) else "#cccccc",
                    voxels=int(counts[k])) for k in range(1, 256) if counts[k] > 0]
    meta = _write_solid_tiles(key, [lead, val, cls], ids > 0, src, sigma_xy_um, sigma_z_um, classes, tile, ov_factor)
    meta.update(n_slots=K, n_cells=int(present.sum()), n_touching_pairs=int(len(e)))
    cached.write_text(json.dumps(meta))
    if progress:
        progress(20, 20)
    return meta


def _tiled_spec(d, key):
    """A label layer asked for at its native lattice (stride 1) is served as tiles, and so is a
    colour image (after resampling onto the µm lattice)."""
    spec = d["layers"][key]
    if spec["type"] in ("sdata_labels", "sdata_shapes"):
        return True
    if spec["type"] == "sdata_image":
        return bool(spec.get("_rgb"))
    return spec["type"] == "ssaft" and _is_label_spec(spec) and int(spec.get("stride", 1)) == 1


def _mode_downsample(vol, f):
    """(z, y, x) labels -> (z, ceil(y/f), ceil(x/f)): each block's most frequent NON-EMPTY label,
    0 only where the block holds none.  A plain majority, or picking one voxel per block, would
    erase every cell type sparser than a quarter of a block from the overview."""
    Z, Y, X = vol.shape
    by, bx = -(-Y // f), -(-X // f)
    ids = np.unique(vol)
    ids = ids[ids > 0]
    out = np.zeros((Z, by, bx), np.uint8)
    for z in range(Z):
        pad = np.zeros((by * f, bx * f), np.uint8)
        pad[:Y, :X] = vol[z]
        blk = pad.reshape(by, f, bx, f)
        best = np.zeros((by, bx), np.uint8)
        for c in ids:
            n = (blk == c).sum(axis=(1, 3), dtype=np.uint16)
            win = n > best
            out[z][win] = c
            best[win] = n[win]
    return out


def tiles_for(spec, tile=2048, ov_factor=4, progress=None):
    """A label map at its NATIVE lattice, for the voxel view.

    At 1 µm a section is hundreds of millions of voxels — one GPU texture is out of the question
    and a single ray-march loop could not cross it — so the map is cut into tile x tile xy tiles
    (all z levels each), which the viewer loads nearest-first and marches with block skipping,
    plus a mode-downsampled overview it draws in their place until they arrive.  Empty tiles are
    recorded but not written.  The voxel values and colours are exactly those of the map.
    """
    tile = _auto_tile(spec, 1, tile)
    key = _key(**_label_key(spec), tile=int(tile), o=int(ov_factor), v="tiles2")
    cached = CACHE / f"{key}_tiles.json"
    if cached.exists():
        return json.loads(cached.read_text())
    src = label_volume(spec)
    vol, names, colors = src["vol"], src["names"], src["colors"]
    Z, Y, X = vol.shape
    T = int(tile)
    n_tiles = (-(-Y // T)) * (-(-X // T))
    total = n_tiles + 2
    if progress:
        progress(1, total)
    counts = np.bincount(vol.ravel(), minlength=256)
    classes = [dict(id=k, value=k - 1, name=names[k - 1] if k - 1 < len(names) else f"id {k}",
                    color=colors[k - 1] if k - 1 < len(colors) else "#cccccc",
                    voxels=int(counts[k])) for k in range(1, 256) if counts[k] > 0]
    tiles, nbytes, i = [], 0, 1
    for ty in range(0, Y, T):
        for tx in range(0, X, T):
            blk = np.ascontiguousarray(vol[:, ty:ty + T, tx:tx + T])
            t = dict(x0=tx, y0=ty, w=int(blk.shape[2]), h=int(blk.shape[1]), empty=not blk.any())
            if not t["empty"]:
                fn = f"{key}_t{ty // T}_{tx // T}.bin.gz"
                with gzip.open(CACHE / fn, "wb", compresslevel=6) as fh:
                    fh.write(blk.tobytes())
                t.update(file=f"cache/{fn}", bytes=(CACHE / fn).stat().st_size)
                nbytes += t["bytes"]
            tiles.append(t)
            i += 1
            if progress:
                progress(i, total)
    ov = _mode_downsample(vol, int(ov_factor))
    ofn = f"{key}_ov.bin.gz"
    with gzip.open(CACHE / ofn, "wb", compresslevel=6) as fh:
        fh.write(np.ascontiguousarray(ov).tobytes())
    meta = dict(kind="labels", tiled=True, name=src["name"],
                dims=dict(x=X, y=Y, z=Z), voxel_um=list(src["voxel_um"]),
                origin_um=list(src["origin_um"]), tile=T, tiles=tiles, classes=classes,
                overview=dict(file=f"cache/{ofn}", factor=int(ov_factor),
                              w=int(ov.shape[2]), h=int(ov.shape[1])),
                bytes=nbytes + (CACHE / ofn).stat().st_size)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(total, total)
    return meta


def rgb_tiles_for(spec, tile=2048, ov_factor=8, progress=None):
    """A colour image (Visium HD H&E) -> JPEG tiles on the viewer's µm lattice.

    Such an image is usually rotated or flipped against the µm frame (its affine says so), so it
    is RESAMPLED: box-averaged to about the target pixel first, so the bilinear lookup does not
    alias, then read back through the inverse of its own transform at every `um_per_px` µm.  A
    single-channel scan (ssDNA, DAPI) is carried as grey.  Outside the image the tile is filled with
    the background the viewer keys out: white for a stain on a slide, black for fluorescence."""
    import zarr
    from PIL import Image
    from scipy.ndimage import map_coordinates
    um = float(spec.get("um_per_px", 1.0))
    key = _key(t="rgb", p=str(spec["store"]), n=spec["element"], l=spec.get("level"), um=um,
               tile=int(tile), o=int(ov_factor), u=spec.get("um_per_unit"),
               z0=spec.get("z0_um"), zu=spec.get("z_um"), bg=spec.get("background_key"),
               # a plane / channel pick joins the key only when it is made, so the pictures cut
               # before they existed keep their cache
               **({} if spec.get("plane") is None else dict(pl=spec["plane"])),
               **({} if spec.get("channel") is None else dict(c=spec["channel"])),
               mt=_mtime(spec), v=3)                           # v3: read each array's own metadata, not a stale index
    cached = CACHE / f"{key}_rgb.json"
    if cached.exists():
        return json.loads(cached.read_text())
    g = _open(str(spec["store"]))
    node = g["images"][spec["element"]]
    lv = spec.get("level") or _levels(node)[0]
    S = _element_matrix(node, lv)
    S[:2, :] *= _um_per_unit(g, spec)
    a = node[lv]                                             # (c, y, x), or one plane of (c, z, y, x)
    axn = _axes(_multiscale(node))
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
    if arr.dtype != np.uint8:                                # a fluorescence scan is 16-bit: stretch it
        u8 = np.empty(arr.shape, np.uint8)                   # the way an intensity layer does, per
        for c in range(nc):                                  # channel, and by strips (it is gigabytes)
            s8 = arr[c, ::8, ::8]
            lo_v, hi_v = float(np.percentile(s8, 1.0)), float(np.percentile(s8, 99.8))
            g8 = 255.0 / max(hi_v - lo_v, 1e-6)
            for r in range(0, H, 4096):
                u8[c, r:r + 4096] = np.clip((arr[c, r:r + 4096].astype(np.float32) - lo_v) * g8,
                                            0, 255).astype(np.uint8)
        arr = u8
    bg = spec.get("background_key") or ("white" if nc == 3 else "dark")
    fill = 255.0 if bg == "white" else 0.0
    corners = np.array([[x, y, 0, 1] for x in (0, W) for y in (0, H)], float)
    pts = (S @ corners.T).T[:, :2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    OW, OH = int(np.ceil((x1 - x0) / um)), int(np.ceil((y1 - y0) / um))
    L, t0 = S[:2, :2], S[:2, 3]
    Li = np.linalg.inv(L)
    q = max(1, int(um / np.sqrt(abs(np.linalg.det(L)))))
    out = np.empty((OH, OW, nc), np.uint8)
    n_tiles = (-(-OH // tile)) * (-(-OW // tile))
    total = 3 + n_tiles + 1
    X = x0 + (np.arange(OW)[None, :] + 0.5) * um
    for c in range(nc):
        ch = arr[c]
        if q > 1:
            Hq, Wq = H // q * q, W // q * q
            ch = ch[:Hq, :Wq].reshape(Hq // q, q, Wq // q, q).mean(axis=(1, 3), dtype=np.float32)
        else:
            ch = ch.astype(np.float32)
        for r0 in range(0, OH, 512):
            Y = y0 + (np.arange(r0, min(OH, r0 + 512))[:, None] + 0.5) * um
            dx, dy = X - t0[0], Y - t0[1]
            col = Li[0, 0] * dx + Li[0, 1] * dy                # source pixel, corner convention
            row = Li[1, 0] * dx + Li[1, 1] * dy
            v = map_coordinates(ch, [row / q - 0.5, col / q - 0.5], order=1, mode="constant",
                                cval=fill, output=np.float32)
            out[r0:r0 + v.shape[0], :, c] = np.clip(v + 0.5, 0, 255).astype(np.uint8)
        if progress:
            progress(c + 1, total)
    del arr
    if nc == 1:
        out = out[..., 0]
    T, tiles, nbytes, i = int(tile), [], 0, 3
    for ty in range(0, OH, T):
        for tx in range(0, OW, T):
            blk = out[ty:ty + T, tx:tx + T]
            fn = f"{key}_t{ty // T}_{tx // T}.jpg"
            Image.fromarray(blk).save(CACHE / fn, quality=90)
            tiles.append(dict(x0=tx, y0=ty, w=int(blk.shape[1]), h=int(blk.shape[0]), empty=False,
                              file=f"cache/{fn}", bytes=(CACHE / fn).stat().st_size))
            nbytes += tiles[-1]["bytes"]
            i += 1
            if progress:
                progress(i, total)
    f = int(ov_factor)
    ow, oh = -(-OW // f), -(-OH // f)
    ofn = f"{key}_ov.jpg"
    Image.fromarray(out).resize((ow, oh), Image.BOX).save(CACHE / ofn, quality=88)
    meta = dict(kind="rgb", tiled=True, bg_key=bg, name=spec["element"], dims=dict(x=OW, y=OH, z=1),
                voxel_um=[um, um, float(spec.get("z_um", 1.0))],
                origin_um=[float(x0), float(y0), float(spec.get("z0_um", 0.0))],
                tile=T, tiles=tiles, overview=dict(file=f"cache/{ofn}", factor=f, w=ow, h=oh),
                bytes=nbytes + (CACHE / ofn).stat().st_size)
    cached.write_text(json.dumps(meta))
    if progress:
        progress(total, total)
    return meta


def build_tiles():
    """Every catalog label layer served at full resolution, cut once in the background."""
    for d in CATALOG.values():
        for key, spec in d["layers"].items():
            if not _tiled_spec(d, key):
                continue
            st = TILES[(d["id"], key)]
            try:
                build = {"sdata_image": rgb_tiles_for, "sdata_shapes": shapes_for}.get(spec["type"], tiles_for)
                meta = build(spec, progress=lambda i, n, st=st: st.update(done=i, total=n))
                st.update(status="ready", meta=meta)
                if meta["kind"] == "shapes":
                    print(f"shapes ready: {d['id']}/{key}, {meta['n_cells']} cells, {meta['n_vertices']:,} of "
                          f"{meta['n_vertices_raw']:,} vertices, {meta['bytes'] / 1e6:.1f} MB", flush=True)
                    continue
                n = sum(not t["empty"] for t in meta["tiles"])
                print(f"tiles ready: {d['id']}/{key}, {meta['dims']} at {meta['voxel_um']} um, "
                      f"{n}/{len(meta['tiles'])} tiles, {meta['bytes'] / 1e6:.1f} MB", flush=True)
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                st.update(status="error", error=f"{type(e).__name__}: {e}")


def build_solids():
    """Every catalog label layer's solid volumes, built once in the background at start-up, so
    only the very first run of a new dataset makes anyone wait, and the page can show progress."""
    for d in CATALOG.values():
        for key, spec in d["layers"].items():
            sp = _solid_spec(d, key)
            if sp is None:
                continue
            st = SOLID[(d["id"], key)]
            try:
                build = instance_solid_tiles_for if spec.get("table") else solid_tiles_for
                meta = build(spec, progress=lambda i, n, st=st: st.update(done=i, total=n), **sp)
                st.update(status="ready", meta=meta)
                print(f"solid ready: {d['id']}/{key}, {len(meta['classes'])} cell types", flush=True)
            except Exception as e:                               # noqa: BLE001
                traceback.print_exc()
                st.update(status="error", error=f"{type(e).__name__}: {e}")



# ----------------------------------------------------------------- genes
# Every transcript of a dataset as points: one gene at a time, placed in the same µm frame as its
# layers.  The points element is indexed once per store (sorted by gene, shuffled within a gene, so
# the first `GENE_CAP` of a gene are a random sample of it) and each gene is then a slice.
GENES = {}              # dataset id -> {status, done, total, meta | error}
GENE_CAP = 2_000_000    # points sent per gene, at most


ARRAY_PLATFORMS = ("visiumhd", "stereoseq", "stomics", "stratamap")


def _array_genes(d):
    """(genes, the label-map spec, why) when a gene has to be drawn as a grid of lattice bins:
    either the platform captures that way, or this store keeps no transcript coordinates."""
    spec = next((v for v in d["layers"].values() if _is_label_spec(v) and not v.get("table")), None)
    if spec is None:
        return None
    g = _open(str(spec["store"]))
    if "points" in g and list(g["points"].group_keys()):
        return None
    t = _vector_table(g)
    if t is None:
        return None
    why = "array" if (d.get("meta") or {}).get("platform") in ARRAY_PLATFORMS else "no_transcripts"
    return [str(x) for x in np.asarray(g[f"tables/{t}/var"]["gene"][:]).tolist()], spec, why


def _genes_spec(d):
    """(store, points element) of a dataset's transcripts: the manifest's `genes`, or the first
    points element of the store its layers come from.  `"genes": false` switches them off."""
    import zarr
    gs = d.get("genes")
    if gs is False:
        return None
    if isinstance(gs, dict):
        return Path(gs["store"]), gs["element"]
    for spec in d["layers"].values():
        st = spec.get("store")
        if not st or not str(st).endswith(".zarr"):
            continue
        g = _open(str(st))
        if "points" in g:
            names = sorted(g["points"].group_keys())
            if names:
                return Path(st), names[0]
    return None


# ---- array platforms (Visium HD, Stereo-seq, StrataMap) -------------------------------------------
# These have no transcripts to scatter: expression is a count per lattice bin.  The full bin x gene
# matrix is CSR, so one gene would mean reading all of it (hundreds of millions of entries); the
# SSAF-T signal-gene matrix beside it (`ssaft_downsampled_vectors`, the KDE samples the annotation was
# made from) is dense, so a gene is one column.  A gene is served as a small grey image on the
# cell-type map's own lattice, which the viewer draws over the map in the gene's colour.
GRID_MAX = 8192          # a gene image keeps the map's own lattice (2 µm for Visium HD) up to this


def _vector_table(g):
    t = "ssaft_downsampled_vectors"
    if "tables" in g and t in g["tables"] and "spatial" in g[f"tables/{t}"].get("obsm", {}):
        return t
    return None


def _grid_lattice(spec):
    """The lattice a gene image is drawn on: the cell-type map's own, downsampled to GRID_MAX.

    It is read through the SAME placement the map layer uses (`label_volume`), so a gene can never be
    laid out differently from the map — and a map the viewer cannot place (a rotated element) raises
    here too, rather than drawing the gene in the wrong spot."""
    src = label_volume(spec)
    z, h, w = src["vol"].shape
    vx, vy, vz = src["voxel_um"]
    ox, oy, oz = src["origin_um"]
    f = max(1, int(np.ceil(max(w, h) / GRID_MAX)))    # only a map larger than one texture is binned
    return dict(w=-(-w // f), h=-(-h // f), vx=vx * f, vy=vy * f, ox=ox, oy=oy,
                z0=oz, z1=oz + z * vz)


def _sample_scale(g, spec, xy, lat):
    """1 or µm-per-unit, whichever places the samples inside the map's own lattice."""
    upu = _um_per_unit(g, spec)
    if abs(upu - 1.0) < 1e-9:
        return 1.0
    x0, y0 = lat["ox"], lat["oy"]
    x1, y1 = x0 + lat["w"] * lat["vx"], y0 + lat["h"] * lat["vy"]
    sub = xy[:: max(1, len(xy) // 20000)]

    def inside(k):
        x, y = sub[:, 0] * k, sub[:, 1] * k
        return float(((x >= x0) & (x < x1) & (y >= y0) & (y < y1)).mean())
    return 1.0 if inside(1.0) >= inside(upu) else upu


def gene_grid(spec, gene):
    """One gene of an array dataset as (uint8 grid, meta): its signal on the map's lattice."""
    g = _open(str(spec["store"]))
    t = _vector_table(g)
    if t is None:
        raise KeyError("this dataset has no binned expression")
    names = [str(x) for x in np.asarray(g[f"tables/{t}/var"]["gene"][:]).tolist()]
    if gene not in names:
        raise KeyError(gene)
    lat = _grid_lattice(spec)
    key = _key(t="ggrid", p=str(spec["store"]), n=spec["element"], g=gene, mt=_mtime(spec),
               m=GRID_MAX, v=2)                                  # v2: the samples' own units
    cached = CACHE / f"{key}.bin.gz"
    if cached.exists():
        return f"cache/{cached.name}", lat
    col = np.asarray(g[f"tables/{t}/X"][:, names.index(gene)], np.float32)
    xy = np.asarray(g[f"tables/{t}/obsm/spatial"][:], np.float64)
    # The samples' coordinates are µm in some stores and `global` units in others (a Xenium store's
    # global is image pixels).  Take whichever reading puts more of them on the map.
    k = _sample_scale(g, spec, xy, lat)
    ix = np.floor((xy[:, 0] * k - lat["ox"]) / lat["vx"]).astype(np.int64)
    iy = np.floor((xy[:, 1] * k - lat["oy"]) / lat["vy"]).astype(np.int64)
    ok = (ix >= 0) & (iy >= 0) & (ix < lat["w"]) & (iy < lat["h"]) & (col > 0)
    grid = np.zeros(lat["h"] * lat["w"], np.float32)
    np.maximum.at(grid, iy[ok] * lat["w"] + ix[ok], col[ok])      # the strongest sample in each bin
    hi = float(np.percentile(grid[grid > 0], 99.5)) if (grid > 0).any() else 1.0
    out = np.clip(grid * (255.0 / max(hi, 1e-9)), 0, 255).astype(np.uint8)
    with gzip.open(cached, "wb", compresslevel=6) as fh:
        fh.write(out.tobytes())
    return f"cache/{cached.name}", lat


def _points_matrix(node):
    """A points element's (x, y, z) -> the store's `global` frame."""
    txs = dict(node.attrs).get("coordinateTransformations", []) or []
    glob = [t for t in txs if (t.get("output") or {}).get("name") == "global"] or txs[:1]
    M = np.eye(4)
    for t in glob[:1]:
        M = _tx_matrix(t, ["x", "y", "z"])[0] @ M
    return M


def gene_index_for(store, element, progress=None):
    """Index of one points element: float32 (x, y, z) µm, grouped by gene, and each gene's slice."""
    import pyarrow.compute as pc
    import pyarrow.dataset as pds
    import zarr
    pdir = Path(store) / "points" / element
    mt = max((f.stat().st_mtime for f in pdir.rglob("*") if f.is_file()), default=0)
    g = _open(str(store))
    upu = _um_per_unit(g, {})
    key = _key(t="genes", p=str(store), n=element, mt=round(mt, 3), u=upu, v=1)
    jf = CACHE / f"{key}.genes.json"
    if jf.exists():
        return json.loads(jf.read_text())
    node = g["points"][element]
    fk = (dict(node.attrs).get("spatialdata_attrs") or {}).get("feature_key") or "gene"
    ds = pds.dataset(str(pdir / "points.parquet"))
    has_z = "z" in ds.schema.names
    cols = ["x", "y"] + (["z"] if has_z else []) + [fk]
    # Half a billion transcripts is a routine size here, so the index is built in batches and the
    # sorted copy is written straight into the .npy: the whole table as float64, and a second copy
    # of it to sort, is tens of gigabytes — enough for the kernel to kill the server.
    M = _points_matrix(node)
    n = ds.count_rows()
    xyz = np.empty((n, 3), np.float32)
    codes = np.empty(n, np.int32)
    idx, i, step = {}, 0, max(1, n // 50)
    for b in ds.to_batches(columns=cols, batch_size=1 << 22):
        k = b.num_rows
        if not k:
            continue
        P = np.ones((4, k))
        P[0] = b.column("x").to_numpy(zero_copy_only=False)
        P[1] = b.column("y").to_numpy(zero_copy_only=False)
        if has_z:
            P[2] = b.column("z").to_numpy(zero_copy_only=False)
        else:
            P[2] = 0.0
        Q = (M @ P)[:3]
        Q[:2] *= upu
        xyz[i:i + k] = Q.T.astype(np.float32)
        enc = pc.dictionary_encode(b.column(fk).cast("string"))
        lut = np.array([idx.setdefault(str(nm), len(idx)) for nm in enc.dictionary.to_pylist()], np.int32)
        codes[i:i + k] = lut[np.asarray(enc.indices.to_numpy(zero_copy_only=False), np.int64)]
        i += k
        if progress and (i // step) != ((i - k) // step):
            progress(min(49, i // step), 52)
    names = [nm for nm, _ in sorted(idx.items(), key=lambda kv: kv[1])]
    n = i
    xyz, codes = xyz[:n], codes[:n]
    if progress:
        progress(50, 52)
    # grouped by gene, each gene's points in random order (the viewer sends the first GENE_CAP of
    # them, so that prefix has to be a sample of the gene, not a corner of the section)
    order = np.lexsort((np.random.default_rng(0).random(n, dtype=np.float32), codes))
    out = np.lib.format.open_memmap(CACHE / f"{key}.genes.npy", mode="w+",
                                    dtype=np.float32, shape=(n, 3))
    for s0 in range(0, n, 1 << 23):                      # gather in slices: no second copy in memory
        out[s0:s0 + (1 << 23)] = xyz[order[s0:s0 + (1 << 23)]]
    out.flush()
    zmin, zmax = (float(xyz[:, 2].min()), float(xyz[:, 2].max())) if n else (0.0, 0.0)
    del xyz, order, out
    if progress:
        progress(51, 52)
    counts = np.bincount(codes, minlength=len(names))
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    genes = sorted(([nm, int(c), int(o)] for nm, c, o in zip(names, counts, offs)), key=lambda r: (-r[1], r[0]))
    meta = dict(key=key, genes=genes, n=int(n), cap=GENE_CAP, z_um=[zmin, zmax])
    jf.write_text(json.dumps(meta))
    if progress:
        progress(52, 52)
    return meta


_GENE_NPY = {}


def gene_points(meta, gene):
    """(float32 (n, 3) µm points of one gene, its full count)."""
    row = next((r for r in meta["genes"] if r[0] == gene), None)
    if row is None:
        raise KeyError(gene)
    a = _GENE_NPY.get(meta["key"])
    if a is None:
        a = _GENE_NPY[meta["key"]] = np.load(CACHE / f"{meta['key']}.genes.npy", mmap_mode="r")
    _, cnt, off = row
    return np.asarray(a[off:off + min(cnt, GENE_CAP)]), cnt


def _genes_build(d):
    st = GENES[d["id"]]
    try:
        store, el = st["src"]
        meta = _genes_subprocess(store, el, st)
        st.update(status="ready", meta=meta)
        print(f"genes ready: {d['id']}, {len(meta['genes'])} genes, {meta['n']:,} transcripts", flush=True)
    except Exception as e:                                       # noqa: BLE001
        traceback.print_exc()
        st.update(status="error", error=f"{type(e).__name__}: {e}")


def _genes_subprocess(store, element, st):
    """Build the index in a child process, then read it from the cache.

    Indexing hundreds of millions of transcripts is the one job here that can exhaust the machine.
    Out of memory the kernel kills the process, and that is not an exception anything can catch, so
    the job is given its own process: the server loses a gene list, not the site."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--build-genes", str(store), str(element),
           "--serve-dir", str(SERVE)]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    return gene_index_for(store, element)                # now a cache hit


def hvg_genes(store, all_genes):
    """The genes SSAF-T kept for its vector field when it selected highly variable genes (HVG ∪ the
    ontology's markers), from the var of the ssaft_downsampled_vectors table.  None when no selection was
    made: the table then carries the whole panel, and there is nothing to restrict to."""
    import zarr
    g = _open(str(store))
    if "tables" not in g or "ssaft_downsampled_vectors" not in g["tables"]:
        return None
    var = g["tables"]["ssaft_downsampled_vectors"]["var"]
    col = "gene" if "gene" in var else dict(var.attrs).get("_index", "_index")
    if col not in var:
        return None
    node = var[col]
    if isinstance(node, zarr.Group):                            # categorical
        cats = [str(x) for x in np.asarray(node["categories"][:]).tolist()]
        names = [cats[c] for c in np.asarray(node["codes"][:]) if c >= 0]
    else:
        names = [str(x) for x in np.asarray(node[:]).tolist()]
    sel, full = set(names) & set(all_genes), set(all_genes)
    return sorted(sel) if sel and sel < full else None


def genes_status(d):
    """The dataset's gene index: built in the background the first time it is asked for."""
    with LOCK:
        st = GENES.get(d["id"])
        if st is None:
            src = _genes_spec(d)
            st = GENES[d["id"]] = (dict(status="none") if src is None
                                   else dict(status="building", done=0, total=0, src=src))
            if src is not None:
                threading.Thread(target=_genes_build, args=(d,), daemon=True, name="genes").start()
    return st

# ----------------------------------------------------------------- downloads
# The SSAF-T result of a dataset, as one zip of its store's SSAF-T elements: the cell-type map, the
# cell segmentation and their tables, with the store's own group metadata so the zip IS a zarr store.
# Nothing of the vendor's original (images, transcripts, their tables) is redistributed.  The zip is
# built as it is sent — zarr chunks are already compressed, so it is stored, not compressed again,
# and nothing is written to disk.
DL_GROUPS = ("labels", "shapes", "tables")
DOWNLOADS = {}          # store path -> {bytes, files} (the walk is slow on this filesystem)
DL_README = """SSAF-T result — {name}

This zip IS a zarr store: the SSAF-T elements of {dataset}, as produced by the SSAF-T
segmentation-free pipeline and annotated by CeLLMan.

  labels/ssaft_celltype_map        the cell-type map (value k = cellman_celltype_names[k - 1], 0 = no call)
  labels/ssaft_cell_labels_3d      cell ids per voxel, where the run segmented cells in 3D
  shapes/ssaft_cell_boundaries_*   cell outlines (2D, and per z plane in 3D)
  tables/ssaft_cells               one row per cell: its type, volume, centroid

The KDE vector-field samples the annotation was made from (tables/ssaft_downsampled_vectors) are
working data, not a result, and are not included.

Open it WITHOUT unpacking (zarr 3):

    import zarr
    from zarr.storage import ZipStore
    g = zarr.open_group(store=ZipStore("{file}", mode="r"), mode="r")
    m = g["labels/ssaft_celltype_map/s0"]          # the map
    names = [str(x) for x in g["tables/ssaft_cells/uns/cellman_celltype_names"][:]]

Unpacked, the directory is an ordinary zarr store and SpatialData reads it.
The vendor's original data are not included; see the dataset's publication for them.
"""


def _dl_root_json(store, arcs):
    """The store's root metadata with its index cut down to what this zip holds: a reader otherwise
    sees elements that are not in the zip (the vendor's images and points, or the KDE samples left
    out of the result zip) and fails on the first touch."""
    p = Path(store) / "zarr.json"
    if not p.is_file():
        return None
    try:
        j = json.loads(p.read_text())
        cm = (j.get("consolidated_metadata") or {}).get("metadata")
        if isinstance(cm, dict):
            elements = {a.rsplit("/", 1)[0] for a in arcs if a.count("/") >= 2}
            groups = {e.split("/")[0] for e in elements}
            keep = {k: v for k, v in cm.items()
                    if k in groups or any(k == e or e.startswith(k + "/") for e in elements)}
            j["consolidated_metadata"]["metadata"] = keep
        return json.dumps(j).encode()
    except Exception:                                            # noqa: BLE001
        return p.read_bytes()


VECTORS = "ssaft_downsampled_vectors"    # working data of the annotation, not a result


def _ssaft_files(store):
    """(absolute path, path inside the zip) of the store's SSAF-T result elements.  The KDE vector
    samples are left out: they are working data, and 3.5x the size of every result put together."""
    store = Path(store)
    out = []
    for grp in DL_GROUPS:
        g = store / grp
        if not g.is_dir():
            continue
        if (g / "zarr.json").is_file():
            out.append((g / "zarr.json", f"{grp}/zarr.json"))
        for el in sorted(g.iterdir()):
            if not el.is_dir() or not el.name.startswith("ssaft_"):
                continue
            if el.name == VECTORS:
                continue
            for f in sorted(el.rglob("*")):
                if f.is_file():
                    out.append((f, str(f.relative_to(store))))
    return out


def _dl_cache_file():
    return CACHE / "downloads.json"


def _dl_signature(store):
    """Cheap stamp of a store: when it is rewritten, its directories are touched."""
    p = Path(store)
    return round(max([p.stat().st_mtime] + [c.stat().st_mtime for c in p.iterdir() if c.is_dir()]), 3)


def _dl_load_cache():
    try:
        for k, v in json.loads(_dl_cache_file().read_text()).items():
            if Path(k).is_dir() and v.get("sig") == _dl_signature(k):
                DOWNLOADS[k] = {x: y for x, y in v.items() if x != "sig"}
    except Exception:                                            # noqa: BLE001
        pass


def _dl_save_cache():
    with LOCK:
        out = {}
        for k, v in DOWNLOADS.items():
            try:
                out[k] = dict(v, sig=_dl_signature(k))
            except OSError:
                continue
        _dl_cache_file().write_text(json.dumps(out))


def download_stat(store):
    st = DOWNLOADS.get(str(store))
    if st is None:
        fs = _ssaft_files(store)
        st = DOWNLOADS[str(store)] = dict(files=len(fs), bytes=sum(f.stat().st_size for f, _ in fs))
    return st


def _dl_store(d):
    for spec in d["layers"].values():
        if str(spec.get("store", "")).endswith(".zarr"):
            return spec["store"]
    return None


class _Sink:
    """A write-only file for zipfile: the socket is not seekable, so the archive streams out."""

    def __init__(self, w):
        self._w, self._n = w, 0

    def write(self, b):
        self._w.write(b)
        self._n += len(b)
        return len(b)

    def tell(self):
        return self._n

    def flush(self):
        self._w.flush()


def _write_zip(out, files, d, name):
    """Stream the archive: stored (never re-compressed), zip64, with a README beside the store."""
    import zipfile
    sink = _Sink(out)
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        z.writestr("README.txt", DL_README.format(name=(d.get("meta") or {}).get("title", d["id"]),
                                                  dataset=d["id"], file=name))
        root = _dl_root_json(d and _dl_store(d), [a for _, a in files])
        if root:
            z.writestr("zarr.json", root)
        for path, arc in files:
            try:
                z.write(path, arc)
            except FileNotFoundError:                            # the store changed while sending
                continue


def build_downloads():
    """The size of every dataset's result, measured once in the background: the walk is slow enough
    over this filesystem that the website would block on it.  Kept in the cache across restarts and
    re-measured when a store is rewritten."""
    _dl_load_cache()
    for d in CATALOG.values():
        st = _dl_store(d)
        if st:
            try:
                download_stat(st)
            except Exception as e:                               # noqa: BLE001
                print(f"download size failed: {d['id']}: {e}", flush=True)
    _dl_save_cache()
    print(f"download sizes ready: {len(DOWNLOADS)} stores", flush=True)


# ----------------------------------------------------------------- http
def _safe(p):
    q = Path(p).resolve()
    if ROOT not in q.parents and q != ROOT:
        raise ValueError(f"path outside --root: {q}")
    return q


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".gz": "application/octet-stream", ".json": "application/json"}

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(b)

    def _static_allowed(self, path):
        """In single-file mode only the page and the converted layers are served — never the
        portal's other data (the DMD tiles) or its source.  The path is normalised first, so
        `/cache/../data/...` cannot walk out of the cache."""
        if ONLY is None and CATALOG is None:
            return True
        p = posixpath.normpath(unquote(path))
        if p == "/store.html":
            return True
        # dataset thumbnails for the website's cards: images only, one directory, no walking out
        if (p.startswith("/thumbs/") and "/" not in p[len("/thumbs/"):]
                and p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
            return True
        # the modules that read a visitor's own store in their browser: this directory's own
        # files, not anyone's data
        if p.startswith("/local/") and p.endswith(".js"):
            return True
        if not p.startswith("/cache/") or "/" in p[len("/cache/"):]:
            return False
        # the cached JSON still carries the source path; the page only needs the binary
        return CATALOG is None or p.endswith((".bin.gz", ".jpg"))

    def _gate(self):
        """With --token-file, data requests pass only with the secret: as `?t=` on the shared link, as the
        `X-SSAF-T` header the page adds to its own requests, or as the cookie the link sets.  The page
        itself (static viewer code, no data) is always served, so a browser that blocks or withholds the
        cookie — a link clicked in a messenger, an embedded browser — still works through the header;
        the page takes the token out of the address bar.  No redirect: a cross-site redirect chain is
        exactly where browsers withhold cookies.  Returns True when the request has been answered here."""
        self._cookie = None
        if TOKEN is None:
            return False
        import hmac
        from http.cookies import SimpleCookie
        u = urlparse(self.path)
        given = (parse_qs(u.query).get("t") or [""])[0] or self.headers.get("X-SSAF-T", "")
        jar = SimpleCookie(self.headers.get("Cookie", ""))
        cookie = jar[TOKEN_COOKIE].value if TOKEN_COOKIE in jar else ""
        if given and hmac.compare_digest(given, TOKEN):
            self._cookie = f"{TOKEN_COOKIE}={TOKEN}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
            return False
        # the page and its mode flags hold no data: always served, so a refused visitor sees the right page
        if (cookie and hmac.compare_digest(cookie, TOKEN)) or u.path in ("/", "/store.html", "/api/config"):
            return False
        import sys, time
        print(f"{time.strftime('%H:%M:%S')} refused {self.command} {u.path} from {self.client_address[0]}: "
              f"link token {'wrong' if given else 'absent'}, cookie {'wrong' if cookie else 'absent'}, "
              f"agent {self.headers.get('User-Agent', '')[:60]!r}", file=sys.stderr, flush=True)
        body = b"This viewer needs its access link.\n"
        self.send_response(403)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return True

    def do_HEAD(self):
        if self._gate():
            return
        if urlparse(self.path).path == "/":
            self.path = "/store.html"
        if not self._static_allowed(urlparse(self.path).path):
            return self.send_error(404)
        return super().do_HEAD()

    def do_GET(self):
        if self._gate():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/config":
            return self._json(dict(only=str(ONLY) if ONLY else None,
                                   min_level=ONLY_MIN_LEVEL if ONLY else 0,
                                   catalog=CATALOG is not None,
                                   # the store browser's starting directory: only a development server has one
                                   root=str(ROOT) if (CATALOG is None and ONLY is None) else None))
        if CATALOG is not None and u.path in ("/api/browse", "/api/inventory", "/api/layer",
                                              "/api/tif"):
            return self._json(dict(error="catalog mode: datasets are requested by id"), 403)
        if ONLY is not None and u.path in ("/api/browse", "/api/inventory", "/api/layer"):
            return self._json(dict(error="single-file mode: only /api/tif on the one file"), 403)
        try:
            if u.path == "/api/datasets":
                if CATALOG is None:
                    return self._json(dict(error="not in catalog mode"), 404)
                return self._json(dict(datasets=[_public(d) for d in CATALOG.values()]))
            if u.path == "/api/dataset_layer":
                d = (CATALOG or {}).get(q.get("id", ""))
                if d is None or q.get("key") not in d["layers"]:
                    return self._json(dict(error="no such dataset or layer"), 404)
                st = TILES.get((d["id"], q["key"]))
                if st is not None:                               # full resolution: tiled
                    if st["status"] == "ready":
                        m = dict(st["meta"])
                        m.update(key=q["key"], name=d["layers"][q["key"]].get("label", q["key"]))
                        return self._json(m)
                    if st["status"] == "error":
                        return self._json(dict(error="full-resolution build failed: " + st["error"]), 500)
                    return self._json(dict(status="building", done=st["done"], total=st["total"]), 202)
                with LOCK:
                    return self._json(layer_for(d, q["key"]))
            if u.path in ("/api/download", "/api/download_info"):
                d = (CATALOG or {}).get(q.get("id", ""))
                st = _dl_store(d) if d else None
                if st is None:
                    return self._json(dict(error="no such dataset"), 404)
                name = f"{d['id']}_ssaft.zip"
                if u.path == "/api/download_info":
                    return self._json(dict(name=name, **download_stat(st)))
                files = _ssaft_files(st)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.end_headers()                               # streamed: no length, closed at the end
                _write_zip(self.wfile, files, d, name)
                return
            if u.path in ("/api/genes", "/api/gene_points", "/api/gene_grid"):
                d = (CATALOG or {}).get(q.get("id", ""))
                if d is None:
                    return self._json(dict(error="no such dataset"), 404)
                arr = _array_genes(d)                            # a grid, not transcripts
                if arr is not None:
                    genes, spec, why = arr
                    if u.path == "/api/genes":
                        return self._json(dict(kind="grid", why=why, genes=[[g, 0] for g in sorted(genes)],
                                               n=0, cap=0, z_um=[0, 0], hvg=sorted(genes)))
                    with LOCK:
                        file, lat = gene_grid(spec, q.get("gene", ""))
                    return self._json(dict(file=file, **lat))
                st = genes_status(d)
                if st["status"] == "none":
                    return self._json(dict(genes=[], n=0))
                if st["status"] == "error":
                    return self._json(dict(error="gene index failed: " + st["error"]), 500)
                if st["status"] != "ready":
                    return self._json(dict(status="building", done=st["done"], total=st["total"]), 202)
                if u.path == "/api/genes":
                    m = st["meta"]
                    if "hvg" not in st:
                        st["hvg"] = hvg_genes(st["src"][0], [r[0] for r in m["genes"]])
                    return self._json(dict(genes=[r[:2] for r in m["genes"]], n=m["n"], cap=m["cap"],
                                           z_um=m["z_um"], hvg=st["hvg"]))
                pts, cnt = gene_points(st["meta"], q.get("gene", ""))
                b = gzip.compress(pts.astype(np.float32).tobytes(), compresslevel=1)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(b)))
                self.send_header("X-Count", str(cnt))
                self.end_headers()
                self.wfile.write(b)
                return
            if u.path == "/api/dataset_solid":
                st = SOLID.get((q.get("id", ""), q.get("key", "")))
                if st is None:
                    return self._json(dict(error="no solid view for that layer"), 404)
                if st["status"] == "ready":
                    return self._json(st["meta"])
                if st["status"] == "error":
                    return self._json(dict(error="solid build failed: " + st["error"]), 500)
                return self._json(dict(status="building", done=st["done"], total=st["total"]), 202)
            if u.path == "/api/browse":
                d = _safe(q.get("dir", str(ROOT)))
                dirs, stores = [], []
                for c in sorted(d.iterdir()):
                    if not c.is_dir():
                        continue
                    (stores if c.name.endswith(".zarr") else dirs).append(c.name)
                return self._json(dict(dir=str(d), parent=str(d.parent), dirs=dirs, stores=stores))
            if u.path == "/api/inventory":
                return self._json(inventory(_safe(q["path"])))
            if u.path == "/api/layer" and q.get("group") == "ssaft":
                with LOCK:
                    m = layer_from_ssaft(
                        _safe(q["path"]), q["element"], stride=int(q.get("stride", 1)),
                        zmin=q.get("zmin"), zmax=q.get("zmax"),
                        um_per_px=float(q["um_per_px"]) if "um_per_px" in q else None,
                        z_um=float(q["z_um"]) if "z_um" in q else None)
                return self._json(m)
            if u.path == "/api/layer":
                with LOCK:
                    m = layer_from_zarr(
                        _safe(q["path"]), q["group"], q["element"],
                        stride=int(q.get("stride", 1)),
                        um_per_px=float(q["um_per_px"]) if "um_per_px" in q else None,
                        z_um=float(q.get("z_um", 1.0)), z0_um=float(q.get("z0_um", 0.0)),
                        level=q.get("level"), channel=int(q.get("channel", 0)))
                return self._json(m)
            if u.path == "/api/tif":
                f = _safe(q["file"])
                if ONLY is not None:
                    if f != ONLY:
                        return self._json(dict(error="single-file mode: that file is not served"), 403)
                    if int(q.get("level", 3)) < ONLY_MIN_LEVEL:
                        return self._json(dict(error=f"single-file mode: level >= {ONLY_MIN_LEVEL}"), 403)
                with LOCK:
                    m = layer_from_tif(
                        f, stride=int(q.get("stride", 8)),
                        z_um=float(q.get("z_um", 3.0)),
                        um_per_px=float(q.get("um_per_px", 0.2125)),
                        z0_um=float(q.get("z0_um", 0.0)),
                        level=q.get("level"))
                return self._json(m)
        except Exception as e:                                   # noqa: BLE001
            traceback.print_exc()
            return self._json(dict(error=f"{type(e).__name__}: {e}"), 400)
        if u.path == "/":
            if ONLY is not None:                 # open straight onto the one stack
                self.send_response(302)
                self.send_header("Location", "/store.html?tif=" + quote(str(ONLY)) + "&addtif=1")
                return self.end_headers()
            self.path = "/store.html"
        # check the REWRITTEN path: "/" has just become the page, and must be allowed as the page
        if not self._static_allowed(urlparse(self.path).path):
            return self.send_error(404)
        return super().do_GET()

    def end_headers(self):
        if getattr(self, "_cookie", None):
            self.send_header("Set-Cookie", self._cookie)
            self._cookie = None
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 128
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--root", default="/bce/groups/pnucolab/analysis/l202255179/UNNAMED",
                    help="no file outside this directory is read")
    ap.add_argument("--only", default=None,
                    help="single-file mode: serve this one OME-TIFF stack and nothing else")
    ap.add_argument("--datasets", default=None,
                    help="catalog mode: serve only the datasets in this manifest, by id")
    ap.add_argument("--serve-dir", default=None,
                    help="serve the page and keep the cache in this directory instead of the portal's own: "
                         "a private viewer keeps its derived tiles out of the public server's cache")
    ap.add_argument("--build-genes", nargs=2, metavar=("STORE", "ELEMENT"), default=None,
                    help="build one points element's gene index into the cache and exit; the server "
                         "runs itself this way so an out-of-memory index cannot take it down")
    ap.add_argument("--token-file", default=None,
                    help="require the secret in this file (a shared access link ?t=<secret>); read from a file "
                         "so the secret never appears in the process list")
    a = ap.parse_args()
    ROOT = Path(a.root).resolve()
    if a.token_file:
        TOKEN = Path(a.token_file).read_text().strip()
        if len(TOKEN) < 24:
            raise SystemExit("--token-file: the secret must be at least 24 characters")
    if a.serve_dir:
        SERVE = Path(a.serve_dir).resolve()
        CACHE = SERVE / "cache"
        CACHE.mkdir(parents=True, exist_ok=True)
    if a.build_genes:                                    # the child of _genes_subprocess
        gene_index_for(*a.build_genes, progress=lambda i, n: print(f"progress {i} {n}", flush=True))
        raise SystemExit(0)
    if a.only:
        ONLY = _safe(a.only)
        if not ONLY.is_file():
            raise SystemExit(f"--only: not a file: {ONLY}")
    if a.datasets:
        CATALOG = {}
        for d in json.loads(Path(a.datasets).read_text())["datasets"]:
            for k, spec in d["layers"].items():      # fail at start-up, not on a visitor's click
                src = Path(spec.get("file") or spec.get("store")).resolve()
                if not src.exists():
                    raise SystemExit(f"--datasets: {d['id']}/{k}: missing {src}")
                if spec["type"] == "sdata_image":
                    spec["_rgb"] = sdata_is_rgb(spec)
            CATALOG[d["id"]] = d
            for k in d["layers"]:
                if _solid_spec(d, k) is not None:
                    SOLID[(d["id"], k)] = dict(status="building", done=0, total=0)
                if _tiled_spec(d, k):
                    TILES[(d["id"], k)] = dict(status="building", done=0, total=0)
        # the voxel view first: it is what a visitor opens; solids follow
        threading.Thread(target=lambda: (build_tiles(), build_solids(), build_downloads()), daemon=True,
                         name="build").start()
    handler = functools.partial(Handler, directory=str(SERVE))
    with Server((a.bind, a.port), handler) as srv:
        what = (f"ONLY {ONLY}" if ONLY else
                f"CATALOG {len(CATALOG)} dataset(s)" if CATALOG is not None else f"data root {ROOT}")
        print(f"serving {SERVE} ({what}) at http://{a.bind}:{a.port}", flush=True)
        srv.serve_forever()
