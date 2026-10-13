#!/usr/bin/env python
"""A small synthetic SSAF-T store laid out like a real one (Xenium), for trying the app without data.

    python tests/make_fixture.py OUT_DIR      -> OUT_DIR/XENTEST/ssaft_sdata.zarr

Needs spatialdata (written with its own writer, so the format is SpatialData's).  The element names,
frames and table follow a real SSAF-T Xenium store: the `global` frame is image pixels (0.2125 µm), the
transcripts are µm scaled into it, and the SSAF-T elements are written on a 1 µm lattice into it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from anndata import AnnData
from scipy.ndimage import gaussian_filter, maximum_filter
import spatialdata as sd
from spatialdata.models import (Image2DModel, Labels3DModel, PointsModel, ShapesModel, TableModel)
from spatialdata.transformations import Identity, Scale, Sequence, Translation

PX = 0.2125
rng = np.random.default_rng(0)
out = Path(sys.argv[1]) / "XENTEST" / "ssaft_sdata.zarr"
W_UM, H_UM, Z = 420, 330, 8           # the section, µm
Z0 = 3.0                              # the map's bottom, µm
names = ["epithelial cell", "fibroblast", "T cell", "macrophage", "endothelial cell", "B cell"]
colors = ["#e3663e", "#2aa37a", "#2f6fed", "#b14fd8", "#d8a62a", "#4ab8d8"]

# image: two channels (DAPI, boundary), uint16, on the pixel grid
h, w = int(H_UM / PX), int(W_UM / PX)
nuc = maximum_filter((rng.random((h, w)) > 0.9985).astype(np.float32), size=25)
img = np.stack([gaussian_filter(nuc, 3) * 30000 + rng.normal(800, 80, (h, w)),
                gaussian_filter(rng.random((h, w)), 8) * 20000]).clip(0, 65535).astype(np.uint16)
image = Image2DModel.parse(img, dims=("c", "y", "x"), c_coords=["DAPI", "boundary"], scale_factors=[2, 2],
                           transformations={"global": Identity()})

# cell-type map: 1 µm lattice, classes in blobs; 0 = no call
field = gaussian_filter(rng.random((Z, H_UM, W_UM)), (1, 12, 12))
cmap = np.digitize(field, np.quantile(field, np.linspace(0, 1, len(names) + 2)[1:-1])).astype(np.int32)
cmap[:, :, :20] = 0
to_px = Sequence([Scale([1.0, 1 / PX, 1 / PX], axes=("z", "y", "x")), Translation([Z0], axes=("z",))])
celltype_map = Labels3DModel.parse(cmap, dims=("z", "y", "x"), transformations={"global": to_px})

# cells: ellipsoid blobs with ids, a type each
n_cells = 600
cy, cx = rng.uniform(8, H_UM - 8, n_cells), rng.uniform(28, W_UM - 8, n_cells)
cz, r = rng.uniform(1, Z - 1, n_cells), rng.uniform(3, 6, n_cells)
ids = np.zeros((Z, H_UM, W_UM), np.int32)
zz, yy, xx = np.mgrid[0:Z, 0:H_UM, 0:W_UM]
for i in range(n_cells):
    y0, y1, x0, x1 = int(max(0, cy[i] - r[i])), int(cy[i] + r[i]) + 1, int(max(0, cx[i] - r[i])), int(cx[i] + r[i]) + 1
    sl = np.s_[:, y0:y1, x0:x1]
    inside = ((yy[sl] - cy[i]) ** 2 + (xx[sl] - cx[i]) ** 2) / r[i] ** 2 + (zz[sl] - cz[i]) ** 2 / 4 <= 1
    ids[sl][inside & (ids[sl] == 0)] = i + 1
cell_labels = Labels3DModel.parse(ids, dims=("z", "y", "x"), transformations={"global": to_px})
cell_type = rng.integers(0, len(names), n_cells)
um_to_px = Scale([1 / PX, 1 / PX], axes=("x", "y"))
polys = [shapely.Point(x, y).buffer(rr, 12) for x, y, rr in zip(cx, cy, r)]
cid = [f"cell_{i + 1}" for i in range(n_cells)]
b2d = gpd.GeoDataFrame(geometry=polys, index=pd.Index(cid, name="cell_id"))
rows = []
for i in range(n_cells):
    for z in range(Z):
        rr = r[i] * np.sqrt(max(0.0, 1 - (z - cz[i]) ** 2 / 4))
        if rr > 1:
            rows.append(dict(plane_cell_id=f"{cid[i]}_z{z}", cell_id=cid[i], z_index=z, z_um=Z0 + z,
                             geometry=shapely.Point(cx[i], cy[i]).buffer(rr, 12)))
b3d = gpd.GeoDataFrame(rows).set_index("plane_cell_id")
obs = pd.DataFrame(dict(cell_id=cid, label_id=np.arange(1, n_cells + 1, dtype=np.int32),
                        cellman_annotation=pd.Categorical([names[k] for k in cell_type], categories=names),
                        region=pd.Categorical(["ssaft_cell_labels_3d"] * n_cells)), index=cid)
table = TableModel.parse(AnnData(obs=obs, uns=dict(cellman_celltype_names=np.array(names),
                                                    cellman_annotation_colors=np.array(colors))),
                         region="ssaft_cell_labels_3d", region_key="region", instance_key="label_id")

# transcripts, µm, 30 genes; each gene leans to one cell type's ground
genes = [f"GENE{i:02d}" for i in range(30)]
n_tx = 400_000
tx = pd.DataFrame(dict(x=rng.uniform(0, W_UM, n_tx), y=rng.uniform(0, H_UM, n_tx), z=rng.uniform(Z0, Z0 + Z, n_tx)))
under = cmap[np.clip((tx.z - Z0).astype(int), 0, Z - 1), tx.y.astype(int).clip(0, H_UM - 1), tx.x.astype(int).clip(0, W_UM - 1)]
g = np.where(rng.random(n_tx) < 0.7, (under * 5 + rng.integers(0, 5, n_tx)) % 30, rng.integers(0, 30, n_tx))
tx["feature_name"] = pd.Categorical([genes[k] for k in g], categories=genes)
points = PointsModel.parse(tx, coordinates=dict(x="x", y="y", z="z"), feature_key="feature_name",
                           transformations={"global": Scale([1 / PX, 1 / PX, 1.0], axes=("x", "y", "z"))})

sdata = sd.SpatialData(images=dict(morphology_focus=image),
                       labels=dict(ssaft_celltype_map=celltype_map, ssaft_cell_labels_3d=cell_labels),
                       shapes=dict(ssaft_cell_boundaries_2d=ShapesModel.parse(b2d, transformations={"global": um_to_px}),
                                   ssaft_cell_boundaries_3d=ShapesModel.parse(b3d, transformations={"global": um_to_px})),
                       points=dict(transcripts=points), tables=dict(ssaft_cells=table))
out.parent.mkdir(parents=True, exist_ok=True)
sdata.write(out, overwrite=True)
print("wrote", out)
