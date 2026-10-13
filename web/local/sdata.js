/**
 * What is in a SpatialData store, and where each element sits.
 *
 * A store is a zarr group of `images/`, `labels/`, `shapes/` and `points/`, and every element
 * carries its own transform into a shared coordinate system (`global`): an image is written on its
 * pixel lattice, polygons may be in the vendor's own units, transcripts in microns.  Drawing them
 * together means putting each one through ITS transform, never through a pixel size typed in by
 * hand — so this module's job is to find the elements and reduce each transform to one 3x3 matrix
 * over (x, y, 1).
 */
import { openArray, readNode, readStrings } from './zarr.js';
export const IDENTITY = [1, 0, 0, 0, 1, 0];
export const apply = (m, x, y) => [
    m[0] * x + m[1] * y + m[2],
    m[3] * x + m[4] * y + m[5]
];
/** a then b (b ∘ a) */
export const compose = (b, a) => [
    b[0] * a[0] + b[1] * a[3], b[0] * a[1] + b[1] * a[4], b[0] * a[2] + b[1] * a[5] + b[2],
    b[3] * a[0] + b[4] * a[3], b[3] * a[1] + b[4] * a[4], b[3] * a[2] + b[4] * a[5] + b[5]
];
export const invert = (m) => {
    const det = m[0] * m[4] - m[1] * m[3] || 1e-12;
    return [
        m[4] / det, -m[1] / det, (m[1] * m[5] - m[4] * m[2]) / det,
        -m[3] / det, m[0] / det, (m[3] * m[2] - m[0] * m[5]) / det
    ];
};
/** One NGFF transform, reduced to what it does to x and y. `axes` names the element's own axes. */
function matrixOf(t, axes) {
    if (!t || t.type === 'identity')
        return IDENTITY;
    const pick = (list, fallback) => {
        const xi = axes.indexOf('x'), yi = axes.indexOf('y');
        return [list?.[xi] ?? fallback, list?.[yi] ?? fallback];
    };
    if (t.type === 'scale') {
        const [sx, sy] = pick(t.scale, 1);
        return [sx, 0, 0, 0, sy, 0];
    }
    if (t.type === 'translation') {
        const [tx, ty] = pick(t.translation, 0);
        return [1, 0, tx, 0, 1, ty];
    }
    if (t.type === 'sequence')
        return (t.transformations ?? []).reduce((acc, s) => compose(matrixOf(s, axes), acc), IDENTITY);
    if (t.type === 'affine') {
        // rows are the OUTPUT axes, columns the input axes then a constant; both are named by the
        // transform itself, so the row and column for x and y are looked up rather than assumed
        const ins = t.input?.axes?.map((a) => a.name) ?? axes;
        const outs = t.output?.axes?.map((a) => a.name) ?? axes;
        const rows = t.affine;
        const row = (name) => rows[outs.indexOf(name)] ?? [];
        const at = (name, input) => row(name)[ins.indexOf(input)] ?? 0;
        const rx = row('x'), ry = row('y');
        return [
            at('x', 'x'), at('x', 'y'), rx[rx.length - 1] ?? 0,
            at('y', 'x'), at('y', 'y'), ry[ry.length - 1] ?? 0
        ];
    }
    return IDENTITY;
}
/** the transform of an element into `global` (or the only one it has) */
function toGlobal(list, axes) {
    if (!Array.isArray(list) || !list.length)
        return IDENTITY;
    const t = list.find((q) => (q?.output?.name ?? q?.coordinateSystem) === 'global') ?? list[0];
    return matrixOf(t, axes);
}
/**
 * The cell-type palette, from the first table that carries one.
 *
 * SSAF-T writes the names and colours of a mapping into its table's `uns`
 * (`cellman_celltype_names`, `cellman_annotation_colors`), and a label map's value k is that
 * mapping's k-th type (1-based, 0 = no call), which is how the cell-type map is painted.
 */
async function readPalette(src, index) {
    const tables = index
        ? Object.keys(index).filter((k) => k.startsWith('tables/') && !k.slice(7).includes('/')).map((k) => k.slice(7))
        : (await src.list('tables')).filter((n) => !n.startsWith('.') && !n.endsWith('.json'));
    for (const table of tables) {
        const base = `tables/${table}/uns`;
        const names = await readStrings(src, `${base}/cellman_celltype_names`, index ?? undefined).catch(() => null);
        if (!names)
            continue;
        const colours = (await readStrings(src, `${base}/cellman_annotation_colors`, index ?? undefined).catch(() => null)) ??
            (await readStrings(src, `${base}/cellman_celltype_colors`, index ?? undefined).catch(() => null)) ??
            [];
        return { names, colours: colours.slice(0, names.length), table };
    }
    return null;
}
const axesOf = (list) => Array.isArray(list) ? list.map((a) => (typeof a === 'string' ? a : a.name)) : [];
async function readImage(src, group, name, index) {
    const path = `${group}/${name}`;
    const node = await readNode(src, path, index);
    if (!node)
        return null;
    const ome = (node.attrs.ome ?? node.attrs);
    const ms = ome.multiscales?.[0];
    if (!ms)
        return null;
    const axes = axesOf(ms.axes);
    const matrix = toGlobal(ms.coordinateTransformations, axes);
    const levels = [];
    for (const d of ms.datasets ?? []) {
        const array = await openArray(src, `${path}/${d.path}`, index).catch(() => null);
        if (!array)
            continue;
        levels.push({
            path: `${path}/${d.path}`,
            array,
            scale: toGlobal(d.coordinateTransformations, axes),
            width: array.shape[array.shape.length - 1],
            height: array.shape[array.shape.length - 2]
        });
    }
    if (!levels.length)
        return null;
    const ci = axes.indexOf('c'), zi = axes.indexOf('z');
    const channels = ci >= 0 ? levels[0].array.shape[ci] : 1;
    return {
        kind: group === 'labels' ? 'labels' : 'image',
        name,
        path,
        axes,
        levels,
        matrix,
        channels,
        depth: zi >= 0 ? levels[0].array.shape[zi] : 1,
        rgb: group === 'images' && channels === 3
    };
}
async function readTable(src, group, name, index) {
    const path = `${group}/${name}`;
    const node = await readNode(src, path, index);
    if (!node)
        return null;
    const axes = axesOf(node.attrs.axes) || ['x', 'y'];
    const file = group === 'shapes' ? `${path}/shapes.parquet` : `${path}/points.parquet`;
    return {
        kind: group,
        name,
        path,
        file,
        matrix: toGlobal(node.attrs.coordinateTransformations, axes.length ? axes : ['x', 'y'])
    };
}
/**
 * The elements directly under a group.
 *
 * Consolidated metadata (zarr v3) is one flat map keyed by path — `images`, `images/HE`,
 * `images/HE/s0` — so a group's children are the keys one segment below it.  Without it (zarr v2,
 * or a store written without consolidating) the folder is listed instead, which only a store on
 * disk can do; a store served over HTTP has to carry its consolidated metadata.
 */
async function childrenOf(src, group, index) {
    if (index) {
        const prefix = `${group}/`;
        return Object.keys(index)
            .filter((k) => k.startsWith(prefix) && !k.slice(prefix.length).includes('/'))
            .map((k) => k.slice(prefix.length));
    }
    const names = await src.list(group);
    return names.filter((n) => !n.startsWith('.') && !n.endsWith('.json'));
}
export async function openStore(src) {
    const root = await readNode(src, '');
    if (!root)
        throw new Error('this folder is not a zarr store (no zarr.json or .zgroup in it)');
    const index = (root.version === 3 ? (root.meta.consolidated_metadata?.metadata ?? null) : null);
    const elements = [];
    for (const group of ['images', 'labels'])
        for (const name of await childrenOf(src, group, index)) {
            const el = await readImage(src, group, name, index ?? undefined).catch(() => null);
            if (el)
                elements.push(el);
        }
    for (const group of ['shapes', 'points'])
        for (const name of await childrenOf(src, group, index)) {
            const el = await readTable(src, group, name, index ?? undefined).catch(() => null);
            if (el)
                elements.push(el);
        }
    if (!elements.length)
        throw new Error('no images, labels, shapes or points in this store');
    // the view opens on everything an image covers; polygons and points are placed inside it
    let bounds = null;
    for (const el of elements) {
        if (el.kind !== 'image' && el.kind !== 'labels')
            continue;
        const m = compose(el.matrix, el.levels[0].scale);
        const corners = [
            apply(m, 0, 0), apply(m, el.levels[0].width, 0),
            apply(m, 0, el.levels[0].height), apply(m, el.levels[0].width, el.levels[0].height)
        ];
        const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
        const box = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
        bounds = bounds
            ? [Math.min(bounds[0], box[0]), Math.min(bounds[1], box[1]), Math.max(bounds[2], box[2]), Math.max(bounds[3], box[3])]
            : box;
    }
    const palette = await readPalette(src, index).catch(() => null);
    return { source: src, elements, bounds, palette };
}
