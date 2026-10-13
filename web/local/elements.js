/**
 * Elements as the canvas wants them: an image as pixels, outlines as rings of global coordinates,
 * transcripts as one flat array of points with a gene per point.
 *
 * Everything here places its output in the GLOBAL frame through the element's own transform, so the
 * viewer can draw an image, the polygons cut from it and the transcripts under it in one pass
 * without knowing where any of them came from.
 */
import { apply, compose, invert } from './sdata.js';
import { readPlane, BYTES_PER } from './zarr.js';
import { readTable } from './parquet.js';
/* ------------------------------------------------------------------ images */
/** the level whose pixels are closest to (but still finer than) one screen pixel */
export function pickLevel(el, unitsPerPixel) {
    let best = 0;
    for (let i = 0; i < el.levels.length; i++) {
        const m = compose(el.matrix, el.levels[i].scale);
        const size = Math.hypot(m[0], m[3]); // global units covered by one level pixel
        if (size <= unitsPerPixel * 1.05)
            best = i;
    }
    return best;
}
const rgbOf = (hex) => {
    if (!hex)
        return null;
    const h = hex.trim().replace('#', '');
    const full = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
    if (full.length < 6)
        return null;
    const n = Number.parseInt(full.slice(0, 6), 16);
    return Number.isNaN(n) ? null : [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
/**
 * Paint a label map: every value its cell type's colour, 0 (no call) transparent.
 *
 * `hidden` holds the values the viewer is leaving out, which are drawn transparent rather than
 * removed, so the layers underneath show through where that type was.
 */
export function paintLabels(values, width, height, colours, hidden) {
    const out = new ImageData(width, height);
    const px = out.data;
    const lut = new Map();
    for (let i = 0, n = width * height; i < n; i++) {
        const v = Number(values[i]);
        if (!v || hidden?.has(v))
            continue;
        let colour = lut.get(v);
        if (colour === undefined) {
            colour = rgbOf(colours(v)) ?? labelColour(v);
            lut.set(v, colour);
        }
        if (!colour)
            continue;
        px[i * 4] = colour[0];
        px[i * 4 + 1] = colour[1];
        px[i * 4 + 2] = colour[2];
        px[i * 4 + 3] = 255;
    }
    return out;
}
/** how many pixels each label value covers, for listing the types a view actually holds */
export function labelCounts(values, n) {
    const counts = new Map();
    for (let i = 0; i < n; i++) {
        const v = Number(values[i]);
        if (v)
            counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    return counts;
}
/**
 * The part of an image that covers a window in global units, as pixels.
 *
 * `channel` picks one channel of a multi-channel image, or -1 for an RGB image drawn as colour.
 * 16-bit data is stretched between the 1st and 99.5th percentile of the window itself, the same
 * way the main viewer shows a stain: a nuclear image has a long dark tail and a fixed window
 * would show almost nothing.
 */
export async function loadImageWindow(src, el, level, window, options = {}) {
    const lv = el.levels[Math.min(level, el.levels.length - 1)];
    const toGlobal = compose(el.matrix, lv.scale);
    const toPixels = invert(toGlobal);
    const corners = [
        apply(toPixels, window[0], window[1]), apply(toPixels, window[2], window[1]),
        apply(toPixels, window[0], window[3]), apply(toPixels, window[2], window[3])
    ];
    const x0 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[0]))));
    const y0 = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[1]))));
    const x1 = Math.min(lv.width, Math.ceil(Math.max(...corners.map((c) => c[0]))));
    const y1 = Math.min(lv.height, Math.ceil(Math.max(...corners.map((c) => c[1]))));
    const width = x1 - x0, height = y1 - y0;
    if (width <= 0 || height <= 0)
        return null;
    // a wide view of a big element is read every step-th pixel rather than refused: the screen
    // holds a couple of million pixels, so nothing is lost, and the view is never empty
    const cap = options.maxPixels ?? 8e6;
    const step = Math.max(1, Math.ceil(Math.sqrt((width * height) / cap)));
    const nd = lv.array.shape.length;
    const lead = [];
    for (let i = 0; i < nd - 2; i++) {
        const axis = el.axes[i];
        lead.push(axis === 'c' ? Math.max(0, options.channel ?? 0) : axis === 'z' ? (options.z ?? Math.floor(el.depth / 2)) : 0);
    }
    const rgb = el.rgb && (options.channel ?? -1) < 0;
    const planes = [];
    for (const c of rgb ? [0, 1, 2] : [null]) {
        const at = lead.slice();
        if (c !== null) {
            const ci = el.axes.indexOf('c');
            if (ci >= 0 && ci < nd - 2)
                at[ci] = c;
        }
        planes.push(await readPlane(src, lv.array, at, x0, y0, width, height, step));
    }
    const w = Math.ceil(width / step), h = Math.ceil(height / step);
    const out = new ImageData(w, h);
    const px = out.data;
    if (rgb) {
        for (let i = 0, n = w * h; i < n; i++) {
            px[i * 4] = planes[0][i];
            px[i * 4 + 1] = planes[1][i];
            px[i * 4 + 2] = planes[2][i];
            px[i * 4 + 3] = 255;
        }
    }
    else if (el.kind === 'labels') {
        const offset = [step, 0, x0, 0, step, y0];
        return {
            data: paintLabels(planes[0], w, h, options.colours ?? (() => undefined), options.hidden),
            matrix: compose(toGlobal, offset),
            labels: { values: planes[0], width: w, height: h }
        };
    }
    else {
        const [lo, hi] = stretch(planes[0]);
        const span = hi > lo ? hi - lo : 1;
        for (let i = 0, n = w * h; i < n; i++) {
            const v = Math.max(0, Math.min(255, ((Number(planes[0][i]) - lo) / span) * 255));
            px[i * 4] = v;
            px[i * 4 + 1] = v;
            px[i * 4 + 2] = v;
            px[i * 4 + 3] = 255;
        }
    }
    // the tile's own pixels start at (x0, y0) of this level, and each one covers `step` of them
    const offset = [step, 0, x0, 0, step, y0];
    return { data: out, matrix: compose(toGlobal, offset) };
}
/** 1st and 99.5th percentile, from a sample: the stretch a stain needs to be visible at all */
function stretch(plane) {
    const step = Math.max(1, Math.floor(plane.length / 50000));
    const sample = [];
    for (let i = 0; i < plane.length; i += step)
        sample.push(Number(plane[i]));
    if (!sample.length)
        return [0, 255];
    sample.sort((a, b) => a - b);
    return [sample[Math.floor(sample.length * 0.01)], sample[Math.floor(sample.length * 0.995)]];
}
/** a stable, well-spread colour per label id (ids are arbitrary, so the hue is hashed) */
export function labelColour(id) {
    const h = ((id * 2654435761) % 360 + 360) % 360;
    const s = 0.55 + ((id * 97) % 30) / 100;
    const v = 0.75 + ((id * 57) % 20) / 100;
    const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
    const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
        : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}
/** WKB (what geoparquet stores) -> rings, already in global units */
function wkbRings(bytes, m, out, circles) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let p = 0;
    const u8 = () => view.getUint8(p++);
    let le = true;
    const u32 = () => { const v = view.getUint32(p, le); p += 4; return v; };
    const f64 = () => { const v = view.getFloat64(p, le); p += 8; return v; };
    function geometry(depth = 0) {
        if (depth > 4 || p >= bytes.byteLength)
            return;
        le = u8() === 1;
        const type = u32();
        const base = type % 1000; // 1000/2000/3000 are the Z/M/ZM variants
        const dims = type >= 3000 ? 4 : type >= 1000 ? 3 : 2;
        const skip = dims - 2;
        if (base === 1) { // point
            const x = f64(), y = f64();
            for (let d = 0; d < skip; d++)
                f64();
            const [gx, gy] = apply(m, x, y);
            circles.push({ x: gx, y: gy, r: 0 });
            return;
        }
        if (base === 2 || base === 3) { // line string / polygon
            const nRings = base === 3 ? u32() : 1;
            for (let r = 0; r < nRings; r++) {
                const n = u32();
                const ring = new Float32Array(n * 2);
                for (let i = 0; i < n; i++) {
                    const x = f64(), y = f64();
                    for (let d = 0; d < skip; d++)
                        f64();
                    const [gx, gy] = apply(m, x, y);
                    ring[i * 2] = gx;
                    ring[i * 2 + 1] = gy;
                }
                out.push(ring);
            }
            return;
        }
        if (base >= 4 && base <= 7) { // multi* and collections
            const n = u32();
            for (let i = 0; i < n; i++)
                geometry(depth + 1);
            return;
        }
    }
    geometry();
}
/** GeoJSON (what a reader hands back for a geoparquet column) -> rings in global units */
function geoRings(geom, m, out, circles, radius = 0) {
    const ring = (coords) => {
        const arr = new Float32Array(coords.length * 2);
        for (let i = 0; i < coords.length; i++) {
            const [gx, gy] = apply(m, coords[i][0], coords[i][1]);
            arr[i * 2] = gx;
            arr[i * 2 + 1] = gy;
        }
        out.push(arr);
    };
    const point = (c) => {
        const [gx, gy] = apply(m, c[0], c[1]);
        circles.push({ x: gx, y: gy, r: radius * Math.hypot(m[0], m[3]) });
    };
    switch (geom?.type) {
        case 'Polygon':
            (geom.coordinates ?? []).forEach(ring);
            return;
        case 'MultiPolygon':
            (geom.coordinates ?? []).forEach((poly) => poly.forEach(ring));
            return;
        case 'LineString':
            ring(geom.coordinates ?? []);
            return;
        case 'MultiLineString':
            (geom.coordinates ?? []).forEach(ring);
            return;
        case 'Point':
            point(geom.coordinates ?? [0, 0]);
            return;
        case 'MultiPoint':
            (geom.coordinates ?? []).forEach(point);
            return;
        case 'GeometryCollection':
            (geom.geometries ?? []).forEach((g) => geoRings(g, m, out, circles, radius));
            return;
    }
}
export async function loadShapes(src, el, limit = 200000) {
    const { rows, total, truncated } = await readTable(src, el.file, undefined, limit);
    const rings = [];
    const circles = [];
    let count = 0;
    for (const row of rows) {
        const geom = row.geometry;
        const radius = Number(row.radius ?? 0);
        if (geom && typeof geom === 'object' && 'type' in geom) {
            // a geoparquet column comes back as GeoJSON; a circle keeps its radius in its own column
            geoRings(geom, el.matrix, rings, circles, radius);
            count++;
        }
        else if (geom instanceof Uint8Array) {
            wkbRings(geom, el.matrix, rings, circles); // raw WKB: no geo metadata to convert it
            count++;
        }
        else if (Number.isFinite(Number(row.x)) && Number.isFinite(Number(row.y))) {
            const [gx, gy] = apply(el.matrix, Number(row.x), Number(row.y));
            circles.push({ x: gx, y: gy, r: radius * Math.hypot(el.matrix[0], el.matrix[3]) });
            count++;
        }
    }
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const ring of rings)
        for (let i = 0; i < ring.length; i += 2) {
            if (ring[i] < x0)
                x0 = ring[i];
            if (ring[i] > x1)
                x1 = ring[i];
            if (ring[i + 1] < y0)
                y0 = ring[i + 1];
            if (ring[i + 1] > y1)
                y1 = ring[i + 1];
        }
    for (const c of circles) {
        x0 = Math.min(x0, c.x - c.r);
        x1 = Math.max(x1, c.x + c.r);
        y0 = Math.min(y0, c.y - c.r);
        y1 = Math.max(y1, c.y + c.r);
    }
    return {
        rings,
        circles,
        count,
        total,
        truncated,
        bounds: Number.isFinite(x0) ? [x0, y0, x1, y1] : [0, 0, 0, 0]
    };
}
const GENE_COLUMNS = ['gene', 'feature_name', 'target', 'gene_name', 'genes'];
export async function loadPoints(src, el, limit = 2000000) {
    const { rows, total, truncated } = await readTable(src, el.file, undefined, limit);
    const n = rows.length;
    const xy = new Float32Array(n * 2);
    const key = n ? GENE_COLUMNS.find((c) => c in rows[0]) : undefined;
    const names = [];
    const index = new Map();
    const gene = key ? new Uint16Array(n) : null;
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        const row = rows[i];
        const [gx, gy] = apply(el.matrix, Number(row.x), Number(row.y));
        xy[i * 2] = gx;
        xy[i * 2 + 1] = gy;
        if (gx < x0)
            x0 = gx;
        if (gx > x1)
            x1 = gx;
        if (gy < y0)
            y0 = gy;
        if (gy > y1)
            y1 = gy;
        if (gene && key) {
            const name = String(row[key]);
            let g = index.get(name);
            if (g === undefined) {
                g = names.length;
                if (g > 65535)
                    continue; // a panel never has this many; a stray column might
                index.set(name, g);
                names.push(name);
            }
            gene[i] = g;
        }
    }
    const counts = new Array(names.length).fill(0);
    if (gene)
        for (let i = 0; i < n; i++)
            counts[gene[i]]++;
    return {
        xy,
        gene,
        genes: names,
        counts,
        total,
        truncated,
        bounds: Number.isFinite(x0) ? [x0, y0, x1, y1] : [0, 0, 0, 0]
    };
}
export { BYTES_PER };
