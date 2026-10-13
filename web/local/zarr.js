/**
 * Just enough zarr to read a SpatialData store in the browser.
 *
 * Our stores are zarr v3 with `bytes` + `zstd` chunks, but a visitor's store may be v2 (what
 * SpatialData wrote until 0.2) or use gzip, so both layouts and the codecs we can decode without a
 * wasm build are here.  Anything else fails by name — a reader that silently returns zeros would
 * show an empty tissue and look like a bug in the data.
 */
import { decompress as fzstd } from './vendor/fzstd/index.js';
// fzstd sizes the output from the frame header when it is not given one
const zstdDecompress = fzstd;
const json = async (src, path) => {
    const b = await src.bytes(path);
    if (!b)
        return null;
    try {
        return JSON.parse(new TextDecoder().decode(b));
    }
    catch {
        return null;
    }
};
/**
 * Group or array metadata of either zarr version, with v3's attributes unwrapped.
 *
 * `index` is a store's consolidated metadata, keyed by path: when it has the node, nothing is
 * fetched.  A store of a few dozen elements is otherwise a few dozen round trips before anything
 * can be drawn, which over a slow link is the whole wait.
 */
export async function readNode(src, path, index) {
    const inline = index?.[path.replace(/^\/+|\/+$/g, '')];
    if (inline)
        return { version: 3, meta: inline, attrs: (inline.attributes ?? {}) };
    const v3 = await json(src, `${path}/zarr.json`);
    if (v3)
        return { version: 3, meta: v3, attrs: (v3.attributes ?? {}) };
    const [group, array, attrs] = await Promise.all([
        json(src, `${path}/.zgroup`),
        json(src, `${path}/.zarray`),
        json(src, `${path}/.zattrs`)
    ]);
    if (!group && !array)
        return null;
    return { version: 2, meta: array ?? group, attrs: (attrs ?? {}), isArray: Boolean(array) };
}
const DTYPES = {
    '|u1': 'uint8', '|i1': 'int8', '|b1': 'uint8',
    '<u2': 'uint16', '<i2': 'int16', '<u4': 'uint32', '<i4': 'int32',
    '<u8': 'uint64', '<i8': 'int64', '<f4': 'float32', '<f8': 'float64',
    uint8: 'uint8', int8: 'int8', bool: 'uint8', uint16: 'uint16', int16: 'int16',
    uint32: 'uint32', int32: 'int32', uint64: 'uint64', int64: 'int64',
    float32: 'float32', float64: 'float64'
};
export const BYTES_PER = {
    uint8: 1, int8: 1, uint16: 2, int16: 2, uint32: 4, int32: 4,
    uint64: 8, int64: 8, float32: 4, float64: 8
};
function gunzip(raw) {
    // DecompressionStream is in every browser we target; it is the only async step in decoding
    const ds = new DecompressionStream('gzip');
    const stream = new Blob([raw]).stream().pipeThrough(ds);
    return new Response(stream).arrayBuffer().then((b) => new Uint8Array(b));
}
function decoderFor(names) {
    // `bytes`, `transpose` and the `vlen-*` codecs say how values become bytes, not how those bytes
    // are compressed; only what is left can be a compressor
    const codec = names.find((n) => n !== 'bytes' && n !== 'transpose' && !n.startsWith('vlen-'));
    if (!codec)
        return (raw) => raw;
    if (codec === 'zstd')
        return (raw) => zstdDecompress(raw);
    if (codec === 'blosc')
        throw new Error('this store is blosc-compressed, which this page cannot decode yet');
    if (codec === 'gzip' || codec === 'zlib')
        throw new Error('this store is gzip-compressed; chunks are decoded asynchronously (not yet wired)');
    throw new Error(`unsupported zarr codec: ${codec}`);
}
export async function openArray(src, path, index) {
    const node = await readNode(src, path, index);
    if (!node)
        throw new Error(`no zarr array at ${path}`);
    const m = node.meta;
    if (node.version === 3) {
        const codecs = (m.codecs ?? []).map((c) => c.name);
        const dt = typeof m.data_type === 'string' ? m.data_type : m.data_type?.name;
        const dtype = DTYPES[dt];
        if (!dtype)
            throw new Error(`unsupported zarr data type: ${dt}`);
        const sep = m.chunk_key_encoding?.configuration?.separator ?? '/';
        return {
            path,
            shape: m.shape,
            chunks: m.chunk_grid?.configuration?.chunk_shape ?? m.shape,
            dtype,
            littleEndian: true,
            fillValue: Number(m.fill_value ?? 0),
            separator: sep,
            prefix: m.chunk_key_encoding?.name === 'v2' ? '' : 'c',
            decode: decoderFor(codecs),
            attrs: node.attrs
        };
    }
    const dtype = DTYPES[String(m.dtype).replace(/^[|<>]/, (c) => c)] ?? DTYPES[String(m.dtype)];
    if (!dtype)
        throw new Error(`unsupported zarr data type: ${m.dtype}`);
    const comp = m.compressor?.id ?? m.compressor?.name ?? null;
    return {
        path,
        shape: m.shape,
        chunks: m.chunks,
        dtype,
        littleEndian: String(m.dtype)[0] !== '>',
        fillValue: Number(m.fill_value ?? 0),
        separator: m.dimension_separator ?? '.',
        prefix: '',
        decode: comp ? decoderFor([comp]) : (raw) => raw,
        attrs: node.attrs
    };
}
/**
 * A one-dimensional array of strings — the cell-type names and colours a store keeps in a table's
 * `uns`.  zarr v3 writes them with the `vlen-utf8` codec: a count, then a length and its bytes for
 * every string, which is what is unpacked here.  Returns null when the array is not there or is
 * written some other way, and the caller falls back to colours of its own.
 */
export async function readStrings(src, path, index) {
    const node = await readNode(src, path, index);
    if (!node || node.version !== 3)
        return null;
    const m = node.meta;
    const codecs = (m.codecs ?? []).map((c) => c.name);
    if (!codecs.includes('vlen-utf8'))
        return null;
    const sep = m.chunk_key_encoding?.configuration?.separator ?? '/';
    const prefix = m.chunk_key_encoding?.name === 'v2' ? '' : 'c';
    const chunk = m.chunk_grid?.configuration?.chunk_shape?.[0] ?? m.shape[0];
    const decode = decoderFor(codecs);
    const out = [];
    const text = new TextDecoder();
    for (let c = 0; c * chunk < m.shape[0]; c++) {
        const key = prefix ? [prefix, c].join(sep) : String(c);
        const raw = await src.bytes(`${path}/${key}`);
        if (!raw)
            continue;
        const bytes = decode(raw);
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const n = view.getUint32(0, true);
        let p = 4;
        for (let i = 0; i < n && p + 4 <= bytes.byteLength; i++) {
            const len = view.getUint32(p, true);
            p += 4;
            out.push(text.decode(bytes.subarray(p, p + len)));
            p += len;
        }
    }
    return out.length ? out : null;
}
const typed = (dtype, buf, n) => {
    switch (dtype) {
        case 'uint8': return new Uint8Array(buf, 0, n);
        case 'int8': return new Int8Array(buf, 0, n);
        case 'uint16': return new Uint16Array(buf, 0, n);
        case 'int16': return new Int16Array(buf, 0, n);
        case 'uint32': return new Uint32Array(buf, 0, n);
        case 'int32': return new Int32Array(buf, 0, n);
        case 'float32': return new Float32Array(buf, 0, n);
        case 'float64': return new Float64Array(buf, 0, n);
        case 'uint64':
        case 'int64': return new BigInt64Array(buf, 0, n);
        default: throw new Error(`unsupported dtype ${dtype}`);
    }
};
/**
 * One chunk as a typed array, or null where the store holds no chunk for those indices (zarr
 * leaves a chunk of the fill value unwritten).
 */
export async function readChunk(src, a, index) {
    const key = a.prefix ? [a.prefix, ...index].join(a.separator) : index.join(a.separator);
    const raw = await src.bytes(`${a.path}/${key}`);
    if (!raw)
        return null;
    const bytes = a.decode(raw);
    const n = a.chunks.reduce((p, c) => p * c, 1);
    // the decoded buffer may be a view into a larger one; copy only when it is not aligned
    const aligned = bytes.byteOffset % 8 === 0 ? bytes : new Uint8Array(bytes);
    const buf = aligned.buffer;
    return typed(a.dtype, buf.slice(aligned.byteOffset, aligned.byteOffset + n * BYTES_PER[a.dtype]), n);
}
/**
 * A rectangle of one 2D plane of an array, as `width * height` values of the array's own type.
 *
 * `at` fixes every axis but the last two (the channel of an image, the z of a stack); `x0, y0` and
 * the size are in that plane's own pixels, and `step` takes every step-th pixel of them.  Only the
 * chunks the rectangle touches are read.
 */
export async function readPlane(src, a, at, x0, y0, width, height, step = 1) {
    const nd = a.shape.length;
    const [cy, cx] = [a.chunks[nd - 2], a.chunks[nd - 1]];
    const ow = Math.ceil(width / step), oh = Math.ceil(height / step);
    const out = typed(a.dtype, new ArrayBuffer(ow * oh * BYTES_PER[a.dtype]), ow * oh);
    if (a.fillValue)
        out.fill(a.dtype === 'uint64' || a.dtype === 'int64' ? BigInt(a.fillValue) : a.fillValue);
    const lead = at.map((v, i) => Math.floor(v / a.chunks[i]));
    const within = at.map((v, i) => v % a.chunks[i]);
    for (let ty = Math.floor(y0 / cy); ty <= Math.floor((y0 + height - 1) / cy); ty++) {
        for (let tx = Math.floor(x0 / cx); tx <= Math.floor((x0 + width - 1) / cx); tx++) {
            const chunk = await readChunk(src, a, [...lead, ty, tx]);
            if (!chunk)
                continue;
            const planeOffset = within.reduce((off, v, i) => (off + v * a.chunks.slice(i + 1).reduce((p, c) => p * c, 1)), 0);
            const yStart = Math.max(y0, ty * cy), yEnd = Math.min(y0 + height, (ty + 1) * cy, a.shape[nd - 2]);
            const xStart = Math.max(x0, tx * cx), xEnd = Math.min(x0 + width, (tx + 1) * cx, a.shape[nd - 1]);
            // `step` keeps a wide view bounded: every step-th pixel is taken, which is what the
            // screen can show anyway, instead of refusing to draw a window too big to hold
            const firstY = yStart + ((step - ((yStart - y0) % step)) % step);
            const firstX = xStart + ((step - ((xStart - x0) % step)) % step);
            for (let y = firstY; y < yEnd; y += step) {
                const row = planeOffset + (y - ty * cy) * cx - tx * cx;
                const dst = ((y - y0) / step) * ow;
                for (let x = firstX; x < xEnd; x += step)
                    out[dst + (x - x0) / step] = chunk[row + x];
            }
        }
    }
    return out;
}
