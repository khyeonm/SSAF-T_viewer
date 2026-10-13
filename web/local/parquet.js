/**
 * The parquet side of a SpatialData store: cell outlines (`shapes/<name>/shapes.parquet`) and
 * transcripts (`points/<name>/points.parquet`).
 *
 * Both are read straight from the visitor's disk through `StoreSource`, by byte range — a parquet
 * reader asks for the footer, then only the column chunks it was asked for, so opening a 25 MB
 * transcript table costs a few hundred kilobytes until a column is actually wanted.
 */
import { parquetMetadataAsync, parquetReadObjects } from './vendor/hyparquet/index.js';
import { decompress as zstdDecompress } from './vendor/fzstd/index.js';
// SNAPPY is built into hyparquet; zstd is the other codec a store is likely to use
const compressors = {
    ZSTD: (input, outputLength) => zstdDecompress(input, new Uint8Array(outputLength))
};
/** hyparquet reads through this: a length and ranged slices, which is exactly what a File gives. */
async function asyncBuffer(src, path) {
    const byteLength = await src.size(path);
    if (byteLength === null)
        return null;
    return {
        byteLength,
        async slice(start, end) {
            const bytes = await src.range(path, start, end ?? byteLength);
            return (bytes ?? new Uint8Array(0)).buffer;
        }
    };
}
/** does this path hold a parquet file, rather than a directory of them? */
async function isParquet(src, path) {
    const size = await src.size(path);
    if (size === null || size < 8)
        return false;
    // a directory answers a HEAD with its listing's length, so the magic at the end decides it
    const tail = await src.range(path, size - 4, size);
    return Boolean(tail && tail[0] === 0x50 && tail[1] === 0x41 && tail[2] === 0x52 && tail[3] === 0x31);
}
/**
 * The parts of one logical table.  Dask writes a DIRECTORY, `points.parquet/part.0.parquet`; a
 * table small enough to hold in one piece is written as a file under the same name.  Either way
 * the caller gets a list of files to read in order.
 *
 * A directory on disk is listed.  Over HTTP there is nothing to list, so the parts are probed by
 * the name dask gives them, which is how the sample store on this site is read.
 */
export async function parts(src, file) {
    if (await isParquet(src, file))
        return [file];
    const listed = (await src.list(file)).filter((n) => n.endsWith('.parquet')).sort();
    if (listed.length)
        return listed.map((n) => `${file}/${n}`);
    const probed = [];
    for (let i = 0; i < 1024; i++) {
        const path = `${file}/part.${i}.parquet`;
        if (!(await isParquet(src, path)))
            break;
        probed.push(path);
    }
    return probed;
}
/** Up to `limit` rows of the named columns, across every part of the table. */
export async function readTable(src, file, columns, limit) {
    const files = await parts(src, file);
    if (!files.length)
        throw new Error(`no parquet file at ${file}`);
    const rows = [];
    let total = 0;
    for (const path of files) {
        const buffer = await asyncBuffer(src, path);
        if (!buffer)
            continue;
        const metadata = await parquetMetadataAsync(buffer);
        const n = Number(metadata.num_rows);
        total += n;
        const want = Math.min(n, Math.max(0, limit - rows.length));
        if (want > 0) {
            const got = await parquetReadObjects({
                file: buffer,
                metadata,
                columns,
                compressors,
                rowStart: 0,
                rowEnd: want
            });
            for (const row of got)
                rows.push(row);
        }
    }
    return { rows, total, truncated: rows.length < total };
}
/** The column names of a table, without reading any of its data. */
export async function columnsOf(src, file) {
    const files = await parts(src, file);
    if (!files.length)
        return [];
    const buffer = await asyncBuffer(src, files[0]);
    if (!buffer)
        return [];
    const metadata = await parquetMetadataAsync(buffer);
    return metadata.schema.slice(1).map((s) => s.name);
}
