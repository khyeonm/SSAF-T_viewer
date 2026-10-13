/**
 * Where a store's bytes come from.
 *
 * The page reads a SpatialData store the visitor picked on their OWN machine: nothing is uploaded,
 * and the site never sees a path or a byte of it.  A picked directory arrives as a
 * FileSystemDirectoryHandle and every read below
 * goes through it.  The same interface over `fetch` serves the sample store this site publishes,
 * so the readers, and the viewer on top of them, have one code path for both.
 *
 * Reads are by byte range because that is what parquet needs: its footer first, then the column
 * chunks the view actually asks for, never the whole file.
 */
const clean = (path) => path.replace(/^\/+|\/+$/g, '');
/** A store the visitor picked in the file dialog. Their machine, their files, read lazily. */
export function directorySource(handle, name) {
    const dirs = new Map();
    async function dir(path) {
        const key = clean(path);
        if (!key)
            return handle;
        let got = dirs.get(key);
        if (!got) {
            got = (async () => {
                const parts = key.split('/');
                let node = handle;
                for (const part of parts) {
                    if (!node)
                        return null;
                    node = await node.getDirectoryHandle(part).catch(() => null);
                }
                return node;
            })();
            dirs.set(key, got);
        }
        return got;
    }
    async function file(path) {
        const parts = clean(path).split('/');
        const parent = await dir(parts.slice(0, -1).join('/'));
        if (!parent)
            return null;
        const fh = await parent.getFileHandle(parts[parts.length - 1]).catch(() => null);
        return fh ? await fh.getFile() : null;
    }
    return {
        name: name ?? handle.name,
        async bytes(path) {
            const f = await file(path);
            return f ? new Uint8Array(await f.arrayBuffer()) : null;
        },
        async range(path, start, end) {
            const f = await file(path);
            return f ? new Uint8Array(await f.slice(start, end).arrayBuffer()) : null;
        },
        async size(path) {
            const f = await file(path);
            return f ? f.size : null;
        },
        async list(path) {
            const d = await dir(path);
            if (!d)
                return [];
            const names = [];
            for await (const key of d.keys())
                names.push(key);
            return names;
        }
    };
}
/**
 * A folder handed over by `<input type="file" webkitdirectory>`.
 *
 * It is what Firefox and Safari offer in place of a directory handle: the browser lists every file
 * in the folder up front and gives the page a File for each.  A File is a reference, not a copy —
 * the bytes are still read on demand, and still never leave the machine — but the listing itself
 * is eager, so a store with a hundred thousand chunks takes a moment to open, and there is no
 * handle to keep, so the folder has to be picked again next time.
 */
export function fileListSource(files, name) {
	const byPath = new Map();
	for (const file of files) {
		const parts = (file.webkitRelativePath || file.name).split('/');
		// the first segment is the folder the visitor picked; paths inside the store go below it
		byPath.set(parts.slice(1).join('/') || file.name, file);
	}
	const of = (path) => byPath.get(clean(path)) ?? null;
	return {
		name: name || files[0]?.webkitRelativePath?.split('/')[0] || 'store',
		async bytes(path) {
			const f = of(path);
			return f ? new Uint8Array(await f.arrayBuffer()) : null;
		},
		async range(path, start, end) {
			const f = of(path);
			return f ? new Uint8Array(await f.slice(start, end).arrayBuffer()) : null;
		},
		async size(path) {
			return of(path)?.size ?? null;
		},
		async list(path) {
			const prefix = clean(path) ? `${clean(path)}/` : '';
			const names = new Set();
			for (const key of byPath.keys())
				if (key.startsWith(prefix)) {
					const rest = key.slice(prefix.length);
					if (rest) names.add(rest.split('/')[0]);
				}
			return [...names];
		}
	};
}

/**
 * A store served over HTTP: the sample this site publishes, and what the tests drive.  A directory
 * has no listing over HTTP, so `list` is only ever asked about paths the metadata already named —
 * zarr v3 consolidated metadata and the SpatialData attributes name every element, so the readers
 * never have to browse.
 */
export function httpSource(base, name) {
    const url = (path) => `${base.replace(/\/+$/, '')}/${clean(path)}`;
    const read = async (path, headers) => {
        const r = await fetch(url(path), { headers });
        if (!r.ok)
            return null;
        return { status: r.status, bytes: new Uint8Array(await r.arrayBuffer()) };
    };
    return {
        name,
        async bytes(path) {
            return (await read(path))?.bytes ?? null;
        },
        async range(path, start, end) {
            const got = await read(path, { Range: `bytes=${start}-${end - 1}` });
            if (!got)
                return null;
            // a server that does not do ranges answers 200 with the whole file; cut it here rather
            // than hand a parquet reader bytes from the wrong offset
            return got.status === 206 ? got.bytes : got.bytes.subarray(start, end);
        },
        async size(path) {
            const r = await fetch(url(path), { method: 'HEAD' });
            const n = r.ok ? Number(r.headers.get('content-length')) : NaN;
            return Number.isFinite(n) ? n : null;
        },
        async list() {
            return [];
        }
    };
}
