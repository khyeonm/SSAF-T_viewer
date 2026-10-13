/**
 * Getting hold of the visitor's folder, whichever way their browser offers.
 *
 * Chrome and Edge have a directory picker that hands over a HANDLE: a file is read when it is
 * asked for, and a store of any size opens at once.  Firefox and Safari have no such thing, but
 * they do have the folder input every browser has had for years: it lists the folder's files up
 * front and hands over a File for each — still a reference, not a copy, so the bytes stay on the
 * machine either way; only the opening is slower, in proportion to how many chunks the store has.
 *
 * Both are here, in one module, because the page needs them twice: once in the viewer proper, and
 * once in the reduced page a browser without WebGL2 gets, where opening a store is all there is.
 */
import { directorySource, fileListSource, httpSource } from './source.js';

export const hasPicker = () => typeof window !== 'undefined' && 'showDirectoryPicker' in window;

/**
 * Ask for a folder and return a source for it, or null when the visitor cancelled.
 * `input` is the page's `<input type="file" webkitdirectory>`; `force` takes that path even where
 * the picker exists, which is how the fallback is exercised.
 */
export async function pickFolderSource({ input, force = false, status = () => {} } = {}) {
	if (hasPicker() && !force) {
		const handle = await window.showDirectoryPicker({ id: 'ssaft-store', mode: 'read' }).catch((e) => {
			if (e && e.name === 'AbortError') return null;
			throw e;
		});
		return handle ? directorySource(handle, handle.name) : null;
	}
	if (!input) throw new Error('this browser cannot open a folder');
	const files = await new Promise((done) => {
		input.addEventListener('change', () => done(input.files), { once: true });
		input.value = '';           // so that picking the same folder twice still fires
		input.click();
	});
	if (!files || !files.length) return null;
	status(`reading ${files.length.toLocaleString()} files…`);
	return fileListSource(files);
}

/** a store served over HTTP, for `?local=<url>` */
export const sourceForUrl = (url) => httpSource(url, url.split('/').filter(Boolean).pop() || 'store');
