/**
 * The desktop app's part of the viewer page.
 *
 * store.html is the site's viewer page, unchanged; the app's server adds this script after it.  The page
 * runs in catalog mode, as on the site, with a catalog of one: the SSAF-T store (.zarr) the user opened.
 * The server's /api/open reads it and makes it that one dataset; from then on its Show boxes, the solid
 * view and the genes are the page's own code.  What this script changes:
 *   - the dataset section (its title, the dataset list, the dataset's facts) is hidden: there is one store,
 *     named in the window title;
 *   - "Your own data" (the in-browser reader, a 2D canvas the site falls back on for a visitor's folder)
 *     becomes an Open button that sends the store to the server, so it gets the full viewer;
 *   - a .zarr folder dropped anywhere on the window is opened the same way.
 * The page's functions are classic-script globals, so they are called and replaced by name.
 */
'use strict';
(() => {
	if (!window.__viewer) return;                // no WebGL2: the page's own fallback is in charge
	const app = window.ssaft || null;            // the Electron bridge (preload.js); absent in a browser
	const name = (p) => String(p).replace(/[\\/]+$/, '').split(/[\\/]/).pop();
	const hello = () => setStatus('Open an SSAF-T store (.zarr folder): File > Open, the button above, or drop it on the window');

	async function current() {
		const r = await fetch('/api/datasets');
		return r.ok ? (await r.json()).datasets[0] || null : null;
	}
	async function show(d) {
		$('#dsSelect').innerHTML = `<option value="${esc(d.id)}">${esc(d.title)}</option>`;
		document.title = `${d.title} — SSAF-T Viewer`;
		await selectDataset(d);
	}

	// the page's boot calls this once the server has answered /api/config
	openCatalog = async () => {
		const d = await current();
		if (d) await show(d); else { clearDataset(); hello(); }
	};

	async function open(path) {
		if (!path) return;
		setStatus(`reading ${name(path)} …`);
		const r = await fetch('/api/open?path=' + encodeURIComponent(path));
		const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
		if (!r.ok) throw new Error(`${name(path)}: ${j.error || r.status}`);
		if (app) app.opened(path);
		await show(await current());
	}
	const openSafely = (p) => open(p).catch((e) => setStatus(e.message, true));

	/* ---------------------------------------------------------------- the title */
	// the site header's wordmark.  The page's own h1 is left in place, hidden: its boot still writes the
	// catalog title into it.
	const brand = document.createElement('div');
	brand.className = 'desk-brand';
	brand.innerHTML = '<div class="desk-word">SSAF-T <span>Viewer</span></div>';
	$('#panel h1').hidden = true;
	brand.append($('#sub'));
	$('#panel .head').prepend(brand);

	/* ---------------------------------------------------------------- the panel */
	// the dataset section: its heading, the list and the facts go; Show and the layers stay
	for (const n of [$('#catalogSection h2'), $('#dsSelect'), $('#dsInfo')]) n.hidden = true;
	const box = $('#localOpen');
	box.innerHTML = `
		<div class="row">
			<button id="openZarr" title="An SSAF-T store: the ssaft_sdata.zarr folder">Open .zarr folder…</button>
		</div>
		<p class="hint">Or drop the folder on the window. The first opening takes longer.</p>
		<button id="openLocal" hidden></button><p id="localHint" hidden></p>
		<input type="file" id="localFiles" hidden>`;   // the page's boot still binds these three
	$('#catalogSection').prepend(box);
	$('#openZarr').onclick = async () => openSafely(app ? await app.openDialog() : prompt('Path of the .zarr folder'));
	// the site's in-browser reader is not used here, but its button is still bound by the page's boot
	window.openLocalStore = () => $('#openZarr').click();

	/* ---------------------------------------------------------------- drag and drop */
	const veil = Object.assign(document.createElement('div'), { id: 'deskDrop', hidden: true });
	veil.innerHTML = '<div>Drop an SSAF-T store (.zarr folder) to open it</div>';
	document.body.append(veil);
	let depth = 0;
	const isFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
	addEventListener('dragenter', (e) => { if (!isFiles(e)) return; e.preventDefault(); depth++; veil.hidden = false; });
	addEventListener('dragover', (e) => { if (!isFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
	addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; veil.hidden = true; } });
	addEventListener('drop', (e) => {
		if (!isFiles(e)) return;
		e.preventDefault();
		depth = 0;
		veil.hidden = true;
		const f = e.dataTransfer.files[0];
		const p = f && app ? app.pathForFile(f) : '';
		if (!p) { setStatus('Could not read the path of what was dropped; use the Open button instead', true); return; }
		openSafely(p);
	});

	if (app) app.onOpenPath(openSafely);
})();
