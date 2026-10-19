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
 *   - a .zarr folder dropped anywhere on the window is opened the same way;
 *   - until a store is open, the panel holds only its title and the Open block; the rest of it, and the
 *     controls over the view, appear once a store is open;
 *   - a store's sections come as their loading ends (Show, the layers and the view with the first layer on
 *     screen, each layer's block with its layer, Gene expression with the gene list), and a ring turns beside
 *     the title while anything is still loading.
 * The page's functions are classic-script globals, so they are called and replaced by name.
 */
'use strict';
(() => {
	if (!window.__viewer) return;                // no WebGL2: the page's own fallback is in charge
	const app = window.ssaft || null;            // the Electron bridge (preload.js); absent in a browser
	const name = (p) => String(p).replace(/[\\/]+$/, '').split(/[\\/]/).pop();
	const hello = () => setStatus('');

	async function current() {
		const r = await fetch('/api/datasets');
		return r.ok ? (await r.json()).datasets[0] || null : null;
	}
	async function show(d) {
		$('#dsSelect').innerHTML = `<option value="${esc(d.id)}">${esc(d.title)}</option>`;
		document.title = `${d.title} — SSAF-T Viewer`;
		const run = stage.start();
		empty(false);
		await selectDataset(d);
		if (run === stage.run) stage.base = true;     // the layers it opens with are on screen
	}

	// the page's boot calls this once the server has answered /api/config
	openCatalog = async () => {
		const d = await current();
		if (d) await show(d); else { clearDataset(); hello(); empty(true); document.title = 'SSAF-T Viewer'; }
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
	brand.innerHTML = '<div class="desk-word">SSAF-T <span>Viewer</span><i id="deskSpin" hidden title="Loading"></i></div>';
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
		<p class="hint">Or drag and drop a .zarr folder onto the window. The first opening takes longer.</p>
		<div class="desk-links" ${app ? '' : 'hidden'}><a href="#" id="clearCache">Clear cache</a></div>
		<button id="openLocal" hidden></button><p id="localHint" hidden></p>
		<input type="file" id="localFiles" hidden>`;   // the page's boot still binds these three
	$('#catalogSection').prepend(box);
	$('#openZarr').onclick = async () => openSafely(app ? await app.openDialog() : prompt('Path of the .zarr folder'));
	if (app) {
		$('#clearCache').onclick = (e) => { e.preventDefault(); app.clearCache(); };
	}
	// the site's in-browser reader is not used here, but its button is still bound by the page's boot
	window.openLocalStore = () => $('#openZarr').click();

	/* ---------------------------------------------------------------- before a store is open */
	// the panel keeps its place, with only the title and the Open block in it; Show, the layers, the genes and
	// the view controls (and the toolbar, axes and help over the view) come with a store.  The status line moves
	// from the bottom of the view controls to under the Open button, so "reading …" and errors show either way.
	$('#localOpen').append($('#status'));
	function empty(on) {
		document.body.classList.toggle('desk-empty', on);
	}
	empty(true);

	/* ---------------------------------------------------------------- loading, section by section */
	// A store's sections are hidden while it loads (body.desk-staging) and come one by one: st-layers once the
	// first layer is on screen (Show, Layers, View, the toolbar), st-genes once the gene list is in.  A layer's
	// own block appears with its layer (the page renders only loaded layers), its Voxels / Solid buttons when
	// its surfaces are ready (store.html).  The ring by the title turns while anything is loading: a layer being
	// fetched, surfaces being made, the gene list; so it also returns when a layer is ticked later.
	const stage = {
		run: 0, base: false, genes: false,
		start() {
			this.run++; this.base = false; this.genes = false;
			document.body.classList.add('desk-staging');
			document.body.classList.remove('st-layers', 'st-genes');
			return this.run;
		}
	};
	const loadGenesPage = loadGenes;
	loadGenes = async (ds) => {
		const run = stage.run;
		try { return await loadGenesPage(ds); }
		finally { if (run === stage.run) stage.genes = true; }
	};
	const spin = $('#deskSpin');
	setInterval(() => {
		const S = window.__viewer, body = document.body;
		if (body.classList.contains('desk-empty')) { spin.hidden = true; return; }
		if (S.layers.length) body.classList.add('st-layers');
		if (stage.genes) body.classList.add('st-genes');
		const busy = !stage.base || !stage.genes || pending.size > 0 || S.layers.some((L) => L.solidWait);
		spin.hidden = !busy;
		if (!busy) body.classList.remove('desk-staging');
	}, 300);

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
