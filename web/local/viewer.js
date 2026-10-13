/**
 * The view of a store the visitor opened themselves.
 *
 * The viewer's own datasets are cut up by the server first — tiles, smoothed surfaces, gene
 * indexes — and drawn in WebGL from those.  A folder on the visitor's disk has none of that and
 * cannot get it: nothing is uploaded, so nothing can be prepared.  This draws it directly instead,
 * reading only the chunks the current view needs, on a 2D canvas of its own: the image, the
 * cell-type map, the outlines and the transcripts, each through its own transform into the store's
 * `global` frame, which is what makes them line up.
 *
 * It is mounted over the WebGL canvas and taken down again when the visitor goes back to the
 * catalog, so the two never run at once.
 */
import { openStore, apply, compose } from './sdata.js';
import {
	labelColour, labelCounts, loadImageWindow, loadPoints, loadShapes, paintLabels, pickLevel
} from './elements.js';

const el = (tag, props = {}, kids = []) => {
	const node = Object.assign(document.createElement(tag), props);
	for (const k of [].concat(kids)) node.append(k);
	return node;
};
const PALETTE = ['#2f6fed', '#e3663e', '#2aa37a', '#b14fd8', '#d8a62a', '#4ab8d8'];
const kindLabel = (e) =>
	e.kind === 'image' ? 'image' : e.kind === 'labels' ? 'labels' : e.kind === 'shapes' ? 'shapes' : 'points';
const hex = (rgb) => '#' + rgb.map((c) => c.toString(16).padStart(2, '0')).join('');

/**
 * Read a store and draw it.  `canvas` is drawn on, `panel` is filled with its layers, `status`
 * reports progress the way the rest of the viewer does.  Returns a handle whose `close()` puts
 * everything back.
 */
export async function openLocal({ source, canvas, panel, status = () => {} }) {
	status(`reading ${source.name}…`);
	const store = await openStore(source);

	const ctx = canvas.getContext('2d');
	const view = { x: 0, y: 0, scale: 1 };
	const size = { w: canvas.clientWidth || 800, h: canvas.clientHeight || 600 };
	const edits = new Map();                        // cell type -> colour, for this visit only
	const names = store.palette?.names ?? [];
	const typeName = (v) => names[v - 1] ?? `type ${v}`;
	const colourOf = (v) => edits.get(typeName(v)) ?? store.palette?.colours[v - 1];

	const layers = store.elements.map((e, i) => ({
		el: e,
		// one image and one set of outlines open: enough to see where you are, not enough to wait for
		visible: e.kind === 'image'
			? store.elements.findIndex((q) => q.kind === 'image') === i
			: e.kind === 'labels' && !store.elements.some((q) => q.kind === 'image'),
		opacity: 1, channel: -1, z: Math.floor((e.depth ?? 1) / 2),
		colour: PALETTE[i % PALETTE.length], tile: null, shapes: null, points: null,
		types: [], hidden: new Set(), genes: null, busy: false, error: null
	}));

	/* ------------------------------------------------------------ geometry */
	const screenMatrix = () => [
		view.scale, 0, size.w / 2 - view.x * view.scale,
		0, view.scale, size.h / 2 - view.y * view.scale
	];
	const windowOf = () => [
		view.x - size.w / 2 / view.scale, view.y - size.h / 2 / view.scale,
		view.x + size.w / 2 / view.scale, view.y + size.h / 2 / view.scale
	];
	const toGlobal = (sx, sy) => [
		view.x + (sx - size.w / 2) / view.scale,
		view.y + (sy - size.h / 2) / view.scale
	];

	function loadedBounds() {
		let b = null;
		const grow = (q) => (b = b ? [Math.min(b[0], q[0]), Math.min(b[1], q[1]), Math.max(b[2], q[2]), Math.max(b[3], q[3])] : q);
		for (const l of layers) {
			if (l.shapes?.count) grow(l.shapes.bounds);
			if (l.points?.xy.length) grow(l.points.bounds);
		}
		return b;
	}

	function fit() {
		const b = store.bounds ?? loadedBounds() ?? [0, 0, 100, 100];
		const w = Math.max(1e-6, b[2] - b[0]), h = Math.max(1e-6, b[3] - b[1]);
		view.x = (b[0] + b[2]) / 2;
		view.y = (b[1] + b[3]) / 2;
		view.scale = Math.min(size.w / w, size.h / h) * 0.94;
	}

	/* ------------------------------------------------------------- reading */
	let token = 0;
	async function loadImages() {
		const mine = ++token;
		for (const layer of layers) {
			if (!layer.visible || (layer.el.kind !== 'image' && layer.el.kind !== 'labels')) continue;
			layer.busy = true; refresh();
			try {
				const tile = await loadImageWindow(source, layer.el, pickLevel(layer.el, 1 / view.scale), windowOf(), {
					channel: layer.el.rgb ? layer.channel : Math.max(0, layer.channel),
					z: layer.z, colours: colourOf, hidden: layer.hidden
				});
				if (mine !== token) return;
				if (tile) {
					layer.tile?.bitmap?.close?.();
					layer.tile = { bitmap: await createImageBitmap(tile.data), matrix: tile.matrix, labels: tile.labels };
					if (tile.labels) {
						const counts = labelCounts(tile.labels.values, tile.labels.width * tile.labels.height);
						layer.types = [...counts]
							.map(([value, pixels]) => ({ value, name: typeName(value), pixels }))
							.sort((a, b) => b.pixels - a.pixels);
					}
				}
				layer.error = null;
			} catch (e) {
				layer.error = e instanceof Error ? e.message : String(e);
			} finally {
				layer.busy = false;
			}
			draw(); refresh();
		}
	}

	async function loadTable(layer) {
		if (layer.busy || layer.shapes || layer.points) return;
		layer.busy = true; refresh();
		try {
			if (layer.el.kind === 'shapes') layer.shapes = await loadShapes(source, layer.el);
			else layer.points = await loadPoints(source, layer.el);
			if (!store.bounds) { fit(); loadImages(); }
		} catch (e) {
			layer.error = e instanceof Error ? e.message : String(e);
		} finally {
			layer.busy = false; draw(); refresh();
		}
	}

	async function repaintLabels(layer) {
		const kept = layer.tile?.labels;
		if (!kept) return;
		const data = paintLabels(kept.values, kept.width, kept.height, colourOf, layer.hidden);
		layer.tile.bitmap?.close?.();
		layer.tile = { ...layer.tile, bitmap: await createImageBitmap(data) };
		draw();
	}

	let settle = null;
	const scheduleImages = () => {
		clearTimeout(settle);
		settle = setTimeout(loadImages, 160);
	};

	/* ------------------------------------------------------------- drawing */
	let frame = 0;
	const draw = () => {
		if (frame) return;
		frame = requestAnimationFrame(() => { frame = 0; paint(); });
	};

	function paint() {
		const dpr = Math.min(2, window.devicePixelRatio || 1);
		if (canvas.width !== Math.round(size.w * dpr) || canvas.height !== Math.round(size.h * dpr)) {
			canvas.width = Math.round(size.w * dpr);
			canvas.height = Math.round(size.h * dpr);
		}
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.fillStyle = '#05070a';
		ctx.fillRect(0, 0, size.w, size.h);
		const S = screenMatrix();
		for (const layer of layers) {
			if (!layer.visible) continue;
			ctx.globalAlpha = layer.opacity;
			if (layer.tile) {
				const m = compose(S, layer.tile.matrix);
				ctx.setTransform(m[0] * dpr, m[3] * dpr, m[1] * dpr, m[4] * dpr, m[2] * dpr, m[5] * dpr);
				ctx.imageSmoothingEnabled = view.scale < 4;
				ctx.drawImage(layer.tile.bitmap, 0, 0);
				ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			}
			if (layer.shapes) {
				ctx.strokeStyle = layer.colour;
				ctx.lineWidth = Math.max(0.6, Math.min(2, view.scale * 0.3));
				ctx.beginPath();
				for (const ring of layer.shapes.rings) {
					if (ring.length < 4) continue;
					let [px, py] = apply(S, ring[0], ring[1]);
					ctx.moveTo(px, py);
					for (let i = 2; i < ring.length; i += 2) {
						[px, py] = apply(S, ring[i], ring[i + 1]);
						ctx.lineTo(px, py);
					}
					ctx.closePath();
				}
				ctx.stroke();
				for (const c of layer.shapes.circles) {
					const [cx, cy] = apply(S, c.x, c.y);
					ctx.beginPath();
					ctx.arc(cx, cy, Math.max(1, c.r * view.scale), 0, Math.PI * 2);
					ctx.stroke();
				}
			}
			if (layer.points) {
				const p = layer.points;
				const r = Math.max(0.7, Math.min(3, view.scale * 0.5));
				const one = !p.gene || !p.genes.length;
				if (one) ctx.fillStyle = layer.colour;
				for (let i = 0, n = p.xy.length / 2; i < n; i++) {
					const g = p.gene ? p.gene[i] : 0;
					if (layer.genes && !layer.genes.has(g)) continue;
					const [sx, sy] = apply(S, p.xy[i * 2], p.xy[i * 2 + 1]);
					if (sx < -4 || sy < -4 || sx > size.w + 4 || sy > size.h + 4) continue;
					if (!one) ctx.fillStyle = `hsl(${(g * 137.508) % 360} 85% 62%)`;
					ctx.fillRect(sx - r / 2, sy - r / 2, r, r);
				}
			}
		}
		ctx.globalAlpha = 1;
		scaleBar();
	}

	function scaleBar() {
		const target = 120 / view.scale;
		const pow = Math.pow(10, Math.floor(Math.log10(target)));
		const nice = [1, 2, 5, 10].map((q) => q * pow)
			.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
		const px = nice * view.scale;
		ctx.fillStyle = 'rgba(0,0,0,.55)';
		ctx.fillRect(14, size.h - 46, px + 26, 32);
		ctx.fillStyle = '#fff';
		ctx.fillRect(26, size.h - 24, px, 3);
		ctx.font = '11px system-ui, sans-serif';
		ctx.fillText(`${nice} units`, 26, size.h - 31);
	}

	/* --------------------------------------------------------- interaction */
	const onWheel = (e) => {
		e.preventDefault();
		const r = canvas.getBoundingClientRect();
		const before = toGlobal(e.clientX - r.left, e.clientY - r.top);
		view.scale *= Math.exp(-e.deltaY * 0.0015);
		const after = toGlobal(e.clientX - r.left, e.clientY - r.top);
		view.x += before[0] - after[0];
		view.y += before[1] - after[1];
		draw(); scheduleImages();
	};
	let drag = null;
	const onDown = (e) => { canvas.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY }; };
	const onMove = (e) => {
		if (!drag) return;
		view.x -= (e.clientX - drag.x) / view.scale;
		view.y -= (e.clientY - drag.y) / view.scale;
		drag = { x: e.clientX, y: e.clientY };
		draw();
	};
	const onUp = () => { if (drag) scheduleImages(); drag = null; };
	canvas.addEventListener('wheel', onWheel, { passive: false });
	canvas.addEventListener('pointerdown', onDown);
	canvas.addEventListener('pointermove', onMove);
	canvas.addEventListener('pointerup', onUp);
	canvas.addEventListener('pointerleave', onUp);

	const onResize = () => {
		size.w = Math.max(200, canvas.clientWidth);
		size.h = Math.max(200, canvas.clientHeight);
		draw();
	};
	window.addEventListener('resize', onResize);

	/* --------------------------------------------------------------- panel */
	function refresh() {
		panel.textContent = '';
		panel.append(el('h2', { textContent: 'Your data' }));
		const head = el('div', { className: 'localHead' });
		head.append(el('strong', { textContent: source.name }));
		head.append(el('span', {
			textContent: `${layers.length} element${layers.length === 1 ? '' : 's'}` +
				(layers.some((l) => l.busy) ? ' · reading…' : '')
		}));
		panel.append(head);

		for (const layer of layers) {
			const section = el('div', { className: 'localLayer' });
			const row = el('label', { className: 'localRow' });
			const box = el('input', { type: 'checkbox', checked: layer.visible });
			box.onchange = () => {
				layer.visible = box.checked;
				if (layer.visible) {
					if (layer.el.kind === 'shapes' || layer.el.kind === 'points') loadTable(layer);
					else scheduleImages();
				}
				draw(); refresh();
			};
			row.append(box, el('span', { className: 'localName', textContent: layer.el.name, title: layer.el.name }),
				el('small', { textContent: kindLabel(layer.el) }));
			section.append(row);

			if (layer.visible) {
				const body = el('div', { className: 'localBody' });
				if (layer.error) body.append(el('p', { className: 'localErr', textContent: layer.error }));
				body.append(slider('opacity', layer.opacity, 0.1, 1, 0.05, (v) => { layer.opacity = v; draw(); }));
				if (layer.el.kind === 'image' || layer.el.kind === 'labels') {
					const e = layer.el;
					if (e.channels > 1) {
						const sel = el('select');
						if (e.rgb) sel.append(el('option', { value: '-1', textContent: 'RGB' }));
						for (let c = 0; c < e.channels; c++) sel.append(el('option', { value: String(c), textContent: `channel ${c}` }));
						sel.value = String(layer.channel);
						sel.onchange = () => { layer.channel = +sel.value; scheduleImages(); };
						body.append(labelled('channel', sel));
					}
					if (e.depth > 1)
						body.append(slider(`z ${layer.z}`, layer.z, 0, e.depth - 1, 1, (v) => { layer.z = v; scheduleImages(); refresh(); }));
					body.append(el('p', {
						className: 'localMeta',
						textContent: `${e.levels[0].width} × ${e.levels[0].height}` +
							(e.depth > 1 ? ` × ${e.depth}` : '') + ` · ${e.levels.length} level${e.levels.length === 1 ? '' : 's'}`
					}));
					if (e.kind === 'labels' && layer.types.length) body.append(typeList(layer));
				} else if (layer.el.kind === 'shapes') {
					body.append(el('p', {
						className: 'localMeta',
						textContent: layer.shapes
							? `${layer.shapes.count.toLocaleString()} of ${layer.shapes.total.toLocaleString()} shapes`
							: 'reading…'
					}));
					body.append(colourInput(layer.colour, (v) => { layer.colour = v; draw(); }));
				} else {
					body.append(el('p', {
						className: 'localMeta',
						textContent: layer.points
							? `${(layer.points.xy.length / 2).toLocaleString()} of ${layer.points.total.toLocaleString()} points` +
								(layer.points.genes.length ? ` · ${layer.points.genes.length} genes` : '')
							: 'reading…'
					}));
					if (layer.points?.genes.length) body.append(geneList(layer));
				}
				section.append(body);
			}
			panel.append(section);
		}
	}

	const labelled = (text, control) => {
		const row = el('label', { className: 'localCtl' });
		row.append(el('span', { textContent: text }), control);
		return row;
	};

	function slider(text, value, min, max, step, onInput) {
		const input = el('input', { type: 'range', min, max, step, value });
		input.oninput = () => onInput(+input.value);
		return labelled(text, input);
	}

	function colourInput(value, onInput) {
		const input = el('input', { type: 'color', value, className: 'localSwatch' });
		input.oninput = () => onInput(input.value);
		return labelled('colour', input);
	}

	function typeList(layer) {
		const wrap = el('div');
		const head = el('div', { className: 'localTypesHead' });
		head.append(el('span', { textContent: `${layer.types.length} cell type${layer.types.length === 1 ? '' : 's'} in view` }));
		const reset = el('button', { textContent: 'Reset colours', disabled: edits.size === 0 });
		reset.onclick = () => { edits.clear(); for (const l of layers) repaintLabels(l); refresh(); };
		head.append(reset);
		wrap.append(head);
		const total = layer.types.reduce((p, q) => p + q.pixels, 0);
		const list = el('div', { className: 'localTypes' });
		for (const t of layer.types) {
			const row = el('label', { className: 'localType' });
			const on = el('input', { type: 'checkbox', checked: !layer.hidden.has(t.value) });
			on.onchange = () => {
				if (on.checked) layer.hidden.delete(t.value); else layer.hidden.add(t.value);
				repaintLabels(layer);
			};
			const swatch = el('input', {
				type: 'color', className: 'localSwatch',
				value: colourOf(t.value) ?? hex(labelColour(t.value))
			});
			swatch.oninput = () => {
				edits.set(t.name, swatch.value);
				for (const l of layers) if (l.el.kind === 'labels') repaintLabels(l);
				reset.disabled = false;
			};
			row.append(on, swatch, el('span', { textContent: t.name, title: t.name }),
				el('small', { textContent: `${((t.pixels / total) * 100).toFixed(1)}%` }));
			list.append(row);
		}
		wrap.append(list);
		wrap.append(el('p', {
			className: 'localMeta',
			textContent: store.palette
				? `colours from tables/${store.palette.table}`
				: 'this store names no cell types — these colours are the viewer’s own'
		}));
		return wrap;
	}

	function geneList(layer) {
		const wrap = el('div');
		const find = el('input', { type: 'search', placeholder: 'Find a gene', className: 'localFind' });
		const list = el('div', { className: 'localTypes' });
		const p = layer.points;
		const render = () => {
			list.textContent = '';
			const q = find.value.trim().toLowerCase();
			const rows = p.genes.map((name, i) => ({ name, i, n: p.counts[i] }))
				.filter((g) => !q || g.name.toLowerCase().includes(q))
				.sort((a, b) => b.n - a.n).slice(0, 300);
			for (const g of rows) {
				const row = el('label', { className: 'localType' });
				const on = el('input', { type: 'checkbox', checked: !layer.genes || layer.genes.has(g.i) });
				on.onchange = () => {
					const set = layer.genes ? new Set(layer.genes) : new Set(p.genes.map((_, i) => i));
					if (on.checked) set.add(g.i); else set.delete(g.i);
					layer.genes = set.size === p.genes.length ? null : set;
					draw();
				};
				const dot = el('i');
				dot.style.background = `hsl(${(g.i * 137.508) % 360} 85% 62%)`;
				row.append(on, dot, el('span', { textContent: g.name, title: g.name }),
					el('small', { textContent: g.n.toLocaleString() }));
				list.append(row);
			}
		};
		find.oninput = render;
		const buttons = el('div', { className: 'localTypesHead' });
		const all = el('button', { textContent: 'All' });
		all.onclick = () => { layer.genes = null; draw(); render(); };
		const none = el('button', { textContent: 'None' });
		none.onclick = () => { layer.genes = new Set(); draw(); render(); };
		buttons.append(all, none);
		wrap.append(find, buttons, list);
		render();
		return wrap;
	}

	/* ---------------------------------------------------------------- open */
	onResize();
	fit();
	refresh();
	for (const l of layers) if (l.visible && (l.el.kind === 'shapes' || l.el.kind === 'points')) loadTable(l);
	await loadImages();
	draw();
	status(`${source.name}: ${store.elements.length} element(s)` +
		(store.palette ? `, ${store.palette.names.length} cell types` : ''));

	return {
		store,
		fit: () => { fit(); draw(); scheduleImages(); },
		zoom: (by) => { view.scale *= by; draw(); scheduleImages(); },
		close() {
			clearTimeout(settle);
			token++;
			window.removeEventListener('resize', onResize);
			canvas.removeEventListener('wheel', onWheel);
			canvas.removeEventListener('pointerdown', onDown);
			canvas.removeEventListener('pointermove', onMove);
			canvas.removeEventListener('pointerup', onUp);
			canvas.removeEventListener('pointerleave', onUp);
			for (const l of layers) l.tile?.bitmap?.close?.();
			panel.textContent = '';
			ctx.setTransform(1, 0, 0, 1, 0, 0);
			ctx.clearRect(0, 0, canvas.width, canvas.height);
		}
	};
}
