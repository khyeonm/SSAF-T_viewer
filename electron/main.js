/**
 * SSAF-T Viewer: the site's cell-type map viewer, as a desktop app over the user's own SSAF-T store (.zarr).
 *
 * The app is three pieces, two of them the site's own:
 *   server   server/desktop.py over server/app.py (the site's viewer server, unchanged): it reads a store,
 *            cuts its tiles, solid surfaces and gene index into the cache, and serves them on 127.0.0.1.
 *            Packaged, it is a PyInstaller build (resources/server/ssaft-server); in development it is
 *            run with Python (SSAFT_PYTHON, else `python3` / `python`).
 *   page     web/store.html (the site's viewer page, unchanged) plus web/desktop.js, which turns its
 *            dataset section into the one store the user opened and adds the Open button and drag and drop.
 *   this     the window, the menus and file dialogs, and the server's lifetime.
 *
 * The server only answers requests that carry a secret made here at start-up (the page sends it with
 * every request), so another program on the machine cannot use it to read files.
 */
'use strict';
const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require('electron');
const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const IS_WIN = process.platform === 'win32';
const IS_MAC = process.platform === 'darwin';

// The viewer is WebGL2 throughout; a GPU the blocklist would refuse still draws it.
app.commandLine.appendSwitch('ignore-gpu-blocklist');

let win = null;
let server = null;          // { proc, port, token, url }
let quitting = false;
const pendingOpen = [];     // paths asked for before the page was ready

const dirs = () => {
	const ud = app.getPath('userData');
	return { userData: ud, cache: path.join(ud, 'cache'), logs: path.join(ud, 'logs') };
};

/* ------------------------------------------------------------------ the server */
function serverCommand() {
	if (app.isPackaged) {
		const exe = path.join(process.resourcesPath, 'server', IS_WIN ? 'ssaft-server.exe' : 'ssaft-server');
		return { cmd: exe, args: ['--web', path.join(process.resourcesPath, 'web')] };
	}
	// the activated conda / micromamba environment (environment.yml) before whatever `python3` is on PATH
	const env = process.env.CONDA_PREFIX && path.join(process.env.CONDA_PREFIX, IS_WIN ? 'python.exe' : 'bin/python');
	const py = process.env.SSAFT_PYTHON || (env && fs.existsSync(env) ? env : (IS_WIN ? 'python' : 'python3'));
	return { cmd: py, args: [path.join(ROOT, 'server', 'desktop.py'), '--web', path.join(ROOT, 'web')] };
}

function startServer() {
	return new Promise((resolve, reject) => {
		const d = dirs();
		for (const p of [d.cache, d.logs]) fs.mkdirSync(p, { recursive: true });
		const token = crypto.randomBytes(24).toString('hex');
		const tokenFile = path.join(d.userData, 'server.token');
		fs.writeFileSync(tokenFile, token, { mode: 0o600 });
		const { cmd, args } = serverCommand();
		const log = fs.createWriteStream(path.join(d.logs, 'server.log'), { flags: 'a' });
		log.write(`\n==== ${new Date().toISOString()} ${cmd} ${args.join(' ')}\n`);
		const proc = spawn(cmd, [...args, '--port', '0', '--cache', d.cache, '--token-file', tokenFile, '--watch-stdin'], {
			stdio: ['pipe', 'pipe', 'pipe'],
			windowsHide: true,
			env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
		});
		let ready = false;
		let tail = '';
		const timer = setTimeout(() => {
			if (!ready) reject(new Error('the viewer server did not start within 3 minutes\n\n' + tail.slice(-2000)));
		}, 180000);
		const onData = (buf) => {
			const s = buf.toString();
			log.write(s);
			tail = (tail + s).slice(-8000);
			const m = !ready && /SSAFT_SERVER_READY (\d+)/.exec(tail);
			if (m) {
				ready = true;
				clearTimeout(timer);
				const port = Number(m[1]);
				server = { proc, port, token, url: `http://127.0.0.1:${port}` };
				resolve(server);
			}
		};
		proc.stdout.on('data', onData);
		proc.stderr.on('data', onData);
		proc.on('error', (e) => {
			clearTimeout(timer);
			reject(new Error(`could not start the viewer server (${cmd}): ${e.message}`));
		});
		proc.on('exit', (code, sig) => {
			log.write(`==== server exited: code ${code} signal ${sig}\n`);
			log.end();
			if (!ready) {
				clearTimeout(timer);
				reject(new Error(`the viewer server stopped while starting (exit ${code})\n\n` + tail.slice(-2000)));
				return;
			}
			if (server && server.proc === proc) server = null;
			if (!quitting && !proc.expected) serverDied(code, tail);
		});
	});
}

function stopServer() {
	if (!server) return Promise.resolve();
	const { proc } = server;
	server = null;
	proc.expected = true;
	return new Promise((done) => {
		proc.once('exit', done);
		try { proc.stdin.end(); } catch { /* already closed */ }   // the server exits when stdin closes
		setTimeout(() => { try { proc.kill(); } catch { /* gone */ } }, 1500);
		setTimeout(done, 4000);
	});
}

async function serverDied(code, tail) {
	const r = await dialog.showMessageBox(win, {
		type: 'error',
		message: 'The viewer server stopped.',
		detail: `Exit code ${code}. This is most often memory running out on a large store.\n\n${tail.slice(-1500)}`,
		buttons: ['Restart', 'Show log', 'Quit'],
		defaultId: 0
	});
	if (r.response === 1) shell.showItemInFolder(path.join(dirs().logs, 'server.log'));
	if (r.response === 2) return app.quit();
	await restart();
}

async function restart() {
	await stopServer();
	try {
		await startServer();
		loadPage();
	} catch (e) {
		fatal(e);
	}
}

function fatal(e) {
	dialog.showErrorBox('SSAF-T Viewer could not start', String(e && e.message ? e.message : e) +
		`\n\nLog: ${path.join(dirs().logs, 'server.log')}`);
	app.quit();
}

/* ------------------------------------------------------------------ the window */
function loadPage() {
	if (!win || !server) return;
	win.loadURL(`${server.url}/store.html?t=${server.token}`);
}

function createWindow() {
	win = new BrowserWindow({
		width: 1440,
		height: 900,
		minWidth: 900,
		minHeight: 600,
		backgroundColor: '#0e1014',
		title: 'SSAF-T Viewer',
		icon: path.join(ROOT, 'build-resources', 'icon.png'),   // the window's own icon (Linux, Windows taskbar)
		show: false,
		webPreferences: {
			preload: path.join(__dirname, 'preload.js'),
			contextIsolation: true,
			sandbox: true,
			nodeIntegration: false
		}
	});
	win.once('ready-to-show', () => win.show());
	if (!IS_MAC) win.removeMenu();                       // no menu bar: everything is in the panel
	win.webContents.on('before-input-event', (e, input) => shortcut(e, input));
	win.on('closed', () => { win = null; });
	// the page is the viewer and nothing else: links out open in the browser, nothing navigates away
	win.webContents.setWindowOpenHandler(({ url }) => {
		if (/^https?:/.test(url)) shell.openExternal(url);
		return { action: 'deny' };
	});
	win.webContents.on('will-navigate', (e, url) => {
		if (!server || !url.startsWith(server.url)) e.preventDefault();
	});
	win.webContents.on('did-finish-load', () => setTimeout(flushOpen, 300));
	// the server takes a few seconds to start (longer on a first launch, while the OS scans its files):
	// the window says so instead of not being there
	if (server) loadPage(); else win.loadFile(path.join(__dirname, 'starting.html'));
}

/** the keys the menu used to carry */
function shortcut(e, input) {
	if (input.type !== 'keyDown') return;
	const mod = IS_MAC ? input.meta : input.control;
	const key = input.key.toLowerCase();
	if (mod && !input.shift && key === 'o') { e.preventDefault(); pick().then(openPath); }
	else if ((mod && input.shift && key === 'i') || input.key === 'F12') { e.preventDefault(); win.webContents.toggleDevTools(); }
	else if ((mod && key === 'r') || input.key === 'F5') { e.preventDefault(); win.webContents.reload(); }
	else if (mod && input.shift && key === 'p') { e.preventDefault(); win.webContents.send('toggle-log'); }   // the log: not in the UI
}

/* ------------------------------------------------------------------ opening stores */
function openPath(p) {
	if (!p) return;
	pendingOpen.push(p);
	flushOpen();
}
function flushOpen() {
	// only the viewer page listens for stores to open; the starting screen keeps them until it is there
	if (!win || !server || win.webContents.isLoading() || !win.webContents.getURL().startsWith(server.url)) return;
	while (pendingOpen.length) win.webContents.send('open-path', pendingOpen.shift());
}

async function pick() {
	const r = await dialog.showOpenDialog(win, { title: 'Open an SSAF-T store (.zarr folder)', properties: ['openDirectory'] });
	return r.canceled || !r.filePaths.length ? null : r.filePaths[0];
}

const storeArgs = (argv) => argv.slice(1).filter((a) => !a.startsWith('-') && a !== '.' &&
	/\.zarr[\\/]?$/i.test(a) && fs.existsSync(a));

ipcMain.handle('open-dialog', () => pick());
ipcMain.on('opened', (_e, p) => { if (typeof p === 'string') app.addRecentDocument(p); });
ipcMain.on('clear-cache', () => clearCache());
ipcMain.on('show-log', () => shell.showItemInFolder(path.join(dirs().logs, 'server.log')));

/* ------------------------------------------------------------------ cache, menu */
async function clearCache() {
	const r = await dialog.showMessageBox(win, {
		type: 'question',
		message: 'Clear the cache?',
		detail: 'Tiles, solid surfaces and gene indexes cut from your stores are kept so that a store opens quickly the ' +
			'next time. Clearing them frees the disk space; a store is prepared again when it is next opened.',
		buttons: ['Clear', 'Cancel'],
		defaultId: 1,
		cancelId: 1
	});
	if (r.response !== 0) return;
	await stopServer();
	fs.rmSync(dirs().cache, { recursive: true, force: true });
	await restart();
}

/** Windows and Linux: none (the window's menu bar is removed).  macOS always has a menu bar: the app menu
 * (About, Quit) and Edit, without which copy and paste do not work in the page's text fields. */
function buildMenu() {
	Menu.setApplicationMenu(IS_MAC ? Menu.buildFromTemplate([{ role: 'appMenu' }, { role: 'editMenu' }]) : null);
}

/* ------------------------------------------------------------------ lifetime */
if (!app.requestSingleInstanceLock()) {
	app.quit();
} else {
	app.on('second-instance', (_e, argv) => {
		for (const p of storeArgs(argv)) openPath(p);
		if (win) { if (win.isMinimized()) win.restore(); win.focus(); }
	});
	app.on('open-file', (e, p) => { e.preventDefault(); openPath(p); });   // macOS: dropped on the Dock icon
	app.whenReady().then(async () => {
		buildMenu();
		createWindow();
		for (const p of storeArgs(process.argv)) openPath(p);
		try {
			await startServer();
		} catch (e) {
			return fatal(e);
		}
		loadPage();
		app.on('activate', () => { if (!win) createWindow(); });
	});
	app.on('window-all-closed', () => { if (!IS_MAC) app.quit(); });
	app.on('before-quit', () => { quitting = true; });
	app.on('will-quit', (e) => {
		if (server) {
			e.preventDefault();
			stopServer().then(() => app.quit());
		}
	});
}
