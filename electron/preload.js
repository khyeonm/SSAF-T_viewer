/**
 * What the page may ask of the app (web/desktop.js uses it): a file dialog, the path of a file dropped on
 * the window, the stores a launch or the Dock opens, clearing the cache, and the log.  Nothing else of Node or Electron reaches the page.
 */
'use strict';
const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('ssaft', {
	/** a Cancel / Continue question; resolves to true for Continue */
	confirm: (message) => ipcRenderer.invoke('confirm', message),
	/** the app's version (package.json), for the panel's footer */
	version: () => ipcRenderer.invoke('version'),
	/** a folder dialog for the .zarr store; resolves to the chosen path, or null */
	openDialog: () => ipcRenderer.invoke('open-dialog'),
	/** the path on disk of a File dropped on the page */
	pathForFile: (file) => webUtils.getPathForFile(file),
	/** a store the File menu, the Dock or a second launch wants opened */
	onOpenPath: (cb) => ipcRenderer.on('open-path', (_e, p) => cb(p)),
	/** a store opened: for the OS's recent documents */
	opened: (p) => ipcRenderer.send('opened', p),
	/** empty the cache of prepared tiles (asks first; restarts the server) */
	clearCache: () => ipcRenderer.send('clear-cache'),
	/** Ctrl/Cmd+Shift+P: show or hide the log line */
	onToggleLog: (cb) => ipcRenderer.on('toggle-log', () => cb()),
	/** show the server's log file in the file manager */
	showLog: () => ipcRenderer.send('show-log')
});
