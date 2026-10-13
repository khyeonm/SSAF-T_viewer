/**
 * What the page may ask of the app (web/desktop.js uses it): a file dialog, the path of a file dropped on
 * the window, and the stores the menus open.  Nothing else of Node or Electron reaches the page.
 */
'use strict';
const { contextBridge, ipcRenderer, webUtils } = require('electron');

contextBridge.exposeInMainWorld('ssaft', {
	/** a folder dialog for the .zarr store; resolves to the chosen path, or null */
	openDialog: () => ipcRenderer.invoke('open-dialog'),
	/** the path on disk of a File dropped on the page */
	pathForFile: (file) => webUtils.getPathForFile(file),
	/** a store the File menu, the Dock or a second launch wants opened */
	onOpenPath: (cb) => ipcRenderer.on('open-path', (_e, p) => cb(p)),
	/** a store opened: for File > Open Recent */
	opened: (p) => ipcRenderer.send('opened', p)
});
