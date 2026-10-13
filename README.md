# SSAF-T Viewer (desktop)

The SSAF-T site's cell-type map viewer (`/viewer`), as a desktop app for macOS, Windows and Linux.
It opens the SSAF-T store (`ssaft_sdata.zarr`, the folder an SSAF-T run writes) on your own computer and
gives it the same view the site gives its published datasets: the 3D / 2D cell-type map, solid surfaces,
SSAF-T cell segmentation, the image, and genes. Each part has its Show checkbox, and each cell type and gene
has its own checkbox.

The app shows one store at a time. Opening another replaces it, and the store's name is in the window title.
The site's dataset list and the dataset's information panel are not shown.

## How it is put together

| | | |
|---|---|---|
| `server/app.py` | the site's viewer server, **unchanged** | cuts tiles, solid surfaces, outlines and the gene index from a store |
| `web/store.html`, `web/local/` | the site's viewer page, **unchanged** | WebGL viewer |
| `server/desktop.py` | app only | `/api/open`: reads the store and writes its manifest record the way the site's records are written; builds layers when asked; lower memory |
| `web/desktop.js`, `web/desktop.css` | app only | hides the dataset section, Open button, drag and drop |
| `electron/` | app only | window, menus, file dialogs, starts and stops the server |

`server/desktop.py` starts the server on `127.0.0.1` on a free port. Only requests that carry a token
generated at launch are answered. Derived data goes in the cache in the app's data folder:
- Linux: `~/.config/SSAF-T Viewer`
- macOS: `~/Library/Application Support/SSAF-T Viewer`
- Windows: `%APPDATA%\SSAF-T Viewer`

A store opens immediately from the second time on. **File > Clear Cache…** empties the cache.

When the site's viewer changes, `sh scripts/sync-viewer.sh` copies `app.py`, `store.html` and `local/` again.

## Build

Build on each OS separately: PyInstaller cannot cross-compile, so the macOS app is built on a Mac and the
Windows app on Windows. The steps are the same on every platform.

```sh
# 1. Build environment (once): Python packages + PyInstaller + Node
micromamba create -f environment.yml        # or: conda env create -f environment.yml
micromamba activate ssaft-app

# 2. JS dependencies (once): Electron, electron-builder
npm install

# 3-a. Development run (no build): the server runs on the activated environment's Python
#      (or set SSAFT_PYTHON); a Python without the packages stops at start-up with a message saying so
npm start

# 3-b. Build an installer: build/server (frozen server) -> dist/
npm run dist        # macOS: dist/*.dmg   Windows: dist/*.exe (NSIS)   Linux: dist/*.AppImage
npm run dist:dir    # no installer, folder only (fastest to check): dist/<os>-unpacked/
```

How to launch the result:
- Linux: `dist/linux-unpacked/ssaft-viewer`
- macOS: `dist/mac-arm64/SSAF-T Viewer.app` or the dmg
- Windows: `dist\win-unpacked\SSAF-T Viewer.exe` or the installer

### Notes per platform
- **macOS**: the build is unsigned. An app built on your own Mac opens directly. To open it on another Mac,
  right-click it and choose Open, or run `xattr -dr com.apple.quarantine "/Applications/SSAF-T Viewer.app"`.
  To distribute it, you need Apple Developer signing and notarization.
- **Windows**: unsigned, so SmartScreen shows "Windows protected your PC". Click "More info" > "Run anyway".
- If Electron's download is blocked during `npm install`, set `ELECTRON_MIRROR`.
- If you see `Error: Cannot find module ...` on launch, `ELECTRON_RUN_AS_NODE=1` is set in your environment.
  Terminals inside VS Code extensions often set it. Unset it and run again:
  `env -u ELECTRON_RUN_AS_NODE npm start`.

## Trying it

1. Launch the app and use **File > Open SSAF-T Store** (`Ctrl/Cmd+O`) or the **Open .zarr folder…** button
   to choose an `ssaft_sdata.zarr` folder. You can also drop the folder on the window.
2. The first time a store opens, its tiles and solid surfaces are prepared. The panel shows
   "preparing … %". Large 3D sections take several minutes.
3. Under **Show**, tick and untick the layers. Under **Layers**, each cell type has its own checkbox,
   and each gene has its own checkbox under **Gene expression**.

Without data of your own, a small synthetic store can be made (it needs `spatialdata`):

```sh
pip install spatialdata
python tests/make_fixture.py /tmp/ssaft-fixture     # -> /tmp/ssaft-fixture/XENTEST/ssaft_sdata.zarr
```

Logs: **File > Show Server Log**, or `logs/server.log` in the app's data folder.

## Memory

The site's server has a lot of memory. On a laptop, `desktop.py` processes the large volumes in slices
(bincount, solid smoothing, label reading, the tissue envelope) instead. The output files are the same as
the site's; this was checked byte for byte.

Even so, the maps themselves are held in memory, and that sets a floor. For a 10,000 × 10,000 × 32 µm
Xenium section:
- the cell-type map needs about 3 GB per copy;
- the 3D cell segmentation's solid view needs the cell ids as int32, about 12 GB, and more on top.

For large 3D sections, 32 GB of RAM or more is recommended.

## Release (GitHub Actions)

`.github/workflows/release.yml` builds the installers when a version tag is pushed, and attaches them to a
GitHub Release for that tag:

```sh
git tag v0.1.0 && git push origin v0.1.0
```

- `SSAF-T-Viewer-<version>-mac-arm64.dmg`: Apple Silicon (M1 and later)
- `SSAF-T-Viewer-<version>-mac-x64.dmg`: Intel Mac
- `SSAF-T-Viewer-<version>-win-x64.exe`: Windows installer

The version is taken from the tag (`v0.1.0` → `0.1.0`), and a tag with a hyphen (`v0.2.0-rc1`) makes a
pre-release. Running the workflow by hand from the Actions tab builds the same files as workflow artifacts,
without a release. The builds take about 20 to 30 minutes.
