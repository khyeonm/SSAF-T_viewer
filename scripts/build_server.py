#!/usr/bin/env python
"""Build the viewer server (server/desktop.py + server/app.py) into a standalone folder with PyInstaller.

    python scripts/build_server.py            -> build/server/ssaft-server/ssaft-server[.exe]

Run it with the Python that has the server's packages (environment.yml), on each platform the app is
built for: PyInstaller does not cross-compile, so the macOS app is built on a Mac and the Windows one on
Windows.  electron-builder then copies the folder into the app (package.json, extraResources).

app.py imports most of what it uses inside its functions (zarr, scipy, pyarrow, geopandas, PIL...), where
PyInstaller's analysis does not look, so they are named here.  zarr 3 finds its codecs through package
entry points, so the packages' metadata is copied too.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "server"
WORK = ROOT / "build" / "pyinstaller"

HIDDEN = [
    "app", "zarr", "zarr.storage", "zarr.codecs", "numcodecs", "numcodecs.zarr3", "numcodecs.zstd",
    "numcodecs.blosc", "numcodecs.gzip", "numcodecs.lz4", "scipy.ndimage", "pyarrow", "pyarrow.compute",
    "pyarrow.dataset", "pyarrow.parquet", "geopandas", "shapely", "PIL.Image", "tifffile",
]
# zarr picks its codecs at run time, so their modules are named in full; the other packages come with
# PyInstaller's own hooks (pyinstaller-hooks-contrib), which take what they need.  `--collect-all` on them
# shipped their tests and sample data too: thousands of files more for an installer to write and an
# antivirus to scan.
SUBMODULES = ["zarr", "numcodecs"]
METADATA = ["zarr", "numcodecs", "donfig", "geopandas", "shapely", "pyarrow", "numpy", "crc32c"]
EXCLUDE = ["cupy", "cupyx", "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest",
           "dask", "distributed", "numba", "llvmlite", "torch", "zarr.testing", "numcodecs.tests",
           "pyarrow.tests", "pandas.tests", "scipy.tests"]


def main():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit("PyInstaller is not installed in this Python; see environment.yml")
    args = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--console",
            "--name", "ssaft-server", "--distpath", str(OUT), "--workpath", str(WORK),
            "--specpath", str(WORK), "--paths", str(ROOT / "server")]
    for h in HIDDEN:
        args += ["--hidden-import", h]
    for c in SUBMODULES:
        args += ["--collect-submodules", c]
    for m in METADATA:
        try:
            __import__(m)
            args += ["--copy-metadata", m]
        except ImportError:
            pass
    for x in EXCLUDE:
        args += ["--exclude-module", x]
    args.append(str(ROOT / "server" / "desktop.py"))
    shutil.rmtree(OUT / "ssaft-server", ignore_errors=True)
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)
    exe = OUT / "ssaft-server" / ("ssaft-server.exe" if sys.platform == "win32" else "ssaft-server")
    if not exe.exists():
        raise SystemExit(f"build finished but {exe} is missing")
    # smoke test: the frozen server starts, and the packages app.py imports lazily are inside it
    probe = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise SystemExit("the built server does not start:\n" + probe.stdout + probe.stderr)
    print(f"built {exe}")


if __name__ == "__main__":
    main()
