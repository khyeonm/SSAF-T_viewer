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
# pyarrow's parts the server never uses (it reads local parquet with pyarrow.dataset / .parquet / .compute):
# the cloud filesystems, ORC, Flight, Substrait and Gandiva.  pyarrow imports each of them inside a try, so
# their absence is what it expects.  Their shared libraries that libarrow itself links (the AWS, Azure and
# Google Cloud SDKs of the conda-forge build) stay: without them libarrow does not load.
PYARROW_UNUSED = ["_s3fs", "_gcsfs", "_azurefs", "_hdfs", "_orc", "_dataset_orc", "orc", "_flight", "flight",
                  "_substrait", "substrait", "gandiva", "_cuda", "cuda"]
EXCLUDE += [f"pyarrow.{m}" for m in PYARROW_UNUSED]


def prune(internal):
    """Files a running server never opens, removed from the build: pyarrow's C++ headers and Cython sources
    (about 470 files, each one more for the installer to write and an antivirus to scan) and the extension
    modules / libraries of the parts above that PyInstaller's pyarrow hook copies regardless."""
    pa = internal / "pyarrow"
    gone = 0
    for d in ("include", "includes", "src", "tests"):
        if (pa / d).is_dir():
            gone += sum(1 for f in (pa / d).rglob("*") if f.is_file())
            shutil.rmtree(pa / d)
    exts = (".pxd", ".pyx", ".pxi", ".h", ".hpp", ".cc", ".cpp", ".cmake")
    for f in list(pa.rglob("*")):
        stem = f.name.split(".")[0]
        if f.is_file() and (f.suffix in exts or (f.parent == pa and stem in PYARROW_UNUSED)):
            f.unlink()
            gone += 1
    # the libraries only those modules load (libarrow_python_flight / libarrow_substrait, any platform's name)
    for f in list(internal.iterdir()) + list(pa.iterdir()):
        n = f.name.lower()
        if f.is_file() and ("arrow_python_flight" in n or "arrow_substrait" in n or "arrow_flight" in n):
            f.unlink()
            gone += 1
    return gone


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
    print(f"pruned {prune(exe.parent / '_internal')} unused files from the build", flush=True)
    # smoke test: the frozen server starts, and the packages app.py imports lazily are inside it
    probe = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise SystemExit("the built server does not start:\n" + probe.stdout + probe.stderr)
    print(f"built {exe}")


if __name__ == "__main__":
    main()
