#!/bin/sh
# Copy the web viewer's own files into the app, unchanged.
#
# The app runs the SAME viewer as the site: viewer/app.py (the server that cuts tiles, solids and gene
# indexes), viewer/store.html (the WebGL page) and viewer/local/ (the in-browser reader the page falls
# back to without WebGL2).  None of them is edited here -- the desktop parts live beside them
# (server/desktop.py, web/desktop.js) -- so bringing the app up to date with the site is this copy.
#
#   scripts/sync-viewer.sh [path/to/ssaf-t_web/viewer]
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-/bce/groups/pnucolab/analysis/l202255179/ssaf-t/ssaf-t_web/viewer}
cp "$SRC/app.py" "$HERE/server/app.py"
cp "$SRC/store.html" "$HERE/web/store.html"
rm -rf "$HERE/web/local"
mkdir -p "$HERE/web/local"
# the reader's modules and its vendored decoders; no .ts sources or type files are needed at run time
(cd "$SRC/local" && find . -name '*.js' -o -name '*.mjs' -o -name 'README.md') | while read -r f; do
	mkdir -p "$HERE/web/local/$(dirname "$f")"
	cp "$SRC/local/$f" "$HERE/web/local/$f"
done
echo "synced from $SRC:"
(cd "$HERE" && ls -la server/app.py web/store.html && find web/local -type f | wc -l)
