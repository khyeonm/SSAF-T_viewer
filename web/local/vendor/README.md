# Vendored readers

They live OUTSIDE `src/` on purpose: the project type-checks `src/**` with `checkJs`, and
third-party JavaScript carrying its own JSDoc types only adds noise there.  `src/lib/local/*`
imports them by relative path.

The "open your own data" page reads a SpatialData store in the browser, so the formats it has to
decode come with it.  These three are copied here, unchanged, rather than installed: the site's
`node_modules` is shared with a running deployment and its pnpm store is a major version behind, so
adding dependencies would mean re-linking that tree.  They are plain ESM sources, which Vite bundles
like any other module.

| directory               | package                  | version | what it reads                      |
| ----------------------- | ------------------------ | ------- | ---------------------------------- |
| `hyparquet/`            | hyparquet                | 1.31.3  | parquet (shapes, points); SNAPPY pages are built into it |
| `fzstd/`                | fzstd                    | 0.1.1   | zstd: our zarr chunks, and zstd parquet pages |

`hyparquet-compressors` is deliberately NOT here: it reaches the other codecs through bare imports
of its own dependencies, which a vendored copy cannot resolve.  `local/parquet.ts` hands hyparquet a
small `compressors` map built from fzstd instead.

To update one: `npm pack <name>`, unpack, and copy its `src/` (for fzstd, `esm/index.mjs`) over the
directory, keeping this table's version column current.
