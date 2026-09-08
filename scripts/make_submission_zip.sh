#!/usr/bin/env bash
# Build the LaTeX source archive the publisher asks for alongside the PDFs.
# Cambridge journals take it under the "LaTeX Source Files" designation.
#
#   bash scripts/make_submission_zip.sh [OUT]
#
# The submission tree in paper/submission/ reaches its assets by relative path
# (../tables, ../figures, ../references) so that no generated exhibit is ever
# duplicated in the repository. An archive cannot rely on that, so this script
# assembles a flattened copy, rewrites those three paths to local ones, builds
# both documents to prove the copy is self-contained, and only then zips it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/paper/submission/latex-source.zip}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
SRC="$WORK/src"

mkdir -p "$SRC"/{sections,supp,tables,figures}
cd "$ROOT/paper/submission"
cp manuscript.tex supplement.tex preamble.tex frontmatter.tex backmatter.tex README.txt "$SRC/" 2>/dev/null || \
  cp manuscript.tex supplement.tex preamble.tex frontmatter.tex backmatter.tex "$SRC/"
cp sections/*.tex "$SRC/sections/"
cp supp/*.tex "$SRC/supp/"
cp "$ROOT/paper/tables/"*.tex "$SRC/tables/"
cp tables/*.tex "$SRC/tables/"          # AFTER the parent's: the submission tree
                                        # carries the abridged views and a
                                        # corrected copy of tab-h2-coverage
cp "$ROOT/paper/figures/"*.pdf "$SRC/figures/"
cp "$ROOT/paper/references.bib" "$SRC/"

# assets are local in the archive, not one directory up
sed -i 's|{\.\./figures/}{figures/}|{figures/}|; s|\.\./figures/#2\.pdf|figures/#2.pdf|g' "$SRC/preamble.tex"
sed -i 's|\bibliography{\.\./references}|\bibliography{references}|' "$SRC/manuscript.tex" "$SRC/supplement.tex"

# a source archive that does not compile is worse than none
cd "$SRC"
for doc in manuscript supplement; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$doc.tex" >/dev/null 2>&1 || {
    echo "FAILED: $doc.tex does not build from the assembled copy" >&2; exit 1; }
  pages=$(grep -o "Output written on $doc.pdf ([0-9]* pages" "$doc.log" | grep -o '[0-9]*' | tail -1)
  undef=$(grep -c undefined "$doc.log" || true)
  echo "[zip] $doc.pdf: $pages pages, $undef undefined references"
done

latexmk -C >/dev/null 2>&1 || true
rm -f ./*.pdf ./*.aux ./*.log ./*.bbl ./*.blg ./*.out ./*.toc ./*.fls ./*.fdb_latexmk sections/*.aux supp/*.aux
rm -f "$OUT"
zip -q -r -9 "$OUT" . -x '.*'
echo "[zip] wrote $OUT ($(du -h "$OUT" | cut -f1), $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"
