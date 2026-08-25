# Build the paper with latexmk (TeX Live 2026 assumed on PATH).
#   make          -> paper/main.pdf
#   make clean    -> remove build products (keeps the PDF)
#   make distclean-> remove build products and the PDF
#
# Fallback when TeX Live is not reachable (e.g. the external drive holding it
# is unplugged): `make tectonic` builds with the self-contained tectonic
# engine (https://tectonic-typesetting.github.io), which resolves natbib /
# bibtex automatically. Output is identical up to font hinting.

LATEXMK ?= latexmk
TECTONIC ?= tectonic
MAIN = main

.PHONY: all clean distclean tectonic

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

tectonic:
	$(TECTONIC) -X compile --keep-intermediates --keep-logs $(MAIN).tex

clean:
	$(LATEXMK) -c $(MAIN).tex 2>/dev/null || rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf
