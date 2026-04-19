LATEX   = pdflatex
BIBTEX  = bibtex
MAIN    = main
FLAGS   = -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all:
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex
	@echo "PDF ready: $(MAIN).pdf"

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot

clean-all: clean
	rm -f $(MAIN).pdf
