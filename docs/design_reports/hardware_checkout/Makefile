checkout: ../refs/fourth_year.bib checkout_plan.tex system_diagram.jpg
	@echo "Compiling checkout"
	pdflatex checkout_plan.tex
	bibtex checkout_plan.aux
	pdflatex checkout_plan.tex
	pdflatex checkout_plan.tex

clean: 
	rm -f *.bbl *.log *.aux *.dvi *.blg *.out
