final_report: final_report.tex final_report.bib
	pdflatex final_report.tex
	bibtex final_report.aux
	pdflatex final_report.tex
	pdflatex final_report.tex

final_report.bib: ../refs/fourth_year.bib final_report.aux
	bibtex final_report.aux

final_report.aux: final_report.tex
	pdflatex final_report.tex

clean:
	rm -f *.bbl *.log *.aux *.dvi *.blg *.out
