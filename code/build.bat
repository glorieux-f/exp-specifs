@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: freqlists from contingency table
:: python .\freqlists.py --top 0 ..\data\3_contingency\ ..\results\freqlists\
:: generate keyword 100
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords100 --vocab content --top 100 --min-doc-len 1000 
:: matrix of distances
:: python .\keywords-matrix.py --metric rbo --vocab content ..\results\keywords100 ..\results\romans19e-100motscles-rbo.tsv
:: plot formulas without parameters
:: python .\pcoa.py ..\results\romans19e-100motscles-rbo.tsv ..\results\romans19e-100motscles-specifs --exclude "CF" "DF" "*α*" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules de spécificité."
:: plots formulas with parameters
:: python .\pcoa-families.py ..\results\romans19e-100motscles-rbo.tsv --output-prefix ..\results\romans19e-100motscles-spec-families --families "subTF-IDF" "TF-IDF" HGT "G²" "χ²" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules paramétrables de spécificité."
:: 1000 keywords 
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords1000 --vocab content --top 1000 --min-doc-len 1000 --scorer logdice tscore tf cf df tfidfa16 chi2 chi2a0.43 g2 g2a1.4 hgt hgta1.47 subtfidf subtfidfa0.3 subtfidfa0.56 tfidf tfidfa1.14
:: plot keywords dispersion
:: python .\keywords-dispersion.py ..\data\3_contingency\ "..\results\keywords-verne\*.txt" verne1870a-42 --output-dir ..\results\keywords-verne\ --freq cf  --top 820 --cmap inferno_r
:: for stopwords dispersion
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords-stops --vocab nocaps --top 0 --min-doc-len 1000 --scorer logdice tscore tf cf df tfidfa16 chi2 chi2a0.43 g2 g2a1.4 hgt txm hgta1.47 subtfidf subtfidfa0.3 subtfidfa0.56 tfidf tfidfa1.14
python .\stopwords-dispersion.py ..\data\3_contingency\terms.tsv ..\results\keywords-stops\*.txt --output-dir ..\results\stops-plot/