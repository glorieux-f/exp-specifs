@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: generate keyword 100
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords100 --vocab content --top 100 --min-doc-len 1000 
:: matrix of distances
:: python .\keywords-matrix.py --metric rbo --vocab content ..\results\keywords100 ..\results\romans19e-100motscles-rbo.tsv
:: plot formulas without parameters
:: python .\pcoa.py ..\results\romans19e-100motscles-rbo.tsv ..\results\romans19e-100motscles-specifs --exclude "CF" "DF" "*α*" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules de spécificité."
:: plots formulas with parameters
python .\pcoa-families.py ..\results\romans19e-100motscles-rbo.tsv --output-prefix ..\results\romans19e-100motscles-spec-families --families "subTF-IDF" "TF-IDF" HGT "G²" "χ²" --flip 2 --title "Balzac, Dumas, Sand, Verne, Zola ; plus de 7000 chapitres ; 100 mots clés. Distances entre formules paramétrables de spécificité."