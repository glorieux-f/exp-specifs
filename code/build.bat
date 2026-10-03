@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: generate keyword 100
python .\keywords.py ..\data\3_contingency\ ..\results\keywords100 --vocab content --top 100 --min-doc-len 1000 
:: python .\keywords.py ..\data\3_contingency\ ..\results\keywords1000 --vocab content --top 1000 --min-doc-len 1000 --scorer tfidf --scorer subtfidf  --scorer subtfidfa0.56 --scorer tfidfa1.14 --scorer hgta1.47 --scorer hgt --scorer g2a1.4  --scorer chi2a0.43 --scorer tf --scorer extf
