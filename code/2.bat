@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Germinal
set "glob=zola1885*"
:: tour du monde
:: set "glob=verne1872b*"
set "dfmin=1"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer tf --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer g2signed --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer g2 --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer g2pos --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer g2neg --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer fisher --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer fisherneg --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer fisherabs --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer fisherpos --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer logratio --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer simplemaths --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer bm25 --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer extf --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_heatlag.py ..\models\*.bin ..\data\docs.tsv ..\results\2_heatlags
