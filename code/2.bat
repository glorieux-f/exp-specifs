set "glob=zola1885*"
set "scorer=g2"
set "dfmin=1"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer g2 --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer lafon --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_doc-vsm.py ..\data\ ..\models\ --select "%glob%" --scorer bm25 --vocab content --dims 200 --dfmin "%dfmin%"
python .\2_heatmap.py ..\models\*.bin ..\data\docs.tsv ..\results\2_heatmaps
