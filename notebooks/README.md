# Notebooks folder

Copy your existing Task 2 notebooks here as-is (kept for reference only —
they no longer drive production code):

- task2nb1.ipynb — raw tables -> ml_orders_dataset.parquet  (-> src/data/)
- task2nb2.ipynb — labeling (is_late)                        (-> src/data/labeling.py)
- task2nb3.ipynb — stratified split                           (-> src/pipeline/split.py)
- task2nb4.ipynb — EDA (being edited manually; not part of the production pipeline)
- task2nb5.ipynb — feature engineering                        (-> src/features/, src/pipeline/feature_engineering.py)
- task2nb6.ipynb — model training                              (-> src/pipeline/train.py)

Nothing under src/ or app/ imports from this folder. It exists purely for
traceability between the original notebook cells and the refactored modules
(each module's docstring names which notebook/cell it replaces).
