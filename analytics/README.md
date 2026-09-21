# Module 2 — Analytics Pipeline (Titanic)

## Run
```bash
pip install -r requirements.txt   # or the root requirements.txt
python 01_eda.py         # loads titanic (network on first run), profiles,
                          # cleans, tells the data story, saves titanic.csv
python 02_modeling.py    # reads the SAME titanic.csv, runs the full
                          # classification + regression pipeline
```
Charts are saved to `./charts/`. `titanic.csv` is the one-and-only offline
fallback of the cleaned dataset (`sns.load_dataset` is called exactly once,
inside `01_eda.py`). `best_pipeline.joblib` is the saved, reloadable, fully
fitted pipeline (preprocessing + tuned Random Forest).

## Design decisions (summary — full detail printed by the scripts)
- **Missing values**: threshold rule — <5% missing → drop rows (`embarked`,
  `embark_town`); 5–30% → impute (`age`, median); `deck` (~77% missing) is
  dropped as a column rather than imputed, since the miss rate is far too
  high to trust an imputed value and it's largely redundant with `pclass`.
- **Outliers**: reported via the IQR rule for `age` and `fare` (printed by
  `01_eda.py`); `fare` is right-skewed (mean > median > mode).
- **Correlation matrix**: computed on exactly `survived, pclass, age, sibsp,
  parch, fare` (the 6 numeric columns), excluding the derived `adult_male`
  and `alone` flags.
- **Modeling features**: `pclass, sex, age, sibsp, parch, fare, embarked`.
  Columns like `class`, `who`, `adult_male`, `alone`, `alive`, `deck`,
  `embark_town` are excluded from modeling as duplicates/derivatives of
  other columns (using them would leak the label or double-count a signal).
- **Preprocessing**: a `ColumnTransformer` (median-impute + scale for
  numeric, most-frequent-impute + one-hot for categorical) wrapped in a
  `Pipeline`, fit only on the training split.
- **Imbalance handling**: baseline vs. `class_weight='balanced'` vs. SMOTE
  (train fold only) compared on Random Forest; conclusion printed by the
  script based on the actual F1 values from your run.
- **Best pipeline**: the tuned Random Forest from `GridSearchCV` (full
  preprocessing + estimator) is what gets saved via `joblib.dump`.

## Files
- `01_eda.py` — Part A: load, profile, clean, EDA charts + interpretations.
- `02_modeling.py` — Part B: split, preprocess, train/evaluate 3 classifiers,
  imbalance comparison, hyperparameter tuning, regression side-task, final
  comparison + recommendation, save/reload pipeline.
- `charts/` — saved plot images (generated on run).
- `titanic.csv` — cleaned dataset, the one committed offline fallback (generated on run).
- `best_pipeline.joblib` — saved fitted pipeline (generated on run).

## Note on the written interpretations
Every task above (skew direction, correlation interpretation, the 4
multivariate-chart interpretations, the imbalance conclusion, the
heteroscedasticity conclusion, and the final recommendation) is printed
directly by the scripts as it runs — copy the printed text for your actual
run into this README (or the notebook Markdown cells, if you convert these
scripts to notebooks) so a grader can read it without re-running the code.
