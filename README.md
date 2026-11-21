# Crime in Los Angeles, 2020–2023: Patterns, Statistics and Arrest Prediction

A complete **data-analytics life-cycle** project on the LAPD *Crime Data from 2020 to Present* open-data set
(≈ 1 million police crime reports): problem understanding → data collection → data understanding →
cleaning → transformation → EDA → statistical tests → pattern discovery → modelling (including a
**transformer**, TabPFN) → evaluation → interpretation → recommendations.

| | |
|---|---|
| **Data source** | City of Los Angeles Open Data portal, *Crime Data from 2020 to Present* (published by the Los Angeles Police Department, dataset id `2nrs-mtv8`, https://data.lacity.org) |
| **Main deliverable** | `notebooks/la_crime_analysis.ipynb` (already executed – results are saved inside it) |
| **Presentation** | `presentation/LA_Crime_Analytics.pptx` |

### Results at a glance

* **877,322** reports analysed (incidents 2020-2023); only **9.4 %** end in an arrest – from 44 % for homicide/kidnapping down to
  2–3 % for vehicle crime and fraud.
* Crime type and weapon are the strongest associations with arrest; area, time of day, weekday and victim sex/age have negligible-to-small
  effects (reported with effect sizes). Arrested cases are reported sooner (median 0 vs 1 day).
* **9 modus-operandi patterns** found with TF-IDF + SVD + K-Means; their arrest rates range from 0.7 % to 20 %.
* Predicting an arrest at report time (stratified 80/20 split, 175 K test reports):

  | Model | ROC-AUC | PR-AUC (chance ≈ 0.09) |
  |---|---|---|
  | Logistic Regression | 0.814 | 0.333 |
  | **XGBoost** | **0.842** | **0.390** |

* Transformer comparison (TabPFN v2, 3,000 training rows, 1,500 test reports): TabPFN **0.801** vs XGBoost on the same rows
  **0.771** ROC-AUC (confidence intervals overlap; XGBoost on all rows: 0.845).

---

## 1. Folder structure

```
la_crime_analytics/
├── README.md                     <- this file
├── requirements.txt              <- Python packages (minimum versions)
├── requirements-tested.txt       <- exact versions it was tested with
├── run_all.py                    <- one-command run (clean data + execute notebook)
├── data/
│   └── lapd_crime_2020_present.csv      <- the dataset (raw, unmodified)
├── notebooks/
│   └── la_crime_analysis.ipynb          <- the analysis (all 12 life-cycle stages)
├── src/
│   ├── cleaning.py               <- load, clean, feature engineering (also runnable on its own)
│   ├── crime_groups.py           <- transparent keyword rules: 140 crime descriptions -> 13 groups
│   ├── mo_patterns.py            <- modus-operandi clustering (TF-IDF + SVD + K-Means)
│   ├── modeling.py               <- features, Logistic Regression, XGBoost, TabPFN, metrics
│   ├── stats_utils.py            <- chi-square / Mann-Whitney / odds ratio + effect sizes
│   └── viz_style.py              <- shared chart style
├── outputs/
│   ├── figures/                  <- every chart (PNG)
│   └── tables/                   <- test results, model metrics, cleaning log (CSV/JSON)
└── presentation/
    ├── LA_Crime_Analytics.pptx   <- final slides
    ├── build_deck.py             <- script that rebuilds the slides from outputs/
    └── deck_helpers.py
```

`data/processed/` is created automatically on the first run (a fast cache of the cleaned data).

---

## 2. Requirements

* **Python 3.10 or newer** (developed and tested on Python 3.14.5 / Windows 11; the exact package versions used are in
  `requirements-tested.txt` – use that file if the normal install gives a version conflict)
* About **3 GB of free RAM** and **1 GB of disk space** (the raw CSV is ≈ 300 MB)
* Internet access **only** for installing packages and, once, for downloading the TabPFN model weights
  (≈ 30–60 MB, cached afterwards). No GPU, no account and no API key are needed.
* ≈ 10–15 minutes for a full run on a laptop CPU (≈ 5 minutes without the transformer)

---

## 3. Installation

Open a terminal **inside the project folder** (`la_crime_analytics/`).

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
*(If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass` first, or use `.venv\Scripts\activate.bat` in Command Prompt.)*

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch (needed only by the TabPFN transformer). The default
`pip install torch` is fine; on Linux you can save disk space with the CPU build:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## 4. Configuration

Nothing needs to be configured. The dataset is already in `data/lapd_crime_2020_present.csv`.

Optional settings are in the notebook's **"0 · Setup"** cell:

| Setting | Default | Meaning |
|---|---|---|
| `SEED` | 42 | Random seed (makes results reproducible) |
| `RUN_TABPFN` | `True` | Set `False` (or run with `--no-tabpfn`) to skip the transformer model |
| `TABPFN_N_TRAIN` | 3000 | Training rows given to TabPFN as context (CPU-friendly) |
| `TABPFN_N_TEST` | 1500 | Test rows used for the transformer comparison |

**Using a different copy of the data?** Download *Crime Data from 2020 to Present* from
https://data.lacity.org (Export → CSV) and save it as `data/lapd_crime_2020_present.csv`.
The code expects the original portal column names (`DR_NO`, `Date Rptd`, `DATE OCC`, …).

---

## 5. How to run

### Option A – one command (recommended)
```bash
python run_all.py
```
Cleans the data, then executes the whole notebook top-to-bottom and saves the results into the
notebook, `outputs/figures/` and `outputs/tables/`. Add `--no-tabpfn` to skip the transformer.

### Option B – interactively in Jupyter
```bash
python -m jupyter notebook notebooks/la_crime_analysis.ipynb
```
Then choose **Run → Run All Cells**. The first run builds the cleaned data (1–2 min).

### Option C – cleaning only
```bash
python src/cleaning.py
```
Writes `data/processed/crime_clean.parquet` and `outputs/tables/cleaning_log.json`.

### Rebuilding the slides (optional)
```bash
python presentation/build_deck.py
```

---

## 6. What the notebook contains

| Stage | Notebook section | Key methods |
|---|---|---|
| 1 Problem understanding | §1 | Stakeholders, 6 analytical questions, success criteria |
| 2 Data collection | §2 | Source, collection process, data dictionary |
| 3 Data understanding | §3 | Chunked profiling, missing values, placeholder values, discovery that 2024 is incomplete |
| 4 Cleaning | §4 | Logged, justified fixes (age 0 → missing, 0,0 coordinates, sex X, dates, …) |
| 5 Transformation / integration | §5 | Time features, crime grouping (140 → 13), lookup tables, target variable |
| 6 EDA | §6 | Time trends, hour × weekday heat-map, areas, map, crime types, victims, weapons, reporting delay |
| 7 Statistical analysis | §7 | Chi-square + Cramér's V, Mann-Whitney + rank-biserial r, odds ratios, Kendall trend test, Kruskal-Wallis; Bonferroni |
| 8 Pattern discovery | §8 | Modus-operandi clusters via TF-IDF + SVD + K-Means (k chosen by silhouette) |
| 9 Modelling | §9 | Logistic Regression, XGBoost, **TabPFN v2 (transformer)** vs XGBoost on identical data |
| 10 Evaluation | §10 | ROC-AUC, PR-AUC, precision/recall @ top 9 %, bootstrap CIs, confusion matrix |
| 11 Interpretation | §11 | Exact feature contributions (SHAP values from XGBoost) |
| 12 Communication | §12 | Insights, recommendations, limitations |

**Where the transformer is used:** TabPFN (a transformer pre-trained on millions of synthetic tables)
is one of the three prediction models in §9–10. It learns *in context* – the training rows are fed to the
network at prediction time instead of being used for gradient descent – and is compared with XGBoost
trained on exactly the same rows.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Dataset not found` | Put the CSV at `data/lapd_crime_2020_present.csv` (see §4). |
| `MemoryError` | Close other programs (≥ 3 GB free RAM recommended). The loader is already memory-lean. |
| TabPFN weights cannot be downloaded (offline / firewall) | The notebook prints "TabPFN skipped" and continues with the other models. Or run `python run_all.py --no-tabpfn`. |
| `No module named torch` | `pip install torch`, or run without the transformer (`--no-tabpfn`). |
| Kernel `python3` not found when running the notebook | `python -m ipykernel install --user --name python3` |
| Slides not rebuilding | `pip install python-pptx pillow` |

---

## 8. Data notes and limitations

* The data are **transcribed from paper crime reports** by the publisher and may contain errors;
  unknown locations are recorded as (0°, 0°) and addresses are rounded to the hundred block.
* **2024 is excluded** because recent cases have not had time to reach an outcome (the arrest rate falls
  from ≈ 9 % to 2.5 % during 2024) and monthly volumes drop by more than half.
* Arrest rates are **associations, not causes**, and reflect police practice as well as crime
  characteristics. Victim demographics are those of the *victim*, not the suspect. The model is a
  teaching/analytics example and **must not be used to make decisions about individuals**.
