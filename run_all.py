"""Run the whole project with one command:

    python run_all.py              # clean the data, then execute the notebook (about 10-15 min on a laptop CPU)
    python run_all.py --no-tabpfn  # same, but skip the transformer model (about 5 min, no PyTorch needed)

What it does
    1. checks that the dataset is in data/
    2. builds the cleaned data (src/cleaning.py)  -> data/processed/crime_clean.parquet
    3. executes notebooks/la_crime_analysis.ipynb top to bottom, saving the results into the
       notebook itself, plus figures (outputs/figures) and tables (outputs/tables)
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "lapd_crime_2020_present.csv"
NOTEBOOK = ROOT / "notebooks" / "la_crime_analysis.ipynb"


def main():
    if not DATA.exists():
        sys.exit(f"Dataset not found: {DATA}\nPut the CSV there (see README.md, section 'Data').")

    env = os.environ.copy()
    if "--no-tabpfn" in sys.argv:
        env["NO_TABPFN"] = "1"
        print("TabPFN (transformer) will be skipped.")

    clean_file = ROOT / "data" / "processed" / "crime_clean.parquet"
    if clean_file.exists():
        print("Step 1/2: cleaned data already exists - skipping (delete data/processed to rebuild).")
    else:
        print("Step 1/2: cleaning the data ...")
        subprocess.run([sys.executable, str(ROOT / "src" / "cleaning.py")], check=True, cwd=ROOT, env=env)

    print("Step 2/2: executing the notebook (this takes several minutes) ...")
    os.environ.update({k: v for k, v in env.items() if k == "NO_TABPFN"})
    execute_notebook()
    print("Done. Open notebooks/la_crime_analysis.ipynb, or look in outputs/figures and outputs/tables.")


def execute_notebook():
    """Run every cell of the notebook and save the outputs into it (same as 'Run All')."""
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.read(NOTEBOOK, as_version=4)
    ep = ExecutePreprocessor(timeout=7200, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK.parent)}})
    finally:                                   # keep partial output if a cell fails
        nbformat.write(nb, NOTEBOOK)


if __name__ == "__main__":
    main()
