# Homework #4 — US Traffic Accidents Dashboard

Python/Streamlit + Plotly dashboard using the [US-Accidents Kaggle dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/), covering February 2016–March 2023.

**Public Streamlit URL:** `PASTE_YOUR_DEPLOYED_STREAMLIT_URL_HERE` (add after deployment).

## Data preparation: clean the entire source FIRST, then sample

Open `Homework4_Traffic_Accidents_Preparation.ipynb` in JupyterLab and run all cells in order. The notebook can initialize supporting Python files if opened from the `Classwork` directory. The official Kaggle CSV is downloaded to, or reused from, your **Downloads** folder.

There are two separate data-processing stages:

1. **Full cleaning (all ~7.7 million source records):** Read the original CSV in chunks; validate accident IDs, state, severity, timestamps and coordinates; remove invalid records and duplicate IDs across chunks; preserve missing optional weather values, replace missing weather labels with `Unknown`, and convert implausible weather measurements to missing. Save the ENTIRE cleaned dataset to `Downloads/US_Accidents_Kaggle/US_Accidents_March23_CLEANED.csv.gz`, with its `full_cleaning_summary.json` audit. This saved full cleaned file is not part of the Streamlit deployment. Processing uses chunks but keeps unique IDs in memory for global deduplication, so allow sufficient RAM and disk space.
2. **Sampling of already-cleaned records:** Read the saved full cleaned CSV, select approximately **2%** using a deterministic hash of the existing accident `ID`, compare state/year/severity distributions to the **full cleaned population**, and write the small `data/us_accidents_sample.csv.gz` plus `data/preparation_summary.json` for Streamlit. The sample is selected *after* the complete full-source cleaning stage.

The clean stage is reused without re-running if its file and audit match the unchanged raw source. Set `REBUILD_CLEANED=True` in the notebook to regenerate it, or pass `--rebuild-cleaned` in the CLI.

**Do not upload the large original Kaggle CSV, the full cleaned CSV, or Kaggle credentials to GitHub.** Upload only the small sampled CSV.gz, the portable audit JSON and your source files. The complete cleaned dataset is retained locally for documentation and further work.

To run without Jupyter, from this project directory:

```powershell
python prepare_data.py --input "C:\Users\YOUR_USERNAME\Downloads\US_Accidents_Kaggle\US_Accidents_March23.csv" --fraction 0.02
```

## Running the dashboard

From the project directory:

```powershell
python -m streamlit run app.py
```

Visit the local URL displayed (usually http://localhost:8501). Test the sidebar state, severity and year filters, map, severity chart, weather chart, monthly trend and empty selection handling. The map limits points for performance; other charts use the entire filtered 2% sample.

## GitHub and Streamlit Community Cloud

Commit `app.py`, `dashboard_logic.py`, `prepare_data.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml` and the contents of `data/` (especially `data/us_accidents_sample.csv.gz` and `data/preparation_summary.json`). Optionally include the preparation notebook and unit tests. Do not commit the multi-GB original or complete cleaned files.

On https://share.streamlit.io choose your repository, branch `main`, entry point `app.py`, and deploy. Check the public URL in an incognito browser window, then update this README.

## Validation and limitations

```powershell
python -m unittest discover -s tests -v
```

The tests use small artificial fixtures (never presented as real accidents). The full-cleaning audit counts raw, invalid, duplicated and valid records. Sampling diagnostics compare state, severity and year proportions between the **complete cleaned dataset and its sample**, not the raw and sampled datasets. These are one-dimensional comparisons; rare filtered groups may be less representative.

Accident severities indicate traffic disruption, not injury/fatality severity. Raw incident counts are not rates per mile driven, and changes in reporting coverage influence apparent trends. Weather comparisons are observational, not evidence of causality.

## Report

Write the required eight sections in Overleaf (Introduction; Data Exploration and Preparation; Dashboard Design; Key Features and Visualizations; Insights and Findings; Future Improvements; Conclusion; References). Include **real screenshots** of the running Streamlit dashboard and its charts/map, plus the public URL.

## References

- https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/
- https://docs.streamlit.io/
- https://plotly.com/python/plotly-express/
- https://github.com/Kaggle/kagglehub
