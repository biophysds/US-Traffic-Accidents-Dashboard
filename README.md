# US Traffic Accidents Dashboard

An interactive traffic accident dashboard developed using Python, Streamlit, Pandas, and Plotly. The application explores reported traffic accidents across the United States from February 2016 through March 2023.

Users can investigate geographic distributions, accident severity, weather conditions, and temporal trends through interactive filters and visualizations.

## Dashboard Features

The dashboard includes:

- **State filtering:** Explore accident records from individual states, multiple states, or all available states.
- **Severity filtering:** Select any combination of severity levels 1–4, representing the reported impact on traffic.
- **Time filtering:** Examine accidents within a selected year range.
- **Accident location map:** Interactive geographic visualization of reported accident locations, colored by severity.
- **Severity distribution:** Bar chart comparing accident counts across severity levels.
- **Weather analysis:** Visualizations of reported weather conditions and visibility.
- **Monthly trends:** Time-series visualization of accident records across the dataset's observation period.
- **Dynamic findings:** Descriptive statistics and observations that update with the selected filters.

The interface is organized into three tabs: **Overview & Map**, **Weather & Time**, and **Data Notes**.

## Dataset

**Source:** [US-Accidents: A Countrywide Traffic Accident Dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/)

The original dataset contains 7,728,394 accident records and includes geographic coordinates, timestamps, severity levels, and weather-related measurements.

### Data Preparation

The dataset was processed using Pandas with chunk-based operations to manage memory usage.

The complete dataset was cleaned before sampling:

1. Loaded and inspected all 7,728,394 source records.
2. Validated accident identifiers, state codes, severity levels, timestamps, and geographic coordinates.
3. Removed seven records with invalid timestamps and checked for duplicate accident IDs.
4. Standardized missing weather labels as `Unknown` and retained missing optional measurements.
5. Identified implausible numeric weather measurements and replaced those values with missing values.
6. Converted timestamps and derived year and month fields.
7. Saved the complete cleaned dataset containing **7,728,387 records**.

The full cleaned dataset was retained locally. A smaller subset was prepared separately for the interactive application.

### Sampling Strategy

A reproducible 2% sample was selected from the saved, fully cleaned dataset using deterministic hashing of accident identifiers.

| Measure | Records |
|---|---:|
| Original dataset | 7,728,394 |
| Invalid records excluded | 7 |
| Complete cleaned dataset | 7,728,387 |
| Dashboard sample | 155,283 |
| Actual sampling fraction | 2.009% |

The sample preserves broad patterns in the original cleaned dataset while reducing memory consumption and improving dashboard responsiveness.

Sampling validation compared the state, severity, and year distributions against the complete cleaned dataset. Maximum absolute differences were 0.197, 0.103, and 0.105 percentage points, respectively.

The application uses the sampled dataset for its charts and statistics. The location map additionally limits the number of displayed points to maintain interactivity.

## Exploratory Findings

The cleaned sample contains 155,283 records representing 49 states and spanning 2016–2023.

For the full sample:

- **Severity 2** was the most frequently recorded severity level, accounting for 123,869 records (79.8%).
- **Fair** was the most frequently reported known weather condition, appearing in 51,721 of 151,788 records with specified weather conditions.
- **December 2022** had the highest observed monthly count, with 3,959 sampled accident records.

These findings describe the distribution of recorded accidents within the sample. They should not be interpreted as measurements of accident risk or evidence of causal relationships.

## Project Structure

| File | Description |
|---|---|
| `app.py` | Streamlit interface, interactive filters, and dashboard layout |
| `dashboard_logic.py` | Data filtering, Plotly visualizations, and descriptive findings |
| `prepare_data.py` | Full-dataset cleaning, validation, and reproducible sampling |
| `requirements.txt` | Python package dependencies |
| `data/us_accidents_sample.csv.gz` | Cleaned sample used by the dashboard |
| `data/preparation_summary.json` | Data-cleaning and sampling audit |
| `tests/test_dashboard.py` | Automated tests for data processing and visualizations |

The original source CSV and complete cleaned CSV are maintained locally rather than included in the GitHub repository.

## Running Locally

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the application from the project directory:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local address, typically `http://localhost:8501`.

The included compressed dataset allows the dashboard to run without downloading the original multi-gigabyte source file.

### Reproducing the Data Preparation

The complete data-cleaning and sampling process can be reproduced using the original Kaggle dataset and `prepare_data.py`.

For example:

```bash
python prepare_data.py --input "path/to/US_Accidents_March23.csv" --fraction 0.02
```

This cleans and saves the full dataset before selecting the deployment sample.

### Testing

Run the automated tests using:

```bash
python -m unittest discover -s tests -v
```

The tests cover data cleaning, filtering, visualizations, sampling, and the full-cleaning workflow.

## Limitations

The dashboard is intended for exploratory analysis of reported accident records.

- The dataset does not include every traffic accident in the United States, and reporting coverage varies by location and time.
- The deployed application uses a 2% sample. Small geographic or temporal selections may contain relatively few observations.
- Accident counts are not adjusted for traffic volume, miles traveled, or other exposure measures.
- Severity levels describe traffic impact rather than injury or fatality severity.
- Weather comparisons are descriptive and do not establish causation.
- The dataset begins in February 2016 and ends in March 2023; incomplete coverage and changes in reporting can influence apparent time trends.

## References

- [US-Accidents Dataset — Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [KaggleHub](https://github.com/Kaggle/kagglehub)
