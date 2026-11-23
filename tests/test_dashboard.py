"""Unit and integration tests using artificial test fixtures only (never deployed)."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_logic import (
    filtered_data, key_findings, map_chart, monthly_trend_table,
    severity_chart, trend_chart, visibility_chart, weather_chart,
)
from prepare_data import (clean, clean_full_source, sample_cleaned_source,
                          save_sampled_dataset, prepare)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.DataFrame([
            {"ID": "A1", "State": "NY", "Severity": 1, "Start_Time": "2018-01-03 08:10:11", "Start_Lat": 40.7, "Start_Lng": -73.9, "Weather_Condition": "Fair", "Visibility(mi)": 10.0},
            {"ID": "A2", "State": "CA", "Severity": 4, "Start_Time": "2022-05-03 14:10:11", "Start_Lat": 35.0, "Start_Lng": -118.2, "Weather_Condition": None, "Visibility(mi)": None},
            {"ID": "A3", "State": "NY", "Severity": 2, "Start_Time": "2023-03-03 14:10:11", "Start_Lat": 40.8, "Start_Lng": -73.2, "Weather_Condition": "Rain", "Visibility(mi)": 2.0},
            {"ID": "A4", "State": "ZZ", "Severity": 7, "Start_Time": "bad date", "Start_Lat": None, "Start_Lng": -150.0, "Weather_Condition": "Snow"},
        ])
        cls.df, cls.audit = clean(cls.raw)

    def test_cleaning(self):
        self.assertEqual(len(self.df), 3)
        self.assertEqual(self.audit["removed_missing_or_invalid_required"], 1)
        self.assertEqual(self.audit["weather_unknown_after_cleaning"], 1)
        self.assertEqual(self.audit["invalid_required_field_reasons_overlapping"]["invalid_state"], 1)

    def test_shared_filters(self):
        result = filtered_data(self.df, states=["NY"], severities=[2], years=(2023, 2023))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Weather_Condition"], "Rain")

    def test_filter_empty(self):
        self.assertTrue(filtered_data(self.df, states=["TX"], severities=[4]).empty)
        self.assertTrue(filtered_data(self.df, states=["NY"], severities=[]).empty)

    def test_figures(self):
        fig, n = map_chart(self.df, max_points=2)
        self.assertEqual(n, 2)
        self.assertTrue(fig.data)
        for fig in (severity_chart(self.df), weather_chart(self.df), trend_chart(self.df, 2018, 2023), visibility_chart(self.df)):
            self.assertGreaterEqual(len(fig.data), 1)
        self.assertFalse(severity_chart(self.df).layout.showlegend)

    def test_trend_does_not_invent_edge_months(self):
        table = monthly_trend_table(self.df, 2018, 2023)
        self.assertEqual(table.iloc[0]["Month_Start"], pd.Timestamp("2018-01-01"))
        self.assertEqual(table.iloc[-1]["Month_Start"], pd.Timestamp("2023-03-01"))
        self.assertNotIn(pd.Timestamp("2023-12-01"), set(table["Month_Start"]))
        subset = filtered_data(self.df, years=(2023, 2023))
        self.assertEqual(len(monthly_trend_table(subset, 2023, 2023)), 1)

    def test_optional_weather_outliers_are_not_used(self):
        raw = self.raw.iloc[:1].copy()
        raw["Visibility(mi)"] = -1
        raw["Temperature(F)"] = 999
        raw["Wind_Speed(mph)"] = -7
        cleaned, audit = clean(raw)
        self.assertEqual(len(cleaned), 1)
        for col in ("Visibility(mi)", "Temperature(F)", "Wind_Speed(mph)"):
            self.assertTrue(pd.isna(cleaned.iloc[0][col]))
            self.assertEqual(audit["implausible_optional_weather_values_set_missing"][col], 1)

    def test_full_clean_saved_then_sampled_and_audited(self):
        # Includes duplicate A1 across chunk boundaries, an invalid row and weather outliers.
        raw = pd.concat([self.raw, self.raw.iloc[:1]], ignore_index=True)
        raw.loc[1, "Wind_Speed(mph)"] = -3.0
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            original = folder / "raw.csv"
            cleaned = folder / "cleaned.csv.gz"
            clean_audit_file = folder / "clean_audit.json"
            sampled_file = folder / "sample.csv.gz"
            sample_audit_file = folder / "sample_audit.json"
            raw.to_csv(original, index=False)
            full = clean_full_source(original, cleaned, clean_audit_file, chunk_size=2)
            self.assertTrue(cleaned.is_file())
            self.assertTrue(clean_audit_file.is_file())
            self.assertEqual(full["raw_rows_inspected"], 5)
            self.assertEqual(full["removed_missing_or_invalid_required"], 1)
            self.assertEqual(full["duplicate_ids_removed_from_full_source"], 1)
            self.assertEqual(full["cleaned_rows"], 3)
            full_data = pd.read_csv(cleaned)
            self.assertEqual(len(full_data), 3)
            self.assertTrue(full_data["ID"].is_unique)
            self.assertTrue(pd.isna(full_data.loc[full_data["ID"].eq("A2"), "Wind_Speed(mph)"].iloc[0]))
            sampled, selection = sample_cleaned_source(cleaned, fraction=1, chunk_size=2)
            self.assertEqual(len(sampled), 3)
            self.assertEqual(selection["cleaned_rows_scanned_for_sampling"], 3)
            self.assertEqual(selection["selected_from_cleaned"], 3)
            self.assertEqual(selection["distribution_comparison"]["Severity"]["max_absolute_percentage_point_difference"], 0)
            audit = save_sampled_dataset(sampled, full, selection, output=sampled_file, audit_path=sample_audit_file)
            self.assertEqual(audit["preparation_order"][0], "clean_full_source")
            self.assertEqual(audit["cleaning"]["cleaned_rows"], 3)
            self.assertEqual(audit["output_rows"], 3)
            self.assertNotIn("ID", pd.read_csv(sampled_file).columns)
            reused = clean_full_source(original, cleaned, clean_audit_file, chunk_size=2)
            self.assertEqual(reused, full)

    def test_prepare_wrapper_has_same_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            raw_path = folder / "raw.csv"
            self.raw.to_csv(raw_path, index=False)
            audit = prepare(csv_path=raw_path, fraction=1.0,
                            cleaned_path=folder / "cleaned.csv.gz",
                            clean_audit_path=folder / "full.json", output=folder / "sample.csv.gz", audit_path=folder / "sample_audit.json")
            self.assertEqual(audit["output_rows"], 3)
            self.assertEqual(audit["cleaning"]["raw_rows_inspected"], 4)
            self.assertEqual(audit["sampling"]["cleaned_rows_scanned_for_sampling"], 3)

    def test_findings(self):
        self.assertGreaterEqual(len(key_findings(self.df)), 2)
        self.assertIn("No records", key_findings(self.df.iloc[0:0])[0])


if __name__ == "__main__":
    unittest.main()
