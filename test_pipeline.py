#!/usr/bin/env python3
"""
Test script for step2.py and step7.py pipeline
"""
import unittest
import tempfile
import shutil
import os
import json
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our separated modules
import step2
import step7


class TestStep2Step7Pipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to isolate test files
        self.test_dir = tempfile.mkdtemp()
        
        # Store original values
        self.original_step2_base_dir = step2.BASE_DIR
        self.original_step7_base_dir = step7.BASE_DIR
        
        # Monkey-patch BASE_DIR for both modules
        step2.BASE_DIR = Path(self.test_dir)
        step2.STEP1_JSON = step2.BASE_DIR / "step1.json"
        step2.STEP2_OUTPUT = step2.BASE_DIR / "step27.json"
        
        step7.BASE_DIR = Path(self.test_dir)
        step7.STEP2_OUTPUT = step7.BASE_DIR / "step27.json"
        
        # Ensure directories exist
        os.makedirs(step2.BASE_DIR, exist_ok=True)

        # Create a sample step1.json with two matches: one in-play (status_id=4), one not started (status_id=1)
        sample_data = {
            "live_matches": {
                "code": 0,
                "response_time": 0.123,
                "matches": [
                    {
                        "id": 1,
                        "status_id": 4,
                        "status": "Second half",
                        "match_time": 53,
                        "home_team": "Team A",
                        "away_team": "Team B",
                        "home_scores": [1, 2, 3],
                        "away_scores": [0, 1, 2],
                        "home_team_id": 10,
                        "away_team_id": 20,
                        "competition_id": 100,
                        "competition_name": "League1",
                        "competition_logo": "logo_url1",
                        "home_logo": "home_logo_url",
                        "away_logo": "away_logo_url",
                        "round": "Round 1",
                        "venue_id": "Venue1",
                        "referee_id": "Ref1",
                        "neutral": 0,
                        "coverage": {},
                        "scheduled": "2025-06-06T12:00:00Z",
                        "odds": {
                            "eu": [["ft", "5", 100.0, 200.0, 300.0]]
                        },
                        "environment": {
                            "weather": "1",
                            "temperature": "20 C",
                            "wind": "5 m/s",
                            "pressure": "1013",
                            "humidity": "50"
                        },
                        "events": [
                            {"type": "goal", "time": "45", "team": "home", "player": "Player1", "detail": "details"}
                        ]
                    },
                    {
                        "id": 2,
                        "status_id": 1,
                        "status": "Not started",
                        "match_time": 0,
                        "home_team": "Team C",
                        "away_team": "Team D",
                        "home_scores": [],
                        "away_scores": [],
                        "home_team_id": 30,
                        "away_team_id": 40,
                        "competition_id": 200,
                        "competition_name": "League2",
                        "competition_logo": "logo_url2",
                        "home_logo": "home_logo_url2",
                        "away_logo": "away_logo_url2",
                        "round": "Round 2",
                        "venue_id": "Venue2",
                        "referee_id": "Ref2",
                        "neutral": 1,
                        "coverage": {},
                        "scheduled": "2025-06-07T15:00:00Z",
                        "odds": {},
                        "environment": {},
                        "events": []
                    }
                ]
            },
            "match_details": {
                "1": {"results": [{"id": 1, "home_team_id": 10, "away_team_id": 20, "competition_id": 100, "environment": {}, "events": []}]},
                "2": {"results": [{"id": 2, "home_team_id": 30, "away_team_id": 40, "competition_id": 200, "environment": {}, "events": []}]}
            },
            "match_odds": {
                "1": {"results": {"provider1": {"some": "structure"}}},
                "2": {"results": {}}
            },
            "team_info": {
                "10": {"results": [{"name": "Team A", "logo": "home_logo_url", "country": "CountryA"}]},
                "20": {"results": [{"name": "Team B", "logo": "away_logo_url", "country": "CountryB"}]},
                "30": {"results": [{"name": "Team C", "logo": "home_logo_url2", "country": "CountryC"}]},
                "40": {"results": [{"name": "Team D", "logo": "away_logo_url2", "country": "CountryD"}]}
            },
            "competition_info": {
                "100": {"results": [{"name": "League1", "logo": "logo_url1", "country": "CountryComp1"}]},
                "200": {"results": [{"name": "League2", "logo": "logo_url2", "country": "CountryComp2"}]}
            },
            "countries": {
                "results": [
                    {"id": 10, "name": "CountryA"},
                    {"id": 20, "name": "CountryB"},
                    {"id": 100, "name": "CountryComp1"},
                    {"id": 200, "name": "CountryComp2"}
                ]
            }
        }

        with open(step2.STEP1_JSON, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2)

    def tearDown(self):
        # Restore original BASE_DIR and clean up temp directory
        step2.BASE_DIR = self.original_step2_base_dir
        step7.BASE_DIR = self.original_step7_base_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_run_step2_creates_valid_json(self):
        # Run step2 (merge + flatten)
        summaries = step2.run_step2()

        # Check that summaries is a list with 2 entries
        self.assertIsInstance(summaries, list)
        self.assertEqual(len(summaries), 2)

        # Check that step27.json exists
        self.assertTrue(step2.STEP2_OUTPUT.exists())

        # Load the JSON file and inspect its structure
        with open(step2.STEP2_OUTPUT, "r", encoding="utf-8") as f:
            data = json.load(f)

        # The top-level should have "history" and "step2_processing_summary"
        self.assertIn("history", data)
        self.assertIn("step2_processing_summary", data)

        # Last history entry should have "matches"
        last_batch = data["history"][-1]
        self.assertIn("matches", last_batch)
        matches_dict = last_batch["matches"]

        # There should be exactly 2 matches keyed by their match IDs
        self.assertEqual(set(matches_dict.keys()), {"1", "2"})

        # Inspect the summary of match "1"
        match1 = matches_dict["1"]
        self.assertEqual(match1["match_id"], 1)
        self.assertEqual(match1["status_id"], 4)
        self.assertEqual(match1["teams"]["home"]["name"], "Team A")
        self.assertEqual(match1["teams"]["away"]["name"], "Team B")
        self.assertEqual(match1["competition"]["name"], "League1")
        self.assertIn("environment", match1)
        self.assertIn("odds", match1)
        self.assertIn("events", match1)
        self.assertIn("fetched_at", match1)

        # Inspect the summary of match "2"
        match2 = matches_dict["2"]
        self.assertEqual(match2["match_id"], 2)
        self.assertEqual(match2["status_id"], 1)

    def test_run_step7_filters_and_prints(self):
        # First, run step2 to generate summaries
        summaries = step2.run_step2()

        # Capture stdout of run_step7
        f = io.StringIO()
        with redirect_stdout(f):
            step7.run_step7(summaries_list=summaries)
        output = f.getvalue()

        # Since match 1 is in-play (status_id=4) and match 2 is not (status_id=1),
        # we expect references to "Team A" and "Team B", but not "Team C" or "Team D"
        self.assertIn("Team A", output)
        self.assertIn("Team B", output)
        self.assertNotIn("Team C", output)
        self.assertNotIn("Team D", output)

        # Also, there should be a header that mentions "STEP 7: STATUS FILTER"
        self.assertIn("STEP 7: STATUS FILTER", output)

        # Footer should mention "END OF STATUS FILTER"
        self.assertIn("END OF STATUS FILTER", output)

    def test_step7_reads_from_json_file(self):
        # First, run step2 to generate the JSON file
        step2.run_step2()

        # Now run step7 without passing summaries (should read from file)
        f = io.StringIO()
        with redirect_stdout(f):
            step7.run_step7()  # No summaries_list parameter
        output = f.getvalue()

        # Should still filter and show only the in-play match
        self.assertIn("Team A", output)
        self.assertIn("Team B", output)
        self.assertNotIn("Team C", output)
        self.assertNotIn("Team D", output)
        self.assertIn("STEP 7: STATUS FILTER", output)

    def test_pipeline_integration(self):
        # Test the full pipeline: step2 -> step7
        print("\n=== Testing Full Pipeline ===")
        
        # Run step2
        print("Running Step 2...")
        summaries = step2.run_step2()
        self.assertEqual(len(summaries), 2)
        
        # Run step7 with the summaries
        print("Running Step 7...")
        f = io.StringIO()
        with redirect_stdout(f):
            step7.run_step7(summaries_list=summaries)
        output = f.getvalue()
        
        # Verify the filtering worked correctly
        lines = output.split('\n')
        team_a_lines = [line for line in lines if 'Team A' in line]
        team_c_lines = [line for line in lines if 'Team C' in line]
        
        self.assertTrue(len(team_a_lines) > 0, "Team A should appear in output")
        self.assertEqual(len(team_c_lines), 0, "Team C should not appear in output")
        
        print("Pipeline test completed successfully!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
