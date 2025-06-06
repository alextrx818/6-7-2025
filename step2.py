#!/usr/bin/env python3
"""
step2.py – Phase 1: Merge + Extract + Flatten → write JSON (step2.json)

IMPORTANT: FOOTER LOGIC REFERENCE FOR ALL OUTPUTS
===============================================
This file contains the EXACT footer/summary logic that MUST be replicated 
in step1.log and step1.json. The footer format is IDENTICAL across ALL outputs.

REQUIRED FOOTER FORMAT FOR step1.log, step1.json, AND step2.json:
{
  "footer": "================================================================================",
  "completion_status": "COMPLETE PIPELINE (Step 1→7) – FINISHED SUCCESSFULLY – MM/DD/YYYY HH:MM:SS PM EDT",
  "daily_match_number": XX,
  "total_matches_fetched_all_statuses": "XX matches (ALL status IDs from Step 1 live endpoint)",
  "in_play_matches": "XX (status IDs 2–7)",
  "other_status_matches": "XX (status IDs 0,1,8,9,10,11,12,13)",
  "step1_execution_time": "XX.XX seconds" (for step1 outputs) OR "stepX_execution_time": "XX.XX seconds",
  "total_pipeline_time": "XXX.XX seconds",
  "detailed_data_fetch": {
    "unique_teams_fetched": XX,
    "unique_competitions_fetched": XX,
    "match_details_fetched": XX,
    "match_odds_fetched": XX
  },
  "raw_api_status_breakdown": [
    "Status Name (ID: X): XX matches", ...
  ],
  "footer_end": "================================================================================"
}

CRITICAL: These EXACT summary values with EXACT formatting must appear in:
- step1.log (as log entries)
- step1.json (as JSON footer object)  
- step2.json (as JSON footer object)

The match counts, status breakdowns, and timing must be identical across all files.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
import pytz

# ---------------------------------------------------------------------------
# Constants and Path Configurations
# ---------------------------------------------------------------------------
TZ = pytz.timezone("America/New_York")
BASE_DIR = Path(__file__).resolve().parent
STEP1_JSON = BASE_DIR / "step1.json"
STEP2_OUTPUT = BASE_DIR / "step2.json"
STATUS_FILTER = {2, 3, 4, 5, 6, 7}


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def get_eastern_time() -> str:
    now = datetime.now(TZ)
    return now.strftime("%m/%d/%Y %I:%M:%S %p %Z")


def get_status_description(status_id: int) -> str:
    status_map = {
        0: "Abnormal (suggest hiding)",
        1: "Not started",
        2: "First half",
        3: "Half-time",
        4: "Second half",
        5: "Overtime",
        6: "Overtime (deprecated)",
        7: "Penalty Shoot-out",
        8: "End",
        9: "Delay",
        10: "Interrupt",
        11: "Cut in half",
        12: "Cancel",
        13: "To be determined"
    }
    return status_map.get(status_id, f"Unknown ({status_id})")


# ---------------------------------------------------------------------------
# Processing Metrics Class
# ---------------------------------------------------------------------------
class ProcessingMetrics:
    """Track metrics for Step 2 processing."""
    def __init__(self):
        self.start_time = time.time()
        self.api_response_successful = False
        self.api_response_time = 0
        self.total_matches_received = 0
        self.api_response_code = None
        self.matches_with_status = 0
        self.matches_without_status = 0
        self.status_breakdown = {}
        self.unique_teams_processed = set()
        self.unique_competitions_processed = set()
        self.match_details_processed = 0
        self.match_odds_processed = 0
        self.processing_errors = []
        self.in_play_matches = 0

    def calculate_totals(self):
        self.total_execution_time = time.time() - self.start_time
        self.unique_teams_count = len(self.unique_teams_processed)
        self.unique_competitions_count = len(self.unique_competitions_processed)


metrics = ProcessingMetrics()


# ---------------------------------------------------------------------------
# Data Extraction Functions
# ---------------------------------------------------------------------------
def extract_summary_fields(match: dict) -> dict:
    """Return a compact summary structure for a single match."""
    home_live = home_ht = away_live = away_ht = 0
    sd = match.get("score", [])
    if isinstance(sd, list) and len(sd) > 3:
        hs, as_ = sd[2], sd[3]
        if isinstance(hs, list) and len(hs) > 1:
            home_live, home_ht = hs[0], hs[1]
        if isinstance(as_, list) and len(as_) > 1:
            away_live, away_ht = as_[0], as_[1]

    home_scores = match.get("home_scores", [])
    away_scores = match.get("away_scores", [])
    if home_scores and home_live == 0:
        home_live = home_scores[0] if isinstance(home_scores, list) and home_scores else 0
    if away_scores and away_live == 0:
        away_live = away_scores[0] if isinstance(away_scores, list) and away_scores else 0

    return {
        "match_id": match.get("match_id") or match.get("id"),
        "status_id": match.get("status_id"),
        "status": {
            "id": match.get("status_id"),
            "description": match.get("status", ""),
            "match_time": match.get("match_time", 0),
        },
        "teams": {
            "home": {
                "name": match.get("home_team", "Unknown"),
                "score": {"current": home_live, "halftime": home_ht, "detailed": home_scores},
                "position": match.get("home_position"),
                "country": match.get("home_country"),
                "logo_url": match.get("home_logo"),
            },
            "away": {
                "name": match.get("away_team", "Unknown"),
                "score": {"current": away_live, "halftime": away_ht, "detailed": away_scores},
                "position": match.get("away_position"),
                "country": match.get("away_country"),
                "logo_url": match.get("away_logo"),
            },
        },
        "competition": {
            "name": match.get("competition", "Unknown"),
            "id": match.get("competition_id"),
            "country": match.get("country"),
            "logo_url": match.get("competition_logo"),
        },
        "round": match.get("round", {}),
        "venue": match.get("venue_id"),
        "referee": match.get("referee_id"),
        "neutral": match.get("neutral") == 1,
        "coverage": match.get("coverage", {}),
        "start_time": match.get("scheduled"),
        "odds": extract_odds(match),
        "environment": extract_environment(match),
        "events": extract_events(match),
        "fetched_at": get_eastern_time(),
    }


def extract_odds(match: dict) -> dict:
    raw_odds = match.get("odds", {}) or {}
    data = {
        "full_time_result": {},
        "both_teams_to_score": {},
        "over_under": {},
        "spread": {},
        "raw": raw_odds
    }

    def _safe_minute(v):
        if v is None:
            return None
        m = re.match(r"(\d+)", str(v))
        return int(m.group(1)) if m else None

    def filter_by_time(entries):
        pts = [(_safe_minute(ent[1]), ent) for ent in entries if isinstance(ent, (list, tuple)) and len(ent) > 1]
        pts = [(m, e) for m, e in pts if m is not None]
        in_window = [e for m, e in pts if 3 <= m <= 6]
        if in_window:
            return in_window
        under_ten = [(m, e) for m, e in pts if m < 10]
        return [] if not under_ten else [min(under_ten, key=lambda t: abs(t[0] - 4.5))[1]]

    for key, idxs in [("eu", (2,3,4)), ("asia", (2,3,4)), ("bs", (2,3,4))]:
        entry = (filter_by_time(raw_odds.get(key, [])) or [None])[0]
        if entry and len(entry) >= max(idxs) + 1:
            if key == "eu":
                data["full_time_result"] = {
                    "home": entry[2], "draw": entry[3], "away": entry[4],
                    "timestamp": entry[0], "match_time": entry[1]
                }
            elif key == "asia":
                data["spread"] = {
                    "handicap": entry[3], "home": entry[2], "away": entry[4],
                    "timestamp": entry[0], "match_time": entry[1]
                }
            else:
                line = entry[3]
                data["over_under"][str(line)] = {
                    "line": line, "over": entry[2], "under": entry[4],
                    "timestamp": entry[0], "match_time": entry[1]
                }
                data["primary_over_under"] = data["over_under"][str(line)]

    for m in match.get("betting", {}).get("markets", []):
        if m.get("name") == "Both Teams to Score":
            for sel in m.get("selections", []):
                nm = sel.get("name", "").lower()
                if nm in ("yes", "no"):
                    data["both_teams_to_score"][nm] = sel.get("odds")
    return data


def extract_environment(match: dict) -> dict:
    env = match.get("environment", {}) or {}
    parsed = {"raw": env}
    wc = env.get("weather")
    parsed["weather"] = int(wc) if isinstance(wc, str) and wc.isdigit() else wc
    desc = {
        1: "Sunny", 2: "Partly Cloudy", 3: "Cloudy", 4: "Overcast",
        5: "Foggy", 6: "Light Rain", 7: "Rain", 8: "Heavy Rain",
        9: "Snow", 10: "Thunder"
    }
    parsed["weather_description"] = desc.get(parsed["weather"], "Unknown")

    for key in ("temperature", "wind", "pressure", "humidity"):
        val = env.get(key)
        parsed[key] = val
        m = re.match(r"([\d.-]+)\s*([^\d]*)", str(val))
        num, unit = (float(m.group(1)), m.group(2).strip()) if m else (None, None)
        parsed[f"{key}_value"] = num
        parsed[f"{key}_unit"] = unit

    wv = parsed.get("wind_value") or 0
    mph = wv * 2.237 if "m/s" in str(env.get("wind", "")).lower() else wv
    descs = [
        (1, "Calm"), (4, "Light Air"), (8, "Light Breeze"), (13, "Gentle Breeze"),
        (19, "Moderate Breeze"), (25, "Fresh Breeze"), (32, "Strong Breeze"),
        (39, "Near Gale"), (47, "Gale"), (55, "Strong Gale"), (64, "Storm"), (73, "Violent Storm")
    ]
    parsed["wind_description"] = next((label for lim, label in descs if mph < lim), "Hurricane")
    return parsed


def extract_events(match: dict) -> list:
    return [
        {"type": ev.get("type"), "time": ev.get("time"), "team": ev.get("team"),
         "player": ev.get("player"), "detail": ev.get("detail")}
        for ev in match.get("events", [])
        if ev.get("type") in {"goal", "yellowcard", "redcard", "penalty", "substitution"}
    ]


def first_result(mapping: dict, key):
    wrap = mapping.get(str(key)) if key is not None else None
    if isinstance(wrap, dict):
        res = wrap.get("results") or wrap.get("result") or []
        return res[0] if isinstance(res, list) and res else {}
    return {}


# ---------------------------------------------------------------------------
# Core Processing Functions
# ---------------------------------------------------------------------------
def merge_and_summarize(live: dict, payload: dict) -> dict:
    """Merge live match data with detailed payload and create summary."""
    global metrics
    mid = live.get("id") or live.get("match_id")
    mid_str = str(mid) if mid is not None else None
    if mid:
        metrics.match_details_processed += 1

    dm = payload.get("match_details", {})
    om = payload.get("match_odds", {})
    tm = payload.get("team_info", {})
    cm = payload.get("competition_info", {})
    cw = payload.get("countries", {})
    cl = cw.get("results") or cw.get("result") or []
    countries = {c.get("id"): c.get("name") for c in cl if isinstance(c, dict)}

    detail = first_result(dm, mid)
    odds_wrap = om.get(mid_str, {}) or om.get(mid, {})
    odds_struct = {}

    # Handle different odds data structures more robustly
    if odds_wrap:
        results = odds_wrap.get("results", {})
        if isinstance(results, dict):
            for provider_data in results.values():
                if isinstance(provider_data, dict):
                    odds_struct.update(provider_data)
        elif isinstance(results, list):
            for entry in results:
                if isinstance(entry, dict):
                    odds_struct.update(entry)

    home = first_result(tm, live.get("home_team_id") or detail.get("home_team_id"))
    away = first_result(tm, live.get("away_team_id") or detail.get("away_team_id"))
    comp = first_result(cm, live.get("competition_id") or detail.get("competition_id"))

    merged = {
        **live,
        **detail,
        "odds": odds_struct,
        "environment": detail.get("environment", live.get("environment", {})),
        "events": detail.get("events", live.get("events", [])),
        "home_team": home.get("name") or live.get("home_name"),
        "home_logo": home.get("logo"),
        "home_country": home.get("country") or countries.get(home.get("country_id")),
        "away_team": away.get("name") or live.get("away_name"),
        "away_logo": away.get("logo"),
        "away_country": away.get("country") or countries.get(away.get("country_id")),
        "competition": comp.get("name") or live.get("competition_name"),
        "competition_logo": comp.get("logo"),
        "country": comp.get("country") or countries.get(comp.get("country_id")),
        "odds_raw": odds_wrap
    }

    home_team_id = live.get("home_team_id") or detail.get("home_team_id")
    away_team_id = live.get("away_team_id") or detail.get("away_team_id")
    comp_id = live.get("competition_id") or detail.get("competition_id")

    if home_team_id:
        metrics.unique_teams_processed.add(home_team_id)
    if away_team_id:
        metrics.unique_teams_processed.add(away_team_id)
    if comp_id:
        metrics.unique_competitions_processed.add(comp_id)

    if odds_struct or odds_wrap:
        metrics.match_odds_processed += 1

    return extract_summary_fields(merged)


def save_match_summaries(summaries: list, output_file: Path = STEP2_OUTPUT, pipeline_start_time: float = None) -> bool:
    """
    Save match summaries (including metrics) into step2.json.
    Keeps up to 100 history entries.
    """
    global metrics
    grouped = {str(s.get("match_id")): s for s in summaries if s.get("match_id")}
    batch = {
        "timestamp": datetime.now(TZ).isoformat(),
        "total_matches": len(grouped),
        "matches": grouped
    }

    step2_counter_file = BASE_DIR / "step2_daily_counter.json"
    today_str = datetime.now(TZ).strftime("%Y-%m-%d")
    counter_data = {"date": "", "count": 0}
    if step2_counter_file.exists():
        try:
            with open(step2_counter_file, 'r') as f:
                counter_data = json.load(f)
        except:
            pass

    if counter_data.get("date") != today_str:
        counter_data = {"date": today_str, "count": 0}
    counter_data["count"] += 1

    try:
        with open(step2_counter_file, 'w') as f:
            json.dump(counter_data, f)
    except:
        pass

    # === Build or rotate history in step2.json ===
    try:
        data = {"history": []}
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            if isinstance(loaded_data, dict) and loaded_data.get("history"):
                data = loaded_data
            else:
                data = {"history": [loaded_data]}

        MAX_HISTORY_ENTRIES = 100
        if len(data["history"]) >= MAX_HISTORY_ENTRIES:
            data["history"] = data["history"][-MAX_HISTORY_ENTRIES:]

        data["history"].append(batch)
        data.update({
            "last_updated": batch["timestamp"],
            "total_entries": len(data["history"]),
            "latest_match_count": batch["total_matches"],
            "ny_timestamp": get_eastern_time(),
        })

        # Calculate total pipeline time if provided
        total_pipeline_time = 0
        if pipeline_start_time is not None:
            total_pipeline_time = time.time() - pipeline_start_time

        # Add processing summary footer
        data["step2_processing_summary"] = {
            "header": "="*80,
            "title": "STEP 2 – PROCESSING SUMMARY",
            "divider": "="*80,
            "api_status": (
                "✓ Live matches API data received successfully"
                if metrics.api_response_successful
                else "✗ Live matches API data not received successfully"
            ),
            "response_time": (
                f"{metrics.api_response_time:.2f} seconds"
                if metrics.api_response_time > 0
                else "N/A"
            ),
            "total_matches_returned": metrics.total_matches_received,
            "api_response_code": metrics.api_response_code,
            "matches_with_status_info": f"{metrics.matches_with_status}/{metrics.total_matches_received}",
            "comprehensive_match_summary": {
                "total_matches_fetched": metrics.total_matches_received,
                "matches_with_status": metrics.matches_with_status,
                "in_play_matches": f"{metrics.in_play_matches} (status IDs 2–7)",
                "other_matches": f"{metrics.total_matches_received - metrics.in_play_matches} (all other statuses)",
                "status_coverage": f"{metrics.matches_with_status}/{metrics.total_matches_received} matches have status info"
            },
            "raw_api_status_breakdown": [],
            "detailed_data_fetch": {
                "processing_time": f"{metrics.total_execution_time:.2f} seconds",
                "unique_teams_fetched": metrics.unique_teams_count,
                "unique_competitions_fetched": metrics.unique_competitions_count,
                "match_details_fetched": metrics.match_details_processed,
                "match_odds_fetched": metrics.match_odds_processed
            },
            "pipeline_timing": {
                "step2_processing_time": f"{metrics.total_execution_time:.2f} seconds",
                "total_pipeline_time": f"{total_pipeline_time:.2f} seconds" if total_pipeline_time > 0 else "N/A"
            },
            "footer": "="*80,
            "completion_status": f"STEP 2 – FETCH COMPLETED SUCCESSFULLY – {get_eastern_time()}",
            "daily_match_number": counter_data["count"],
            "total_matches_fetched_all_statuses": f"{metrics.total_matches_received} matches (ALL status IDs from Step 1 live endpoint)",
            "in_play_matches": f"{metrics.in_play_matches} (status IDs 2–7)",
            "other_status_matches": f"{metrics.total_matches_received - metrics.in_play_matches} (status IDs 0,1,8,9,10,11,12,13)",
            "step2_execution_time": f"{metrics.total_execution_time:.2f} seconds",
            "total_pipeline_time": f"{total_pipeline_time:.2f} seconds" if total_pipeline_time > 0 else "N/A",
            "footer_end": "="*80
        }

        # Fill in status breakdown
        status_desc_map = {
            0: "Abnormal (suggest hiding)",
            1: "Not started",
            2: "First half",
            3: "Half-time",
            4: "Second half",
            5: "Overtime",
            6: "Overtime (deprecated)",
            7: "Penalty Shoot-out",
            8: "End",
            9: "Delay",
            10: "Interrupt",
            11: "Cut in half",
            12: "Cancel",
            13: "To be determined"
        }

        for status_id in sorted(metrics.status_breakdown.keys()):
            count = metrics.status_breakdown[status_id]["count"]
            desc = status_desc_map.get(status_id, f"Unknown Status")
            data["step2_processing_summary"]["raw_api_status_breakdown"].append(
                f"{desc} (ID: {status_id}): {count} matches"
            )

        # Save step2.json
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error saving {output_file.name}: {e}")
        return False


async def extract_merge_summarize(data: dict, pipeline_start_time: float = None) -> list:
    """
    This is the core of "run_step2()":
      • data = contents of step1.json
      • merges live & detail, tracks metrics, returns a list of summaries
    """
    global metrics
    metrics = ProcessingMetrics()  # reset

    print("Step 2: Starting extract_merge_summarize...")
    live_matches_data = data.get("live_matches", {})
    if live_matches_data:
        metrics.api_response_successful = (live_matches_data.get("code") == 0)
        metrics.api_response_code = live_matches_data.get("code")
        metrics.api_response_time = live_matches_data.get("response_time", 0)

    matches = (live_matches_data.get("results") or live_matches_data.get("matches") or [])
    metrics.total_matches_received = len(matches)

    for m in matches:
        status_id = m.get("status_id")
        if status_id is not None:
            metrics.matches_with_status += 1
            desc = get_status_description(status_id)
            if status_id not in metrics.status_breakdown:
                metrics.status_breakdown[status_id] = {"description": desc, "count": 0}
            metrics.status_breakdown[status_id]["count"] += 1
            if status_id in STATUS_FILTER:
                metrics.in_play_matches += 1
        else:
            metrics.matches_without_status += 1

    print(f"Step 2: Found {len(matches)} matches to process")
    summaries = []
    for m in matches:
        try:
            summary = merge_and_summarize(m, data)
            summaries.append(summary)
        except Exception as e:
            metrics.processing_errors.append(f"Error processing match {m.get('id')}: {str(e)}")

    print(f"Step 2: Created {len(summaries)} summaries")
    metrics.calculate_totals()

    if summaries:
        success = save_match_summaries(summaries, output_file=STEP2_OUTPUT, pipeline_start_time=pipeline_start_time)
        if success:
            print(f"Step 2: Output written to {STEP2_OUTPUT.name}")
        else:
            print(f"Step 2: Failed to write {STEP2_OUTPUT.name}")
    else:
        print("Step 2: No summaries generated")

    print("Step 2: Processing completed")
    return summaries


def run_step2(pipeline_start_time: float = None, match_number: int = None):
    """
    Load step1.json, pass it into extract_merge_summarize, and block until done.
    """
    if not STEP1_JSON.exists():
        print(f"Error: {STEP1_JSON.name} not found.")
        return []

    with open(STEP1_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Use asyncio.run instead of get_event_loop
    try:
        summaries = asyncio.run(extract_merge_summarize(raw, pipeline_start_time))
    except RuntimeError:
        # Fallback for environments where asyncio.run() isn't available
        summaries = asyncio.get_event_loop().run_until_complete(extract_merge_summarize(raw, pipeline_start_time))
    return summaries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("STEP 2 - DATA PROCESSING STARTED")
    print("="*80)
    
    summaries = run_step2()
    
    print("="*80)
    print("STEP 2 - DATA PROCESSING COMPLETED")
    print(f"Total summaries generated: {len(summaries)}")
    print("="*80)
