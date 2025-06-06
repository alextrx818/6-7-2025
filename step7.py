#!/usr/bin/env python3
"""
step7.py – Phase 2: Read JSON → filter & pretty-print
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
import pytz

# ---------------------------------------------------------------------------
# Constants and Path Configurations
# ---------------------------------------------------------------------------
TZ = pytz.timezone("America/New_York")
BASE_DIR = Path(__file__).resolve().parent
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
# Logging Setup
# ---------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    log_file = BASE_DIR / "step7_matches.log"
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('step7_matches')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


match_logger = setup_logger()


def log_and_print(message: str):
    print(message)
    match_logger.info(message)
    for handler in match_logger.handlers:
        handler.flush()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_daily_fetch_count() -> int:
    counter_file = BASE_DIR / "step6" / "daily_fetch_counter.txt"
    try:
        if counter_file.exists():
            content = counter_file.read_text().strip()
            return int(content) if content else 1
        return 1
    except:
        return 1


def infer_country_from_teams(match_data):
    home_team = match_data.get("teams", {}).get("home", {}).get("name", "").lower()
    away_team = match_data.get("teams", {}).get("away", {}).get("name", "").lower()
    competition = match_data.get("competition", {})
    if isinstance(competition, dict):
        competition = competition.get("name", "").lower()
    else:
        competition = str(competition).lower()
    
    country_indicators = {
        "australia": ["australia", "aussie", "socceroos", "matildas"],
        "argentina": ["argentina", "boca", "river plate", "racing club"],
        "brazil": ["brazil", "sao paulo", "flamengo", "corinthians", "palmeiras"],
        "england": ["england", "manchester", "liverpool", "chelsea", "arsenal", "tottenham"],
        "spain": ["spain", "real madrid", "barcelona", "atletico", "sevilla", "valencia"],
        "germany": ["germany", "bayern", "borussia", "schalke", "hamburg"],
        "france": ["france", "psg", "marseille", "lyon", "monaco", "saint-etienne"],
        "italy": ["italy", "juventus", "inter", "milan", "roma", "napoli", "lazio"],
        "netherlands": ["netherlands", "ajax", "psv", "feyenoord"],
        "portugal": ["portugal", "porto", "benfica", "sporting"],
        "mexico": ["mexico", "america", "guadalajara", "cruz azul", "pumas"],
        "usa": ["usa", "united states", "la galaxy", "seattle sounders", "new york"],
        "south korea": ["korea", "seoul", "busan", "daegu"],
        "japan": ["japan", "tokyo", "osaka", "yokohama", "kashima"],
        "china": ["china", "beijing", "shanghai", "guangzhou"],
        "russia": ["russia", "moscow", "spartak", "cska", "dynamo", "zenit"],
        "norway": ["norway", "oslo", "bergen"],
        "czech republic": ["czech", "praha", "prague", "brno"],
        "austria": ["austria", "vienna", "salzburg"]
    }
    
    if "international" in competition and "friendly" in competition:
        home_countries, away_countries = [], []
        for country, indicators in country_indicators.items():
            for indicator in indicators:
                if indicator in home_team:
                    home_countries.append(country)
                if indicator in away_team:
                    away_countries.append(country)
        if home_countries and away_countries and home_countries[0] != away_countries[0]:
            return "International"
        if home_countries and not away_countries:
            return home_countries[0].title()
        if away_countries and not home_countries:
            return away_countries[0].title()
        if home_countries and away_countries and home_countries[0] == away_countries[0]:
            return home_countries[0].title()

    team_text = f"{home_team} {away_team}"
    for country, indicators in country_indicators.items():
        for indicator in indicators:
            if indicator in team_text:
                return country.title()
    return "Unknown"


# ---------------------------------------------------------------------------
# Sorting and Filtering Functions
# ---------------------------------------------------------------------------
def sort_matches_by_competition_and_time(matches: dict) -> dict:
    competition_groups = {}
    for match_id, match_data in matches.items():
        comp = match_data.get("competition", "Unknown Competition")
        if isinstance(comp, dict):
            comp = comp.get("name", "Unknown Competition")
        elif comp is None:
            comp = "Unknown Competition"

        country = match_data.get("competition", {}).get("country", "Unknown") if isinstance(match_data.get("competition"), dict) else "Unknown"
        if country in [None, "None", "Unknown"]:
            country = infer_country_from_teams(match_data)

        if comp not in competition_groups:
            competition_groups[comp] = {"country": country, "matches": []}
        competition_groups[comp]["matches"].append((match_id, match_data))

    status_order = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    for comp in competition_groups:
        competition_groups[comp]["matches"].sort(
            key=lambda item: (
                status_order.get(item[1].get("status_id", 99), 99),
                item[1].get("match_id", "")
            )
        )

    sorted_comps = sorted(
        competition_groups.items(),
        key=lambda item: (item[1]["country"] or "Unknown", item[0])
    )
    return {comp: data["matches"] for comp, data in sorted_comps}


# ---------------------------------------------------------------------------
# Display Functions
# ---------------------------------------------------------------------------
def write_main_header(fetch_count: int, total: int, generated_at: str, pipeline_time=None):
    header = (
        f"\n{'='*80}\n"
        f"🔥 STEP 7: STATUS FILTER (2–7)\n"
        f"{'='*80}\n"
        f"Filter Time: {get_eastern_time()}\n"
        f"Data Generated: {generated_at}\n"
        f"Pipeline Time: {pipeline_time or 'Not provided'}\n"
        f"Daily Fetch: #{fetch_count}\n"
        f"Statuses Filtered: {sorted(STATUS_FILTER)}\n"
        f"Included Matches Count: {total}\n"
        f"{'='*80}\n"
    )
    print(header)  # Also print to stdout
    match_logger.info(header)
    for h in match_logger.handlers:
        h.flush()


def write_main_footer(fetch_count: int, total: int, generated_at: str, pipeline_time=None, matches=None):
    footer = (
        f"\n{'='*80}\n"
        f"END OF STATUS FILTER – STEP 7\n"
        f"{'='*80}\n"
        f"Summary Time: {get_eastern_time()}\n"
        f"Total Matches (statuses 2–7): {total}\n"
        f"Daily Fetch: #{fetch_count}\n"
        f"{'='*80}\n"
    )
    print(footer)  # Also print to stdout
    match_logger.info(footer)
    if matches and total > 0:
        status_counts = {}
        for match_data in matches.values():
            status_id = match_data.get("status_id")
            if status_id in STATUS_FILTER:
                status_counts[status_id] = status_counts.get(status_id, 0) + 1
        summary_footer = (
            f"\nSTEP 7 – STATUS SUMMARY\n"
            f"{'='*60}\n"
        )
        for status_id in sorted(status_counts.keys()):
            count = status_counts[status_id]
            desc = get_status_description(status_id)
            summary_footer += f"{desc} (ID: {status_id}): {count}\n"
        summary_footer += f"Total: {total}\n" f"{'='*60}\n"
        print(summary_footer)  # Also print to stdout
        match_logger.info(summary_footer)
    for h in match_logger.handlers:
        h.flush()


def write_competition_group_header(competition: str, country: str, match_count: int):
    comp_line = f"🏆 {competition.upper()}"
    info_line = f"📍 {country} | 📊 {match_count} Matches"
    header = (
        f"\n{'='*100}\n"
        f"{'='*100}\n"
        f"{comp_line.center(100)}\n"
        f"{info_line.center(100)}\n"
        f"{'='*100}\n"
        f"{'='*100}\n"
    )
    log_and_print(header)


def format_american_odds(odds_value):
    if not odds_value or odds_value == 0:
        return "N/A"
    try:
        if isinstance(odds_value, str):
            if odds_value.startswith(("+", "-")):
                return odds_value
            try:
                num_val = float(odds_value)
                return f"+{int(num_val)}" if num_val > 0 else str(int(num_val))
            except ValueError:
                return odds_value
        odds_num = float(odds_value)
        return f"+{int(odds_num)}" if odds_num > 0 else str(int(odds_num))
    except:
        return str(odds_value) if odds_value else "N/A"


def format_betting_odds(match_data: dict) -> str:
    odds_lines = []
    odds = match_data.get("odds", {})
    
    full_time = odds.get("full_time_result", {})
    if full_time and isinstance(full_time, dict):
        home_ml = format_american_odds(full_time.get("home"))
        draw_ml = format_american_odds(full_time.get("draw"))
        away_ml = format_american_odds(full_time.get("away"))
        time_stamp = full_time.get("match_time", "0")
        odds_lines.append(
            f"│ ML:     │ Home: {home_ml:>6} │ Draw: {draw_ml:>6} │ Away: {away_ml:>7} │ (@{time_stamp}')"
        )
    
    spread = odds.get("spread", {})
    if spread and isinstance(spread, dict):
        home_spread = format_american_odds(spread.get("home"))
        handicap = spread.get("handicap", 0)
        away_spread = format_american_odds(spread.get("away"))
        time_stamp = spread.get("match_time", "0")
        odds_lines.append(
            f"│ Spread: │ Home: {home_spread:>6} │ Hcap: {handicap:>6} │ Away: {away_spread:>7} │ (@{time_stamp}')"
        )
    
    over_under = odds.get("over_under", {})
    if over_under and isinstance(over_under, dict):
        for line_value, line_data in over_under.items():
            if isinstance(line_data, dict):
                over_odds = format_american_odds(line_data.get("over"))
                line_num = line_data.get("line", line_value)
                under_odds = format_american_odds(line_data.get("under"))
                time_stamp = line_data.get("match_time", "0")
                odds_lines.append(
                    f"│ O/U:    │ Over: {over_odds:>6} │ Line: {line_num:>6} │ Under: {under_odds:>6} │ (@{time_stamp}')"
                )
                break
    
    if not odds_lines:
        return "No betting odds available"
    return "\n".join(odds_lines)


def format_environment_data(match_data: dict) -> str:
    environment = match_data.get("environment", {})
    if not environment:
        return "No environment data available"
    
    weather = environment.get("weather_description", "Unknown")
    temp_c = environment.get("temperature_value", 0)
    temp_unit = environment.get("temperature_unit") or "°C"
    wind_desc = environment.get("wind_description", "Unknown")
    wind_value = environment.get("wind_value", 0)
    wind_unit = environment.get("wind_unit") or "m/s"

    # Convert temperature to Fahrenheit if °C
    if temp_unit == "°C" and temp_c:
        temp_f = (temp_c * 9/5) + 32
        temp_display = f"{temp_f:.1f}°F"
    else:
        temp_display = f"{temp_c}°{temp_unit.replace('°', '') if temp_unit else 'C'}"

    if wind_unit == "m/s" and wind_value:
        wind_mph = wind_value * 2.237
        wind_display = f"{wind_desc}, {wind_mph:.1f} mph"
    else:
        wind_display = f"{wind_desc}, {wind_value} {wind_unit}"

    return f"Weather: {weather}\nTemperature: {temp_display}\nWind: {wind_display}"


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------
def run_step7(summaries_list: list = None):
    """
    Phase 2:
      1. If summaries_list is None, load step2.json from disk.
      2. Filter out anything with status_id not in STATUS_FILTER.
      3. Sort by competition & time.
      4. Pretty-print each competition group to the console + log.
    """
    # 1) Load from disk if needed
    if summaries_list is None:
        if not STEP2_OUTPUT.exists():
            print(f"Error: Cannot find {STEP2_OUTPUT.name} for Step 7.")
            return
        with open(STEP2_OUTPUT, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        # The top‐level has "history" plus other keys; take the last entry's "matches"
        history = all_data.get("history", [])
        if not history:
            print("Error: No history in step2.json.")
            return
        last_batch = history[-1]
        raw_matches = last_batch.get("matches", {})
        generated_at = last_batch.get("timestamp", get_eastern_time())
    else:
        # Convert list of summaries into a dict mapping match_id → summary
        raw_matches = {str(s["match_id"]): s for s in summaries_list}
        generated_at = get_eastern_time()

    # 2) Filter status_id ∈ STATUS_FILTER
    filtered = {
        mid: mdata
        for mid, mdata in raw_matches.items()
        if mdata.get("status_id") in STATUS_FILTER
    }
    total = len(filtered)
    fetch_count = get_daily_fetch_count()

    # 3) Sort by competition & time
    sorted_groups = sort_matches_by_competition_and_time(filtered)

    # 4) Write header
    write_main_header(fetch_count, total, generated_at, pipeline_time=None)

    # 5) Loop through each competition group
    for competition, matches_list in sorted_groups.items():
        if not matches_list:
            continue
            
        # Get country from first match
        first_match = matches_list[0][1]
        country = first_match.get("competition", {}).get("country", "Unknown") if isinstance(first_match.get("competition"), dict) else "Unknown"
        if country in [None, "None", "Unknown"]:
            country = infer_country_from_teams(first_match)
            
        write_competition_group_header(competition, country, len(matches_list))

        for match_id, match_data in matches_list:
            # Build a "pretty block" for each match
            teams = match_data.get("teams", {})
            home_team = teams.get("home", {})
            away_team = teams.get("away", {})
            
            home = home_team.get("name", "Unknown")
            away = away_team.get("name", "Unknown")
            
            status = match_data.get("status", {})
            status_desc = status.get("description", "Unknown")
            time_str = status.get("match_time", "0")
            
            home_score = home_team.get("score", {}).get("current", 0)
            away_score = away_team.get("score", {}).get("current", 0)

            # Basic line
            line = f"• [{status_desc} | {time_str}'] {home} {home_score} – {away_score} {away}"
            log_and_print(line)

            # Betting odds block
            odds_text = format_betting_odds(match_data)
            log_and_print(odds_text)

            # Environment block
            env_text = format_environment_data(match_data)
            log_and_print(env_text)

            # Separator between matches
            log_and_print("-"*80)

    # 6) Write footer
    write_main_footer(fetch_count, total, generated_at, pipeline_time=None, matches=filtered)
    print("Step 7: Completed.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*80)
    print("STEP 7 - STATUS FILTER & DISPLAY STARTED")
    print("="*80)
    
    run_step7()
    
    print("="*80)
    print("STEP 7 - STATUS FILTER & DISPLAY COMPLETED")
    print("="*80)
