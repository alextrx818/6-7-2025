#!/usr/bin/env python3
"""
Mock data generator for testing the pipeline when API access is limited.
"""

import json
import os
from datetime import datetime, timedelta
import random

def generate_mock_football_data():
    """Generate realistic mock football data for testing."""
    
    # Mock teams
    teams = [
        {"id": 1, "name": "Manchester United", "short_name": "MUN", "country": "England"},
        {"id": 2, "name": "Liverpool", "short_name": "LIV", "country": "England"},
        {"id": 3, "name": "Arsenal", "short_name": "ARS", "country": "England"},
        {"id": 4, "name": "Chelsea", "short_name": "CHE", "country": "England"},
        {"id": 5, "name": "Manchester City", "short_name": "MCI", "country": "England"},
        {"id": 6, "name": "Tottenham", "short_name": "TOT", "country": "England"},
        {"id": 7, "name": "Real Madrid", "short_name": "RMA", "country": "Spain"},
        {"id": 8, "name": "Barcelona", "short_name": "BAR", "country": "Spain"},
        {"id": 9, "name": "Bayern Munich", "short_name": "BAY", "country": "Germany"},
        {"id": 10, "name": "PSG", "short_name": "PSG", "country": "France"}
    ]
    
    # Generate matches for the next few days
    matches = []
    base_date = datetime.now()
    
    for i in range(20):  # Generate 20 matches
        match_date = base_date + timedelta(hours=random.randint(1, 168))  # Next week
        home_team = random.choice(teams)
        away_team = random.choice([t for t in teams if t['id'] != home_team['id']])
        
        # Some matches are completed, some are upcoming
        is_completed = random.choice([True, False, False])  # 1/3 completed
        
        match = {
            "id": i + 1,
            "date": match_date.isoformat(),
            "timestamp": int(match_date.timestamp()),
            "home_team": home_team,
            "away_team": away_team,
            "competition": random.choice(["Premier League", "La Liga", "Champions League", "Europa League"]),
            "status": "completed" if is_completed else "scheduled",
            "venue": f"{home_team['name']} Stadium",
            "round": f"Round {random.randint(1, 38)}"
        }
        
        if is_completed:
            match["home_score"] = random.randint(0, 4)
            match["away_score"] = random.randint(0, 4)
            match["result"] = "home" if match["home_score"] > match["away_score"] else ("away" if match["away_score"] > match["home_score"] else "draw")
        else:
            match["home_score"] = None
            match["away_score"] = None
            match["result"] = None
            
        matches.append(match)
    
    return {
        "success": True,
        "data": matches,
        "meta": {
            "total": len(matches),
            "generated_at": datetime.now().isoformat(),
            "type": "mock_data"
        }
    }

def save_mock_data():
    """Save mock data to file for testing."""
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate and save mock data
    mock_data = generate_mock_football_data()
    
    filename = f"data/mock_football_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"✅ Mock data saved to: {filename}")
    print(f"📊 Generated {len(mock_data['data'])} matches")
    
    # Also save as the latest data for pipeline testing
    latest_filename = "data/latest_football_data.json"
    with open(latest_filename, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"📁 Latest data saved to: {latest_filename}")
    
    return filename

if __name__ == "__main__":
    save_mock_data()
