import asyncio
import json
import os
import sys
from datetime import datetime

# Import your modules
from step1 import step1_main
from step27 import extract_merge_summarize, sort_matches_by_competition_and_time

def test_step1_to_step27_pipeline():
    """Complete integration test following the framework commands"""
    
    # Command 1: Load and Execute Step 1
    print("1. Running Step 1...")
    data1 = step1_main()
    with open('step1.json', 'w') as f:
        json.dump(data1, f, indent=2)
    
    # Command 2: Validate structure
    print("2. Validating step1.json structure...")
    required_keys = ['timestamp', 'live_matches', 'match_details', 
                     'match_odds', 'team_info', 'competition_info', 'countries']
    for key in required_keys:
        assert key in data1, f"Missing required key: {key}"
    
    # Command 3: Verify live matches
    print("3. Verifying live matches array...")
    assert 'results' in data1['live_matches']
    matches = data1['live_matches']['results']
    assert isinstance(matches, list)
    print(f"   Found {len(matches)} live matches")
    if matches:
        print(f"   First match sample: {matches[0].get('home_team')} vs {matches[0].get('away_team')}")
    
    # Command 4: Ensure ID mappings
    print("4. Checking ID mappings...")
    missing_details = []
    missing_odds = []
    for match in matches:
        mid = str(match['id'])
        if mid not in data1['match_details']:
            missing_details.append(mid)
        if mid not in data1['match_odds']:
            missing_odds.append(mid)
    
    if missing_details:
        print(f"   WARNING: {len(missing_details)} matches missing details")
    if missing_odds:
        print(f"   WARNING: {len(missing_odds)} matches missing odds")
    
    # Command 5-6: Inspect detailed structures
    if matches:
        mid = str(matches[0]['id'])
        print(f"5. Inspecting detail for match {mid}...")
        if mid in data1['match_details']:
            detail = data1['match_details'][mid]['results'][0]
            print(f"   Detail keys: {list(detail.keys())[:5]}...")
        
        print(f"6. Inspecting odds for match {mid}...")
        if mid in data1['match_odds']:
            odds = data1['match_odds'][mid]['results']
            print(f"   Odds providers: {list(odds.keys())[:3]}...")
    
    # Command 7: Run Step 2.7
    print("7. Running Step 2.7 extract_merge_summarize...")
    loop = asyncio.get_event_loop()
    summaries = loop.run_until_complete(extract_merge_summarize(data1))
    print(f"   Generated {len(summaries)} summaries")
    assert len(summaries) == len(matches), "Summary count mismatch"
    
    # Command 8: Verify summary structure
    if summaries:
        print("8. Verifying summary structure...")
        required_summary_keys = ['match_id', 'status', 'teams', 'competition', 
                                'odds', 'environment', 'events']
        for key in required_summary_keys:
            assert key in summaries[0], f"Missing summary key: {key}"
        
        # Check nested structure
        assert 'home' in summaries[0]['teams']
        assert 'score' in summaries[0]['teams']['home']
        assert 'current' in summaries[0]['teams']['home']['score']
    
    # Command 9: Save step27.json
    print("9. Saving step27.json...")
    data2 = {
        'history': {},
        'last_updated': datetime.now().isoformat(),
        'history_length': len(data2.get('history', {})),  # Number of history snapshots stored
        'current_match_count': len(summaries),            # Matches in this current batch
        'ny_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'matches': {s['match_id']: s for s in summaries}
    }
    with open('step27.json', 'w') as f:
        json.dump(data2, f, indent=2)
    
    # Command 10: Group by competition
    print("10. Grouping by competition...")
    groups = sort_matches_by_competition_and_time(data2['matches'])
    print(f"    Found {len(groups)} competitions")
    for comp, matches in list(groups.items())[:3]:
        print(f"    {comp}: {len(matches)} matches")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    test_step1_to_step27_pipeline()
