#!/usr/bin/env python3
"""
Test different API endpoints to find which ones are authorized.
"""

import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

def test_endpoints():
    username = os.getenv('THESPORTS_USER')
    password = os.getenv('THESPORTS_SECRET')
    
    if not username or not password:
        print("❌ Missing API credentials")
        return
    
    # Common football API endpoints to test
    endpoints = [
        '/v1/football/schedules',
        '/v1/football/matches',
        '/v1/football/fixtures',
        '/v1/football/live',
        '/v1/football/results',  
        '/v1/football/leagues',
        '/v1/football/competitions',
        '/v1/football/teams',
        '/v1/football/standings',
        '/football/schedules',
        '/football/matches',
        '/football/fixtures',
        '/schedules',
        '/matches',
        '/fixtures',
        '/v2/football/schedules',
        '/api/football/schedules',
    ]
    
    base_url = 'https://api.thesports.com'
    
    print(f"🔍 Testing endpoints for user: {username}")
    print("=" * 60)
    
    authorized_endpoints = []
    
    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            response = requests.get(
                url,
                params={'user': username, 'secret': password},
                timeout=5
            )
            
            print(f"\n📡 {endpoint}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and 'err' in data:
                        print(f"   ❌ Error: {data['err']}")
                    elif isinstance(data, dict) and data.get('data'):
                        print(f"   ✅ SUCCESS: {len(data['data'])} items")
                        authorized_endpoints.append(endpoint)
                        # Show sample data
                        print(f"   📊 Sample: {str(data)[:100]}...")
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"   ✅ SUCCESS: {len(data)} items")
                        authorized_endpoints.append(endpoint)
                        print(f"   📊 Sample: {str(data[0])[:100]}...")
                    elif isinstance(data, dict) and len(data) > 0:
                        print(f"   ✅ SUCCESS: Response with {len(data)} keys")
                        authorized_endpoints.append(endpoint)
                        print(f"   📊 Keys: {list(data.keys())}")
                    else:
                        print(f"   ⚠️  Empty response: {data}")
                except:
                    print(f"   📄 Non-JSON: {response.text[:100]}...")
                    if 'err' not in response.text.lower():
                        authorized_endpoints.append(endpoint)
            else:
                print(f"   ❌ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   💥 Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if authorized_endpoints:
        print(f"✅ Found {len(authorized_endpoints)} working endpoints:")
        for endpoint in authorized_endpoints:
            print(f"   • {endpoint}")
    else:
        print("❌ No authorized endpoints found")
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Contact API support to check endpoint permissions")
        print("   2. Verify account subscription level")  
        print("   3. Ask for documentation of available endpoints")
        print("   4. Check if different API version/base URL is needed")

if __name__ == "__main__":
    test_endpoints()
