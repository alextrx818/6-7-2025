#!/usr/bin/env python3
"""
Test script to check API rate limiting status and provide recommendations.
"""

import os
import sys
import time
import requests
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_rate_limits():
    """Test API and detect rate limiting patterns."""
    
    username = os.getenv('THESPORTS_USER')
    password = os.getenv('THESPORTS_SECRET')
    
    if not username or not password:
        print("❌ Missing API credentials in .env file")
        return False
    
    print(f"🔍 Testing API rate limits for user: {username}")
    print(f"⏰ Current time: {datetime.now()}")
    print("-" * 60)
    
    # Test different endpoints with different request patterns
    test_cases = [
        {
            'name': 'Single Request Test',
            'url': 'https://api.thesports.com/v1/football/schedules',
            'params': {'user': username, 'secret': password},
            'count': 1,
            'delay': 0
        },
        {
            'name': 'Rate Limit Detection (3 requests)',
            'url': 'https://api.thesports.com/v1/football/schedules',
            'params': {'user': username, 'secret': password},
            'count': 3,
            'delay': 1
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Running: {test_case['name']}")
        print(f"📡 URL: {test_case['url']}")
        
        for i in range(test_case['count']):
            try:
                start_time = time.time()
                response = requests.get(
                    test_case['url'],
                    params=test_case['params'],
                    timeout=10
                )
                duration = time.time() - start_time
                
                result = {
                    'request_num': i + 1,
                    'status_code': response.status_code,
                    'duration': round(duration, 2),
                    'timestamp': datetime.now().isoformat(),
                    'headers': dict(response.headers),
                    'content_length': len(response.content),
                    'url': test_case['url']
                }
                
                # Check for rate limiting indicators
                rate_limit_headers = {}
                for header in response.headers:
                    if any(keyword in header.lower() for keyword in ['rate', 'limit', 'retry', 'quota']):
                        rate_limit_headers[header] = response.headers[header]
                
                result['rate_limit_headers'] = rate_limit_headers
                
                # Analyze response
                if response.status_code == 200:
                    print(f"   ✅ Request {i+1}: SUCCESS ({response.status_code}) - {duration:.2f}s")
                    try:
                        data = response.json()
                        if isinstance(data, dict) and data.get('data'):
                            print(f"      📊 Data received: {len(data['data'])} items")
                            result['data_count'] = len(data['data'])
                        elif isinstance(data, list):
                            print(f"      📊 Data received: {len(data)} items")
                            result['data_count'] = len(data)
                        else:
                            print(f"      📊 Response type: {type(data)}")
                            result['response_type'] = str(type(data))
                    except:
                        print(f"      📄 Non-JSON response: {len(response.content)} bytes")
                        result['content_preview'] = response.text[:200] + "..." if len(response.text) > 200 else response.text
                        
                elif response.status_code == 401:
                    print(f"   🔒 Request {i+1}: UNAUTHORIZED ({response.status_code})")
                    print(f"      💡 This suggests credentials or IP whitelisting issue")
                    result['error_type'] = 'unauthorized'
                    
                elif response.status_code == 429:
                    print(f"   ⏳ Request {i+1}: RATE LIMITED ({response.status_code})")
                    print(f"      📋 Rate limit headers: {rate_limit_headers}")
                    result['error_type'] = 'rate_limited'
                    
                elif response.status_code == 403:
                    print(f"   🚫 Request {i+1}: FORBIDDEN ({response.status_code})")
                    print(f"      💡 This suggests IP not whitelisted or access denied")
                    result['error_type'] = 'forbidden'
                    
                else:
                    print(f"   ❌ Request {i+1}: ERROR ({response.status_code})")
                    print(f"      📄 Response: {response.text[:100]}...")
                    result['error_type'] = 'other'
                    result['response_preview'] = response.text[:200]
                
                # Print rate limiting headers if present
                if rate_limit_headers:
                    print(f"      🔢 Rate limit info: {rate_limit_headers}")
                
                results.append(result)
                
                # Wait between requests if specified
                if test_case['delay'] > 0 and i < test_case['count'] - 1:
                    print(f"      ⏱️  Waiting {test_case['delay']}s...")
                    time.sleep(test_case['delay'])
                    
            except requests.exceptions.Timeout:
                print(f"   ⏰ Request {i+1}: TIMEOUT")
                results.append({
                    'request_num': i + 1,
                    'error_type': 'timeout',
                    'timestamp': datetime.now().isoformat()
                })
                
            except requests.exceptions.ConnectionError:
                print(f"   🌐 Request {i+1}: CONNECTION ERROR")
                results.append({
                    'request_num': i + 1,
                    'error_type': 'connection_error',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"   💥 Request {i+1}: UNEXPECTED ERROR - {str(e)}")
                results.append({
                    'request_num': i + 1,
                    'error_type': 'unexpected',
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                })
    
    # Analyze results and provide recommendations
    print("\n" + "="*60)
    print("📊 ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    # Count different response types
    status_codes = [r.get('status_code') for r in results if 'status_code' in r]
    error_types = [r.get('error_type') for r in results if 'error_type' in r]
    
    if 200 in status_codes:
        print("✅ GOOD NEWS: API is responding with successful requests!")
        success_count = status_codes.count(200)
        print(f"   📈 {success_count}/{len(status_codes)} requests succeeded")
        
        # Check if we got actual data
        data_counts = [r.get('data_count', 0) for r in results if r.get('data_count', 0) > 0]
        if data_counts:
            print(f"   📊 Data received in {len(data_counts)} requests (avg: {sum(data_counts)/len(data_counts):.1f} items)")
        
    elif 401 in status_codes or 'unauthorized' in error_types:
        print("🔒 ISSUE: Authorization problem detected")
        print("   💡 Recommendations:")
        print("      - Double-check API credentials in .env file")
        print("      - Verify your IP address is whitelisted")
        print("      - Contact API provider if credentials are correct")
        
    elif 429 in status_codes or 'rate_limited' in error_types:
        print("⏳ ISSUE: Rate limiting detected")
        print("   💡 Recommendations:")
        print("      - Wait before making more requests")
        print("      - Implement exponential backoff in your pipeline")
        print("      - Consider reducing request frequency")
        
        # Look for rate limit headers in results
        for result in results:
            if result.get('rate_limit_headers'):
                print(f"   📋 Rate limit details: {result['rate_limit_headers']}")
                break
                
    elif 403 in status_codes or 'forbidden' in error_types:
        print("🚫 ISSUE: Access forbidden")
        print("   💡 Recommendations:")
        print("      - Your IP address may not be whitelisted")
        print("      - Contact API provider to whitelist your IP")
        print("      - Check if API access requires special permissions")
        
    else:
        print("❓ MIXED RESULTS: Multiple issues detected")
        print(f"   📊 Status codes seen: {set(status_codes)}")
        print(f"   🐛 Error types: {set(error_types)}")
    
    # Timing recommendations
    if results:
        durations = [r.get('duration', 0) for r in results if 'duration' in r]
        if durations:
            avg_duration = sum(durations) / len(durations)
            print(f"\n⏱️  TIMING: Average request time: {avg_duration:.2f}s")
            if avg_duration > 5:
                print("   ⚠️  Requests are slow - consider increasing timeouts")
            elif avg_duration < 1:
                print("   ✅ Requests are fast - current setup is good")
    
    # Save detailed results
    log_file = f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump({
            'test_timestamp': datetime.now().isoformat(),
            'username': username,
            'results': results,
            'summary': {
                'total_requests': len(results),
                'status_codes': status_codes,
                'error_types': error_types,
                'success_rate': status_codes.count(200) / len(status_codes) if status_codes else 0
            }
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {log_file}")
    
    # Final recommendation
    print("\n🎯 NEXT STEPS:")
    if 200 in status_codes:
        print("   1. ✅ API is working - you can start the pipeline!")
        print("   2. 🔄 Run: ./start.sh")
        print("   3. 👀 Monitor: tail -f pipeline.log")
        return True
    else:
        print("   1. 🔧 Fix the issues identified above")
        print("   2. 🧪 Re-run this test: python3 test_api_rate_limits.py")
        print("   3. 📞 Contact API support if problems persist")
        return False

if __name__ == "__main__":
    success = test_api_rate_limits()
    sys.exit(0 if success else 1)
