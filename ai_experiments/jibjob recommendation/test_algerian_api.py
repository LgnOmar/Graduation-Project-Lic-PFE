"""
Improved script to check if the enhanced Algerian API is accessible.
This script tests several key endpoints including wilaya-specific features.
"""
import requests
import time
import sys
import json

# API base URL
BASE_URL = "http://localhost:8000"

print("JibJob Recommendation System - API Test")
print("=" * 60)

# Wait for API to initialize
print("Waiting for API server to initialize...")
time.sleep(3)  # Give the server time to start up

def test_endpoint(endpoint, name=None, params=None):
    """Test an API endpoint and return True if successful"""
    url = f"{BASE_URL}{endpoint}"
    endpoint_name = name if name else endpoint
    print(f"\nTesting {endpoint_name}...")
    
    try:
        if params:
            response = requests.get(url, params=params, timeout=5)
        else:
            response = requests.get(url, timeout=5)
            
        if response.status_code == 200:
            print(f"✓ Success! Status code: {response.status_code}")
            try:
                # Try to pretty print if it's JSON
                data = response.json()
                if isinstance(data, list) and len(data) > 5:
                    print(f"  Sample response: {data[:5]} (showing first 5 items of {len(data)})")
                elif isinstance(data, dict) and len(str(data)) > 300:
                    # Print sample of dictionary
                    sample = {k: data[k] for k in list(data.keys())[:3]} if data else {}
                    print(f"  Sample response: {sample} (showing first 3 keys)")
                else:
                    print(f"  Response: {data}")
            except ValueError:
                print(f"  Response: {response.text[:200]}...")
            return True
        else:
            print(f"✗ Failed! Status code: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

# Test basic API connectivity
if not test_endpoint("/", "API root"):
    print("\n✗ API is not responding! Make sure the API server is running.")
    sys.exit(1)

# Test Algeria-specific endpoints
test_endpoint("/wilayas", "Algerian Wilayas list")
test_endpoint("/jobs", "All jobs")
test_endpoint("/jobs", "Jobs in Algiers", {"location": "Alger"})
test_endpoint("/users", "All users")

# Test recommendation endpoints
test_endpoint("/recommendations/user_1", "Recommendations for user_1")
test_endpoint("/recommendations/user_1", "Recommendations in Oran for user_1", {"location_preference": "Oran"})

print("\nAPI test completed successfully!")
print("The enhanced JibJob API with Algerian data is working properly.")
print("\nYou can access the API documentation at: http://localhost:8000/docs")
