"""
Script to run the enhanced demo API with our new Algerian data.
"""
import os
import subprocess
import sys
import time
import webbrowser

def main():
    """Run the demo API and open the API docs in a browser."""
    print("JibJob Recommendation System - Enhanced Algerian Demo")
    print("=" * 60)
    
    # Get the path to the demo_api.py file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demo_api_path = os.path.join(script_dir, "src", "demo_api.py")
    
    # Check that the demo API file exists
    if not os.path.exists(demo_api_path):
        print(f"ERROR: Demo API file not found at: {demo_api_path}")
        sys.exit(1)
    
    print("\nStarting demo API...")
    print(f"Using Python: {sys.executable}")
    print(f"API path: {demo_api_path}")
    
    # Start the demo API in a new process
    api_process = None
    try:
        # Use explicit Python path for more reliability
        python_path = sys.executable
        api_process = subprocess.Popen([python_path, demo_api_path])
        print("API process started.")
        
        # Wait for the server to start
        print("Waiting for server to initialize...")
        time.sleep(3)
        
        # Open the API docs in a browser
        api_url = "http://localhost:8000/docs"
        print(f"Opening API documentation: {api_url}")
        webbrowser.open(api_url)
        
        print("\nAPI is now running. You can test the following endpoints:")
        print("- GET /recommendations/{user_id}: Get job recommendations for a user")
        print("- GET /jobs/{job_id}: Get details for a specific job")
        print("- GET /users/{user_id}: Get details for a specific user")
        print("- GET /jobs?location={wilaya}: Search for jobs in a specific wilaya")
        print("- GET /wilayas: Get a list of all available wilayas")
        
        print("\nPress Ctrl+C to stop the server when done.")
        
        # Keep the script running until user presses Ctrl+C
        api_process.wait()
    
    except KeyboardInterrupt:
        print("\nStopping demo API...")
        if api_process:
            api_process.terminate()
        print("Demo API stopped.")
    
    except Exception as e:
        print(f"ERROR: Failed to start demo API: {e}")
        if api_process:
            api_process.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
