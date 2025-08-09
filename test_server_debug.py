#!/usr/bin/env python3
"""
Test script to check if the server is running in debug mode and what it's sending.
"""
import requests
import json

def test_server_settings():
    """Test if the server is running and check its debug mode setting"""
    try:
        # Check server settings
        response = requests.get('https://localhost:8000/settings', verify=False)
        if response.status_code == 200:
            settings = response.json()
            print("✅ Server is running!")
            print(f"Debug Mode: {settings.get('debug_mode', 'UNKNOWN')}")
            print(f"Preprocessing Enabled: {settings.get('preprocessing_enabled', 'UNKNOWN')}")
            print(f"All Settings: {json.dumps(settings, indent=2)}")
            return settings.get('debug_mode', False)
        else:
            print(f"❌ Server request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def enable_debug_mode():
    """Enable debug mode on the server"""
    try:
        response = requests.post('https://localhost:8000/settings', 
                               json={'debug_mode': True}, 
                               verify=False)
        if response.status_code == 200:
            print("✅ Debug mode enabled!")
            return True
        else:
            print(f"❌ Failed to enable debug mode: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot enable debug mode: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing server debug mode...\n")
    
    debug_enabled = test_server_settings()
    
    if not debug_enabled:
        print("\n🔧 Attempting to enable debug mode...")
        enable_debug_mode()
        
        print("\n🔍 Checking settings again...")
        test_server_debug()
    else:
        print("✅ Debug mode is already enabled!")