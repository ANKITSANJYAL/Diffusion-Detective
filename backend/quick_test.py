#!/usr/bin/env python3
"""Quick test for semantic steering fix"""

import requests
import json

API_BASE = "http://localhost:8000"

print("Testing semantic steering fix...")
print("-" * 50)

payload = {
    "prompt": "A red sports car",
    "num_inference_steps": 20,
    "intervention_active": True,
    "target_concept": "red",
    "injection_attribute": "blue",
    "intervention_strength": 1.0,
    "auto_detect_concepts": True
}

print(f"Prompt: {payload['prompt']}")
print(f"Intervention: {payload['target_concept']} → {payload['injection_attribute']}")
print()
print("Sending request...")

try:
    response = requests.post(
        f"{API_BASE}/generate",
        json=payload,
        timeout=120
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS!")
        print(f"   - Has baseline: {bool(data.get('image_baseline'))}")
        print(f"   - Has intervened: {bool(data.get('image_intervened'))}")
        print(f"   - Logs: {len(data.get('reasoning_logs', []))}")
        print(f"   - Narrative: {data.get('narrative_text', '')[:100]}...")
    else:
        print("❌ FAILED!")
        print(f"   Error: {response.text}")
        
except Exception as e:
    print(f"❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()
