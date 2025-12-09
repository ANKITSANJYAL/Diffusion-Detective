"""
Test Script for Diffusion Detective v2.0
Tests multi-token tracking and semantic steering functionality.
"""

import requests
import json
import time
from pathlib import Path


API_BASE = "http://localhost:8000"


def test_health():
    """Test that the API is running."""
    print("\n🏥 Testing Health Endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_basic_generation():
    """Test basic generation without v2.0 features (backward compatibility)."""
    print("\n📝 Testing Basic Generation (v1.0 compatibility)...")
    
    payload = {
        "prompt": "A majestic tiger on a mountain",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "intervention_active": False,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate", json=payload, timeout=300)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        print(f"  - Has baseline image: {bool(data.get('image_baseline'))}")
        print(f"  - Has intervened image: {bool(data.get('image_intervened'))}")
        print(f"  - Log entries: {len(data.get('reasoning_logs', []))}")
        print(f"  - Narrative length: {len(data.get('narrative_text', ''))}")
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_multi_token_tracking():
    """Test v2.0 multi-token attention tracking."""
    print("\n🎯 Testing Multi-Token Attention Tracking...")
    
    payload = {
        "prompt": "Majestic tiger on mountain at sunset",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "intervention_active": False,
        "auto_detect_concepts": True,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate", json=payload, timeout=300)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        
        # Check for focus scores in logs
        focus_found = False
        for log in data.get('reasoning_logs', []):
            if isinstance(log, dict) and log.get('metadata', {}).get('focus'):
                focus_found = True
                focus_scores = log['metadata']['focus']
                print(f"\n📊 Focus Scores at Step {log['step']}:")
                for concept, scores in focus_scores.items():
                    conf = scores.get('confidence', 0)
                    cat = scores.get('category', 'unknown')
                    print(f"  - {concept.upper()}: {conf:.1f}% ({cat})")
                break
        
        if focus_found:
            print("\n✓ Multi-token tracking working!")
        else:
            print("\n⚠️ No focus scores found in logs")
        
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_semantic_steering():
    """Test v2.0 semantic intervention."""
    print("\n🧪 Testing Semantic Steering...")
    
    payload = {
        "prompt": "A red sports car on the beach",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "intervention_active": True,
        "intervention_strength": 1.5,
        "intervention_step_start": 15,
        "intervention_step_end": 5,
        "target_concept": "red",
        "injection_attribute": "blue",
        "auto_detect_concepts": True,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"\n💡 Hypothesis: What if 'red' becomes 'blue'?")
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate", json=payload, timeout=300)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        
        # Check for semantic steering in logs
        semantic_found = False
        for log in data.get('reasoning_logs', []):
            if isinstance(log, dict):
                msg = log.get('message', '')
                if 'Semantic steering' in msg or ('red' in msg.lower() and 'blue' in msg.lower()):
                    print(f"\n🎨 Intervention Log: {msg}")
                    semantic_found = True
                    break
        
        if semantic_found:
            print("\n✓ Semantic steering detected!")
        else:
            print("\n⚠️ No semantic steering logs found")
        
        # Show narrative
        narrative = data.get('narrative_text', '')
        if narrative:
            print(f"\n📖 Narrative Preview:")
            print(f"{narrative[:200]}...")
        
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_complex_scenario():
    """Test complex v2.0 scenario with multiple tracked concepts and semantic steering."""
    print("\n🎭 Testing Complex Scenario...")
    
    payload = {
        "prompt": "Elegant robot warrior in neon city at night",
        "num_inference_steps": 25,
        "guidance_scale": 8.0,
        "intervention_active": True,
        "intervention_strength": 1.2,
        "intervention_step_start": 20,
        "intervention_step_end": 10,
        "target_concept": "robot",
        "injection_attribute": "samurai",
        "auto_detect_concepts": True,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"\n💡 Hypothesis: Transform 'robot' into 'samurai' aesthetic")
    print(f"   Expected tracking: robot, warrior, city, night, elegant, neon")
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate", json=payload, timeout=300)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        
        # Analyze all focus scores
        all_concepts = set()
        for log in data.get('reasoning_logs', []):
            if isinstance(log, dict) and log.get('metadata', {}).get('focus'):
                focus_scores = log['metadata']['focus']
                all_concepts.update(focus_scores.keys())
        
        print(f"\n📊 Detected Concepts: {', '.join(sorted(all_concepts))}")
        print(f"   Total: {len(all_concepts)} unique concepts tracked")
        
        # Show final narrative
        narrative = data.get('narrative_text', '')
        if narrative:
            print(f"\n📖 Full Narrative:")
            print(f"{narrative}")
        
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def test_balance_analysis():
    """Test that narrative discusses prompt balance."""
    print("\n⚖️ Testing Balance Analysis in Narrative...")
    
    payload = {
        "prompt": "Giant mountain landscape with tiny bird",
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "intervention_active": False,
        "auto_detect_concepts": True,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"\n💡 Expected: Narrative should discuss if 'bird' is neglected vs 'mountain'")
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate", json=payload, timeout=300)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        
        narrative = data.get('narrative_text', '').lower()
        balance_keywords = ['balance', 'neglect', 'ignore', 'focus', 'attention', 'distribution']
        
        found_keywords = [kw for kw in balance_keywords if kw in narrative]
        
        if found_keywords:
            print(f"\n✓ Balance analysis detected!")
            print(f"   Keywords found: {', '.join(found_keywords)}")
        else:
            print(f"\n⚠️ No clear balance analysis in narrative")
        
        print(f"\n📖 Narrative:")
        print(f"{data.get('narrative_text', '')}")
        
        return True
    else:
        print(f"✗ Failed: {response.text}")
        return False


def run_all_tests():
    """Run complete test suite."""
    print("=" * 70)
    print("🧪 DIFFUSION DETECTIVE v2.0 TEST SUITE")
    print("=" * 70)
    
    results = {
        "Health Check": test_health(),
        "Basic Generation (v1.0 compat)": test_basic_generation(),
        "Multi-Token Tracking": test_multi_token_tracking(),
        "Semantic Steering": test_semantic_steering(),
        "Complex Scenario": test_complex_scenario(),
        "Balance Analysis": test_balance_analysis()
    }
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status} | {test_name}")
    
    print("=" * 70)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
