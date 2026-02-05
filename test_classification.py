"""Test script for document classification endpoint."""

import requests
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# API endpoint
API_URL = "http://localhost:8001/classify/text"

number_of_tests = 25  # Multiplier for test data


# Test data: Unclassified documents (base samples)
_base_unclassified = [
    "The annual military parade will be held on June 15th at the downtown square. All citizens are invited to attend.",
    "The Department of Defense announced a new recruitment campaign targeting young professionals interested in cybersecurity careers.",
    "Military service members are entitled to educational benefits including tuition assistance and GI Bill programs.",
    "The military base will host an open house event next Saturday for families and community members.",
    "This month's edition includes stories about community outreach programs and volunteer opportunities at local bases.",
    "We are seeking candidates for administrative positions at regional military offices. Apply online today.",
    "The armed forces remain committed to maintaining peace and security through international partnerships.",
    "The military family support center will offer free workshops on financial planning next week.",
    "Service members should update their emergency contact information in the personnel database.",
    "The military museum will extend visiting hours during the summer months for tourism season."
]

# Generate texts by repeating base samples
unclassified_texts = _base_unclassified * number_of_tests


# Test data: Confidential documents
_base_confidential_texts = [
    "Intelligence reports indicate hostile surveillance activities detected near strategic infrastructure. Immediate countermeasures required.",
    "Task force deployment scheduled for 0300 hours. Radio silence protocols in effect. Secure channel communication only.",
    "Vulnerability analysis of perimeter defense systems reveals critical weaknesses requiring urgent remediation.",
    "Encrypted satellite data shows unauthorized vessel movements in restricted maritime zone. Alert status elevated.",
    "Agent credentials and undercover assignment details for covert operations in foreign territories.",
    "SIGINT intercepts reveal adversary force positioning and tactical intentions for upcoming maneuvers.",
    "Emergency evacuation procedures and safe house locations for high-value personnel during crisis scenarios.",
    "Advanced missile system specifications including range capabilities, payload configurations, and targeting algorithms.",
    "Special operations unit insertion coordinates and extraction timelines for mission in hostile territory.",
    "Sensitive negotiations regarding military base agreements and strategic alliance terms with foreign governments."
]

confidential_texts = _base_confidential_texts * number_of_tests


def classify_text(text: str) -> Tuple[Dict, float]:
    """Call the classification API endpoint.
    
    Args:
        text: Text to classify
        
    Returns:
        Tuple of (classification result dictionary, elapsed time in ms)
    """
    try:
        print(f"Classifying text: {text[:50]}...")
        start_time = time.perf_counter()
        response = requests.post(
            API_URL,
            json={"text": text, "include_context": False},
            headers={"Content-Type": "application/json"},
            timeout=1200  # 20 minutes timeout
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return response.json(), elapsed_ms
    except requests.exceptions.RequestException as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else 0
        print(f"Error calling API: {e}")
        return None, elapsed_ms


def classify_batch(texts: List[str], batch_size: int = 100) -> Tuple[List[Tuple[int, str, Dict]], List[float]]:
    """Classify multiple texts concurrently in batches.
    
    Args:
        texts: List of texts to classify
        batch_size: Number of concurrent requests per batch
        
    Returns:
        Tuple of (List of tuples (index, text, result), List of elapsed times in ms)
    """
    results = []
    timings = []
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(classify_text, text): (i, text) 
            for i, text in enumerate(texts)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_text):
            i, text = future_to_text[future]
            try:
                result, elapsed = future.result()
                results.append((i, text, result))
                timings.append((i, elapsed))
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                results.append((i, text, None))
                timings.append((i, 0))
    
    # Sort by original index to maintain order
    results.sort(key=lambda x: x[0])
    timings.sort(key=lambda x: x[0])
    timing_values = [t[1] for t in timings]
    return results, timing_values


def test_classifications():
    """Test classification endpoint with predefined test data."""
    
    # Start total timer
    total_start_time = time.perf_counter()
    
    print("=" * 80)
    print("DOCUMENT CLASSIFICATION TEST")
    print("=" * 80)
    
    # Acceptable classifications
    confidential_acceptable = ["confidential", "secret", "top secret"]
    unclassified_acceptable = ["unclassified"]
    
    all_timings = []
    
    # Test unclassified documents
    print("\n📄 Testing UNCLASSIFIED documents (100 concurrent requests)...")
    print("-" * 80)
    
    unclassified_correct = 0
    unclassified_total = len(unclassified_texts)
    
    # Process in batch
    unclassified_results, unclassified_timings = classify_batch(unclassified_texts, batch_size=500)
    all_timings.extend(unclassified_timings)
    
    for i, (idx, text, result) in enumerate(unclassified_results, 1):
        print(f"\n[{i}/{unclassified_total}] Testing: {text[:70]}...")
        
        if result:
            classification = result.get("classification", "").lower()
            confidence = result.get("confidence", 0)
            
            is_correct = classification in unclassified_acceptable
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            print(f"    Result: {classification.upper()} (confidence: {confidence:.2%})")
            print(f"    Status: {status}")
            
            if is_correct:
                unclassified_correct += 1
        else:
            print(f"    Status: ✗ FAIL (API Error)")
    
    # Test confidential documents
    print("\n\n🔒 Testing CONFIDENTIAL documents...")
    print("-" * 80)
    
    confidential_correct = 0
    confidential_total = len(confidential_texts)
    
    # Process in batch
    confidential_results, confidential_timings = classify_batch(confidential_texts, batch_size=500)
    all_timings.extend(confidential_timings)
    
    for i, (idx, text, result) in enumerate(confidential_results, 1):
        print(f"\n[{i}/{confidential_total}] Testing: {text[:70]}...")
        
        if result:
            classification = result.get("classification", "").lower()
            confidence = result.get("confidence", 0)
            
            is_correct = classification in confidential_acceptable
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            print(f"    Result: {classification.upper()} (confidence: {confidence:.2%})")
            print(f"    Status: {status}")
            
            if is_correct:
                confidential_correct += 1
        else:
            print(f"    Status: ✗ FAIL (API Error)")
    
    # Calculate total execution time
    total_elapsed_time = (time.perf_counter() - total_start_time) * 1000
    
    # Calculate timing statistics
    avg_time_per_call = sum(all_timings) / len(all_timings) if all_timings else 0
    min_time = min(all_timings) if all_timings else 0
    max_time = max(all_timings) if all_timings else 0
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    total_tests = unclassified_total + confidential_total
    total_correct = unclassified_correct + confidential_correct
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nUnclassified Documents:")
    print(f"  ✓ Correct: {unclassified_correct}/{unclassified_total} ({unclassified_correct/unclassified_total*100:.1f}%)")
    
    print(f"\nConfidential Documents:")
    print(f"  ✓ Correct: {confidential_correct}/{confidential_total} ({confidential_correct/confidential_total*100:.1f}%)")
    
    print(f"\nOverall Accuracy:")
    print(f"  ✓ {total_correct}/{total_tests} correct ({accuracy:.1f}%)")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"  • Total execution time: {total_elapsed_time:.2f}ms ({total_elapsed_time/1000:.2f}s)")
    print(f"  • Average time per API call: {avg_time_per_call:.2f}ms")
    print(f"  • Min time: {min_time:.2f}ms")
    print(f"  • Max time: {max_time:.2f}ms")
    print(f"  • Total API calls: {len(all_timings)}")
    
    print("\n" + "=" * 80)
    
    return {
        "unclassified_correct": unclassified_correct,
        "unclassified_total": unclassified_total,
        "confidential_correct": confidential_correct,
        "confidential_total": confidential_total,
        "total_correct": total_correct,
        "total_tests": total_tests,
        "accuracy": accuracy,
        "total_time_ms": total_elapsed_time,
        "avg_time_per_call_ms": avg_time_per_call,
        "min_time_ms": min_time,
        "max_time_ms": max_time
    }


if __name__ == "__main__":
    # Run tests
    results = test_classifications()
