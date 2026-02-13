"""Test script for document classification endpoint."""

import requests
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# API endpoint
API_URL = "http://localhost:8001/classify/text"

number_of_tests = 1  # Multiplier for test data


# Test data: Training-related documents
_base_training_texts = [
    "Training schedule for next week: Monday - rifle range, Tuesday - tactical exercises, Wednesday - physical fitness.",
    "All personnel must complete the cybersecurity awareness training by end of month.",
    "Leadership development workshop scheduled for officers next Thursday at 0900 hours.",
    "Weekly training schedule includes obstacle course, weapons qualification, and squad tactics.",
    "Combat readiness drills will be conducted every morning at 0600 starting next week.",
    "Professional development training for NCOs focusing on mentorship and team building.",
    "Medical emergency response training required for all first responders this quarter.",
    "Advanced tactical driving course available for personnel assigned to security details.",
    "Annual weapons qualification and safety certification due before deployment.",
    "New equipment familiarization training sessions scheduled throughout the month."
]

# Generate texts by repeating base samples
training_texts = _base_training_texts * number_of_tests


# Test data: Intelligence/Operations documents
_base_operations_texts = [
    "Intelligence reports indicate hostile surveillance activities detected near strategic infrastructure. Immediate countermeasures required.",
    "Task force deployment scheduled for 0300 hours. Radio silence protocols in effect. Secure channel communication only.",
    "Vulnerability analysis of perimeter defense systems reveals critical weaknesses requiring urgent remediation.",
    "Encrypted satellite data shows unauthorized vessel movements in restricted maritime zone. Alert status elevated.",
    "SIGINT intercepts reveal adversary force positioning and tactical intentions for upcoming maneuvers.",
    "Emergency evacuation procedures and safe house locations for high-value personnel during crisis scenarios.",
    "Special operations unit insertion coordinates and extraction timelines for mission in hostile territory.",
    "Cyber operations detected network intrusion attempts targeting classified communication systems.",
    "Patrol routes updated based on intelligence assessment of threat levels in operational area.",
    "Mission briefing: Conduct reconnaissance of enemy positions along the northern border."
]

operations_texts = _base_operations_texts * number_of_tests


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
    
    # Expected tags for different document types
    training_expected_tags = ["training", "schedule", "exercise", "drill", "qualification", "workshop", "course"]
    operations_expected_tags = ["intelligence", "operations", "deployment", "mission", "tactical", "security", "reconnaissance"]
    
    all_timings = []
    
    # Test training documents
    print("\n📄 Testing TRAINING documents (concurrent requests)...")
    print("-" * 80)
    
    training_correct = 0
    training_failed = 0
    training_total = len(training_texts)
    
    # Process in batch
    training_results, training_timings = classify_batch(training_texts, batch_size=500)
    all_timings.extend(training_timings)
    
    for i, (idx, text, result) in enumerate(training_results, 1):
        print(f"\n[{i}/{training_total}] Testing: {text[:70]}...")
        
        if result:
            tags = result.get("tags", [])
            confidence = result.get("confidence", 0)
            chunks_processed = result.get("chunks_processed", 0)
            
            # Check if any expected tags are present
            has_expected_tag = any(tag.lower() in [t.lower() for t in tags] for tag in training_expected_tags)
            is_correct = has_expected_tag and len(tags) > 0
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            print(f"    Tags: {', '.join(tags)}")
            print(f"    Confidence: {confidence:.2%} | Chunks: {chunks_processed}")
            print(f"    Status: {status}")
            
            if is_correct:
                training_correct += 1
        else:
            training_failed += 1
            print(f"    Status: ✗ FAIL (API Error)")
    
    # Test operations/intelligence documents
    print("\n\n🔒 Testing OPERATIONS/INTELLIGENCE documents...")
    print("-" * 80)
    
    operations_correct = 0
    operations_failed = 0
    operations_total = len(operations_texts)
    
    # Process in batch
    operations_results, operations_timings = classify_batch(operations_texts, batch_size=500)
    all_timings.extend(operations_timings)
    
    for i, (idx, text, result) in enumerate(operations_results, 1):
        print(f"\n[{i}/{operations_total}] Testing: {text[:70]}...")
        
        if result:
            tags = result.get("tags", [])
            confidence = result.get("confidence", 0)
            chunks_processed = result.get("chunks_processed", 0)
            
            # Check if any expected tags are present
            has_expected_tag = any(tag.lower() in [t.lower() for t in tags] for tag in operations_expected_tags)
            is_correct = has_expected_tag and len(tags) > 0
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            print(f"    Tags: {', '.join(tags)}")
            print(f"    Confidence: {confidence:.2%} | Chunks: {chunks_processed}")
            print(f"    Status: {status}")
            
            if is_correct:
                operations_correct += 1
        else:
            operations_failed += 1
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
    
    total_tests = training_total + operations_total
    total_correct = training_correct + operations_correct
    total_failed = training_failed + operations_failed
    accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTraining Documents:")
    print(f"  ✓ Correct: {training_correct}/{training_total} ({training_correct/training_total*100:.1f}%)")
    print(f"  ✗ Failed API requests: {training_failed}")
    
    print(f"\nOperations/Intelligence Documents:")
    print(f"  ✓ Correct: {operations_correct}/{operations_total} ({operations_correct/operations_total*100:.1f}%)")
    print(f"  ✗ Failed API requests: {operations_failed}")
    
    print(f"\nOverall Accuracy:")
    print(f"  ✓ {total_correct}/{total_tests} correct ({accuracy:.1f}%)")
    print(f"  ✗ Failed API requests: {total_failed}")
    
    print(f"\n⏱️  Performance Metrics:")
    print(f"  • Total execution time: {total_elapsed_time:.2f}ms ({total_elapsed_time/1000:.2f}s)")
    print(f"  • Average time per API call: {avg_time_per_call:.2f}ms")
    print(f"  • Min time: {min_time:.2f}ms")
    print(f"  • Max time: {max_time:.2f}ms")
    print(f"  • Total API calls: {len(all_timings)}")
    
    print("\n" + "=" * 80)
    
    return {
        "training_correct": training_correct,
        "training_failed": training_failed,
        "training_total": training_total,
        "operations_correct": operations_correct,
        "operations_failed": operations_failed,
        "operations_total": operations_total,
        "total_correct": total_correct,
        "total_failed": total_failed,
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
