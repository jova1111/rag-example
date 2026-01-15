"""Test script for document classification endpoint."""

import requests
from typing import Dict


# API endpoint
API_URL = "http://localhost:8000/classify/text"


# Test data: Unclassified documents
unclassified_texts = [
    "The annual military parade will be held on June 15th at the downtown square. All citizens are invited to attend.",
    "The Department of Defense announced a new recruitment campaign targeting young professionals interested in cybersecurity careers.",
    "Military service members are entitled to educational benefits including tuition assistance and GI Bill programs.",
    "The military base will host an open house event next Saturday for families and community members.",
    "This month's edition includes stories about community outreach programs and volunteer opportunities at local bases.",
    "We are seeking qualified candidates for administrative positions at regional military offices. Apply online today.",
    "The armed forces remain committed to maintaining peace and security through international partnerships.",
    "The military family support center will offer free workshops on financial planning next week.",
    "Service members should update their emergency contact information in the personnel database.",
    "The military museum will extend visiting hours during the summer months for tourism season."
]


# Test data: Confidential documents
confidential_texts = [
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


def classify_text(text: str) -> Dict:
    """Call the classification API endpoint.
    
    Args:
        text: Text to classify
        
    Returns:
        Classification result dictionary
    """
    try:
        response = requests.post(
            API_URL,
            json={"text": text, "include_context": False},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None


def test_classifications():
    """Test classification endpoint with predefined test data."""
    
    print("=" * 80)
    print("DOCUMENT CLASSIFICATION TEST")
    print("=" * 80)
    
    # Acceptable classifications
    confidential_acceptable = ["confidential", "secret", "top secret"]
    unclassified_acceptable = ["unclassified"]
    
    # Test unclassified documents
    print("\n📄 Testing UNCLASSIFIED documents...")
    print("-" * 80)
    
    unclassified_correct = 0
    unclassified_total = len(unclassified_texts)
    
    for i, text in enumerate(unclassified_texts, 1):
        print(f"\n[{i}/{unclassified_total}] Testing: {text[:70]}...")
        result = classify_text(text)
        
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
    
    for i, text in enumerate(confidential_texts, 1):
        print(f"\n[{i}/{confidential_total}] Testing: {text[:70]}...")
        result = classify_text(text)
        
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
    
    print("\n" + "=" * 80)
    
    return {
        "unclassified_correct": unclassified_correct,
        "unclassified_total": unclassified_total,
        "confidential_correct": confidential_correct,
        "confidential_total": confidential_total,
        "total_correct": total_correct,
        "total_tests": total_tests,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    # Check if API is running
    try:
        health_response = requests.get("http://localhost:8000/health")
        health_response.raise_for_status()
        print("✓ API is running and healthy\n")
    except requests.exceptions.RequestException:
        print("✗ Error: API is not running. Please start the API server first:")
        print("  python app.py")
        exit(1)
    
    # Run tests
    results = test_classifications()
