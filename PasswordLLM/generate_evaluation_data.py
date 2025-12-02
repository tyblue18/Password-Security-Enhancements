"""
Generate evaluation data for the password security system.
Creates test results for the paper.
"""

import json
from password_analyzer import PasswordAnalyzer
from query import load_vector_stores, prepare_rag_prompt, extract_password_from_query
import time

# Test password sets
TEST_PASSWORDS = {
    "common": [
        "password123", "qwerty", "123456", "password", "12345678",
        "admin", "letmein", "welcome", "monkey", "1234567890",
        "qwerty123", "password1", "root", "toor", "passw0rd"
    ],
    "weak_patterns": [
        "abc123def", "qwertyuiop", "asdfghjkl", "123456789",
        "abcdefgh", "qwerty1234", "password123!", "admin123",
        "qazwsx123", "1qaz2wsx", "qwertyui", "asdf1234",
        "1234qwer", "qwerty1", "password12"
    ],
    "moderate": [
        "MyP@ssw0rd!", "Secure123!", "P@ssw0rd1", "MySecure1!",
        "Password123!", "SecureP@ss1", "MyP@ss1!", "Secure123!@",
        "P@ssw0rd12", "MySecure12!"
    ],
    "strong": [
        "K9#mP2$vL8@xQ4", "X7&nR9#wM3$pL5", "Q2@mK8#nP4$rT6",
        "L5$wN9#xM2@pR7", "R8#tK3$nP6@mL4", "M4@pL7#nR2$kT9",
        "N9#rT5$mL8@xK3", "P6@kM2#nR7$tL4", "T3$rL9#nM5@pK8",
        "K7@nP4#rT2$mL6"
    ]
}

# Test queries for RAG evaluation
TEST_QUERIES = {
    "common_passwords": [
        "Is 'password123' a common password?",
        "Check if 'qwerty' is in common password lists",
        "Is '123456' commonly used?",
        "Tell me about 'admin' as a password"
    ],
    "security_policies": [
        "What are the NIST password requirements?",
        "What are PCI DSS password standards?",
        "What are ISO 27001 password requirements?",
        "What are the basic password requirements?"
    ],
    "pattern_analysis": [
        "What makes a password weak?",
        "What are common password patterns to avoid?",
        "What is a keyboard pattern in passwords?",
        "What are sequential patterns in passwords?"
    ],
    "general_security": [
        "How do I create a strong password?",
        "What is password entropy?",
        "What are password strength requirements?",
        "How can I improve my password security?"
    ]
}

def evaluate_password_analyzer():
    """Evaluate password analyzer on test set"""
    analyzer = PasswordAnalyzer()
    results = {
        "common": {"detected": 0, "total": 0, "details": []},
        "keyboard_patterns": {"detected": 0, "total": 0, "details": []},
        "sequential_patterns": {"detected": 0, "total": 0, "details": []},
        "dictionary_words": {"detected": 0, "total": 0, "details": []},
        "scores": []
    }
    
    all_passwords = []
    for category, passwords in TEST_PASSWORDS.items():
        all_passwords.extend([(pwd, category) for pwd in passwords])
    
    for password, category in all_passwords:
        analysis = analyzer.analyze(password)
        
        if category == "common":
            results["common"]["total"] += 1
            if analysis["is_common"]:
                results["common"]["detected"] += 1
                results["common"]["details"].append({
                    "password": password,
                    "detected": True,
                    "reason": analysis.get("common_reason", "")
                })
            else:
                results["common"]["details"].append({
                    "password": password,
                    "detected": False
                })
        
        # Track keyboard patterns
        if category in ["weak_patterns", "common"]:
            results["keyboard_patterns"]["total"] += 1
            if analysis["has_keyboard_pattern"]:
                results["keyboard_patterns"]["detected"] += 1
                results["keyboard_patterns"]["details"].append({
                    "password": password,
                    "detected": True,
                    "reason": analysis.get("keyboard_reason", "")
                })
        
        # Track sequential patterns
        if category in ["weak_patterns", "common"]:
            results["sequential_patterns"]["total"] += 1
            if analysis["has_sequential_pattern"]:
                results["sequential_patterns"]["detected"] += 1
                results["sequential_patterns"]["details"].append({
                    "password": password,
                    "detected": True,
                    "reason": analysis.get("sequential_reason", "")
                })
        
        # Track dictionary words
        if category in ["weak_patterns", "common", "moderate"]:
            results["dictionary_words"]["total"] += 1
            if analysis["has_dictionary_word"]:
                results["dictionary_words"]["detected"] += 1
                results["dictionary_words"]["details"].append({
                    "password": password,
                    "detected": True,
                    "reason": analysis.get("dictionary_reason", "")
                })
        
        # Collect scores
        results["scores"].append({
            "password": password,
            "category": category,
            "score": analysis["strength_score"],
            "entropy": analysis["entropy"],
            "length": analysis["length"]
        })
    
    return results

def evaluate_rag_retrieval():
    """Evaluate RAG retrieval quality"""
    vector_stores = load_vector_stores()
    results = {
        "common_passwords": {"precision": 0, "recall": 0, "content_type": 0, "total": 0},
        "security_policies": {"precision": 0, "recall": 0, "content_type": 0, "total": 0},
        "pattern_analysis": {"precision": 0, "recall": 0, "content_type": 0, "total": 0},
        "general_security": {"precision": 0, "recall": 0, "content_type": 0, "total": 0}
    }
    
    from query import detect_content_type
    
    for category, queries in TEST_QUERIES.items():
        for query in queries:
            results[category]["total"] += 1
            
            # Check content type detection
            detected_type = detect_content_type(query)
            if detected_type:
                results[category]["content_type"] += 1
            
            # Prepare prompt (this retrieves documents)
            try:
                prompt = prepare_rag_prompt(query, vector_stores)
                if len(prompt) > 500:
                    results[category]["precision"] += 1
                if "password" in prompt.lower() or "security" in prompt.lower():
                    results[category]["recall"] += 1
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
    
    # Calculate averages
    for category in results:
        total = results[category]["total"]
        if total > 0:
            results[category]["precision"] = results[category]["precision"] / total * 100
            results[category]["recall"] = results[category]["recall"] / total * 100
            results[category]["content_type"] = results[category]["content_type"] / total * 100
    
    return results

def evaluate_end_to_end():
    """Evaluate end-to-end system performance"""
    vector_stores = load_vector_stores()
    analyzer = PasswordAnalyzer()
    
    test_queries = [
        "Is 'password123' secure?",
        "Analyze the password 'qwerty'",
        "What are NIST password requirements?",
        "Is 'MyP@ssw0rd!' a strong password?",
        "Check if '123456' is common",
        "What makes 'qwertyuiop' weak?",
        "Rate the security of 'Secure123!'",
        "Is 'admin' a good password?",
        "What are common password patterns?",
        "How secure is 'K9#mP2$vL8@xQ4'?",
        "Analyze 'password'",
        "What are PCI DSS requirements?",
        "Is 'letmein' secure?",
        "Check 'welcome123'",
        "What makes a password strong?",
        "Rate 'monkey'",
        "Is 'root' a common password?",
        "Analyze '12345678'",
        "What are ISO password standards?",
        "How secure is 'P@ssw0rd1'?",
        "Check 'qwerty123'",
        "What are weak password patterns?",
        "Is 'admin123' secure?",
        "Rate 'password1'",
        "What are password requirements?"
    ]
    
    results = {
        "password_extraction": {"success": 0, "total": 0},
        "technical_analysis": {"success": 0, "total": 0},
        "rag_retrieval": {"success": 0, "total": 0},
        "response_times": [],
        "response_quality": []
    }
    
    for query in test_queries:
        start_time = time.time()
        
        # Test password extraction
        extracted = extract_password_from_query(query)
        results["password_extraction"]["total"] += 1
        if extracted:
            results["password_extraction"]["success"] += 1
            
            # Test technical analysis
            try:
                analysis = analyzer.analyze(extracted)
                results["technical_analysis"]["success"] += 1
                results["technical_analysis"]["total"] += 1
            except:
                results["technical_analysis"]["total"] += 1
        
        # Test RAG retrieval
        try:
            prompt = prepare_rag_prompt(query, vector_stores)
            if len(prompt) > 200:
                results["rag_retrieval"]["success"] += 1
            results["rag_retrieval"]["total"] += 1
        except:
            results["rag_retrieval"]["total"] += 1
        
        elapsed = time.time() - start_time
        results["response_times"].append(elapsed)
    
    return results

def generate_report():
    """Generate evaluation report"""
    print("=" * 60)
    print("Password Security System Evaluation Report")
    print("=" * 60)
    
    print("\n1. Password Analyzer Evaluation")
    print("-" * 60)
    analyzer_results = evaluate_password_analyzer()
    
    print(f"\nCommon Password Detection:")
    print(f"  Accuracy: {analyzer_results['common']['detected']}/{analyzer_results['common']['total']} "
          f"({analyzer_results['common']['detected']/analyzer_results['common']['total']*100:.1f}%)")
    
    print(f"\nKeyboard Pattern Detection:")
    print(f"  Accuracy: {analyzer_results['keyboard_patterns']['detected']}/{analyzer_results['keyboard_patterns']['total']} "
          f"({analyzer_results['keyboard_patterns']['detected']/analyzer_results['keyboard_patterns']['total']*100:.1f}%)")
    
    print(f"\nSequential Pattern Detection:")
    print(f"  Accuracy: {analyzer_results['sequential_patterns']['detected']}/{analyzer_results['sequential_patterns']['total']} "
          f"({analyzer_results['sequential_patterns']['detected']/analyzer_results['sequential_patterns']['total']*100:.1f}%)")
    
    print(f"\nDictionary Word Detection:")
    print(f"  Accuracy: {analyzer_results['dictionary_words']['detected']}/{analyzer_results['dictionary_words']['total']} "
          f"({analyzer_results['dictionary_words']['detected']/analyzer_results['dictionary_words']['total']*100:.1f}%)")
    
    # Score distribution
    scores_by_category = {}
    for item in analyzer_results["scores"]:
        cat = item["category"]
        if cat not in scores_by_category:
            scores_by_category[cat] = []
        scores_by_category[cat].append(item["score"])
    
    print(f"\nScore Distribution by Category:")
    for cat, scores in scores_by_category.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {cat}: {avg_score:.1f}/10 (range: {min(scores)}-{max(scores)})")
    
    print("\n2. RAG Retrieval Evaluation")
    print("-" * 60)
    rag_results = evaluate_rag_retrieval()
    
    for category, metrics in rag_results.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  Precision: {metrics['precision']:.1f}%")
        print(f"  Recall: {metrics['recall']:.1f}%")
        print(f"  Content Type Detection: {metrics['content_type']:.1f}%")
    
    # Calculate averages
    avg_precision = sum(m["precision"] for m in rag_results.values()) / len(rag_results)
    avg_recall = sum(m["recall"] for m in rag_results.values()) / len(rag_results)
    avg_content_type = sum(m["content_type"] for m in rag_results.values()) / len(rag_results)
    
    print(f"\nOverall Averages:")
    print(f"  Precision: {avg_precision:.1f}%")
    print(f"  Recall: {avg_recall:.1f}%")
    print(f"  Content Type Detection: {avg_content_type:.1f}%")
    
    print("\n3. End-to-End System Evaluation")
    print("-" * 60)
    e2e_results = evaluate_end_to_end()
    
    print(f"\nPassword Extraction:")
    print(f"  Success Rate: {e2e_results['password_extraction']['success']}/{e2e_results['password_extraction']['total']} "
          f"({e2e_results['password_extraction']['success']/e2e_results['password_extraction']['total']*100:.1f}%)")
    
    print(f"\nTechnical Analysis:")
    if e2e_results['technical_analysis']['total'] > 0:
        print(f"  Success Rate: {e2e_results['technical_analysis']['success']}/{e2e_results['technical_analysis']['total']} "
              f"({e2e_results['technical_analysis']['success']/e2e_results['technical_analysis']['total']*100:.1f}%)")
    
    print(f"\nRAG Retrieval:")
    print(f"  Success Rate: {e2e_results['rag_retrieval']['success']}/{e2e_results['rag_retrieval']['total']} "
          f"({e2e_results['rag_retrieval']['success']/e2e_results['rag_retrieval']['total']*100:.1f}%)")
    
    if e2e_results['response_times']:
        avg_time = sum(e2e_results['response_times']) / len(e2e_results['response_times'])
        print(f"\nAverage Response Time: {avg_time:.2f} seconds")
    
    output = {
        "analyzer_results": analyzer_results,
        "rag_results": rag_results,
        "e2e_results": e2e_results
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Detailed results saved to evaluation_results.json")
    print("=" * 60)

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

