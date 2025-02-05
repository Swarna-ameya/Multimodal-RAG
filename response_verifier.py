import os


def verify_response(response, matched_items, query):
    """
    Verify if the generated response is accurate and strictly based on document context
    
    Parameters:
    response (str): Generated response from Claude
    matched_items (list): List of context items used for generation
    query (str): Original user query
    
    Returns:
    tuple: (is_valid, feedback)
    """
    # Initialize verification metrics
    verification_results = {
        "source_supported": False,
        "context_only": False,
        "factual_consistency": False,
        "proper_no_answer": False
    }
    
    # If there are no matched items, verify the "cannot answer" response
    if not matched_items:
        expected_response = "I cannot answer this question based on the provided documents."
        verification_results["proper_no_answer"] = expected_response.lower() in response.lower()
        verification_results["context_only"] = verification_results["proper_no_answer"]
        verification_results["source_supported"] = True
        verification_results["factual_consistency"] = True
        
        return (True, ["Response correctly indicates no relevant information in documents"]) if verification_results["proper_no_answer"] else (False, ["Response should indicate no information is available"])
    
    # 1. Check if response cites sources properly
    if "References" in response:
        verification_results["source_supported"] = True
        
        # Extract sources from response
        response_sources = [line.strip() for line in response.split("\n") 
                          if "**[Source:" in line]
        
        # Verify each cited source exists in matched items
        for source in response_sources:
            source_file = source.split("Source:")[1].split(",")[0].strip()
            if not any(os.path.basename(item['path']).startswith(source_file) 
                      for item in matched_items):
                verification_results["source_supported"] = False
                break
    
    # 2. Check if response contains only information from context
    context_text = " ".join([item.get('text', '') for item in matched_items])
    response_sentences = [s.strip() for s in response.split('.') if s.strip()]
    
    # Check each sentence for similarity with context
    verification_results["context_only"] = check_sentences_in_context(
        response_sentences, context_text
    )
    
    # 3. Check factual consistency
    verification_results["factual_consistency"] = check_factual_consistency(
        response, matched_items
    )
    
    # Calculate overall validity with 75% threshold for relevance
    is_valid = (sum(verification_results.values()) / len(verification_results)) >= 0.75
    
    # Generate feedback for improvement
    feedback = generate_verification_feedback(verification_results)
    
    return is_valid, feedback

def check_sentences_in_context(sentences, context):
    """Check if response sentences are derived from context with UTF-8 support"""
    from difflib import SequenceMatcher
    
    def similar(a, b, threshold=0.5):
        # Ensure UTF-8 encoding for comparison
        a = a.encode('utf-8').decode('utf-8', errors='replace').lower()
        b = b.encode('utf-8').decode('utf-8', errors='replace').lower()
        return SequenceMatcher(None, a, b).ratio() > threshold
    
    context_sentences = [s.strip() for s in context.split('.') if s.strip()]
    
    for response_sent in sentences:
        # Skip reference section and very short sentences
        if "References" in response_sent or len(response_sent.split()) < 4:
            continue
            
        # Check if sentence has significant similarity with any context sentence
        if not any(similar(response_sent, context_sent) for context_sent in context_sentences):
            return False
            
    return True

def generate_verification_feedback(results):
    """Generate feedback based on verification results"""
    feedback = []
    
    if not results["source_supported"]:
        feedback.append("Ensure all information is properly cited with references")
    
    if not results["context_only"]:
        feedback.append("Response contains information not found in the documents")
    
    if not results["factual_consistency"]:
        feedback.append("Numerical facts do not match the source material")
        
    if not results.get("proper_no_answer", True):
        feedback.append("Response should clearly state when information is not available")
    
    return feedback if feedback else ["Response meets all verification criteria"]

def extract_key_phrases(text, max_phrases=5):
    """Extract important phrases from text"""
    # Simple extraction based on sentence importance
    sentences = text.split('.')
    important_phrases = []
    
    for sentence in sentences:
        # Basic importance criteria
        if len(sentence.split()) > 5 and any(char.isdigit() for char in sentence):
            important_phrases.append(sentence.strip())
        if len(important_phrases) >= max_phrases:
            break
    
    return important_phrases

def check_factual_consistency(response, matched_items):
    """Check if response facts are consistent with source material"""
    # Extract numerical values and dates from response
    response_numbers = extract_numbers_and_dates(response)
    
    # Extract numerical values and dates from source material
    source_numbers = []
    for item in matched_items:
        if item.get('text'):
            source_numbers.extend(extract_numbers_and_dates(item['text']))
    
    # Check if response numbers exist in source material
    return all(any(abs(rnum - snum) < 0.01 for snum in source_numbers)
              for rnum in response_numbers)

def extract_numbers_and_dates(text):
    """Extract numerical values and dates from text with UTF-8 support"""
    import re
    # Ensure UTF-8 encoding for text processing
    text = text.encode('utf-8').decode('utf-8', errors='replace')
    # Find all numbers (including decimals)
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return [float(num) for num in numbers]