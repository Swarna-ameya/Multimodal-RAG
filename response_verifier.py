import os


def verify_response(response, matched_items, query):
    """
    Verify if the generated response is accurate and relevant to the context
    Logs verification results to console/logs rather than showing in chat
    
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
        "query_addressed": False,
        "context_relevant": False,
        "factual_consistency": False
    }
    
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
    
    # 2. Check if response addresses the query
    query_keywords = set(query.lower().split())
    response_lower = response.lower()
    
    # Remove common words for better keyword matching
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    query_keywords = query_keywords - common_words
    
    if any(keyword in response_lower for keyword in query_keywords):
        verification_results["query_addressed"] = True
    
    # 3. Check context relevance
    context_text = " ".join([item.get('text', '') for item in matched_items])
    if context_text:
        # Check if key phrases from context appear in response
        key_phrases = extract_key_phrases(context_text)
        if any(phrase.lower() in response_lower for phrase in key_phrases):
            verification_results["context_relevant"] = True
    
    # 4. Check factual consistency
    verification_results["factual_consistency"] = check_factual_consistency(
        response, matched_items
    )
    
    # Calculate overall validity
    is_valid = (sum(verification_results.values()) / len(verification_results)) >= 0.75
    
    # Generate feedback for improvement if needed
    feedback = generate_verification_feedback(verification_results)
    
    return is_valid, feedback

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
    """Extract numerical values and dates from text"""
    import re
    # Find all numbers (including decimals)
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return [float(num) for num in numbers]

def generate_verification_feedback(results):
    """Generate feedback based on verification results"""
    feedback = []
    
    if not results["source_supported"]:
        feedback.append("Ensure all cited sources are from the provided context")
    
    if not results["query_addressed"]:
        feedback.append("Response should more directly address the user's query")
    
    if not results["context_relevant"]:
        feedback.append("Include more specific information from the provided context")
    
    if not results["factual_consistency"]:
        feedback.append("Verify numerical facts and dates match the source material")
    
    return feedback if feedback else ["Response meets all verification criteria"]