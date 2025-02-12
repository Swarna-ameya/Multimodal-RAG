import os
import re
from difflib import SequenceMatcher

def check_sentences_in_context(sentences, context):
    """Enhanced checking of response content against source material"""
    def normalize_text(text):
        """Normalize text for comparison"""
        text = text.encode('utf-8').decode('utf-8', errors='replace').lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def find_substantial_overlap(sentence, context_parts, min_length=4):
        """Find substantial overlapping content"""
        sentence_words = set(normalize_text(sentence).split())
        if len(sentence_words) < min_length:
            return True  # Skip very short sentences
            
        for part in context_parts:
            part_words = set(normalize_text(part).split())
            overlap = len(sentence_words.intersection(part_words))
            if overlap >= min(len(sentence_words) * 0.9, len(part_words) * 0.9):  # Increased threshold
                return True
        return False
    
    # Split context into meaningful chunks
    context_parts = []
    current_chunk = []
    
    for line in context.split('\n'):
        if line.strip():
            current_chunk.append(line)
        elif current_chunk:
            context_parts.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        context_parts.append(' '.join(current_chunk))
    
    # Also split by periods for sentence-level checking
    context_parts.extend([s.strip() for s in context.split('.') if s.strip()])
    
    # Check each response sentence
    for sentence in sentences:
        # Skip reference section and very short content
        if "References" in sentence or len(sentence.split()) < 4:
            continue
            
        # Must find substantial overlap with context
        if not find_substantial_overlap(sentence, context_parts):
            return False
            
    return True

def verify_response(response, matched_items, query):
    """Stricter verification of response accuracy"""
    verification_results = {
        "exact_content": False,
        "format_preserved": False,
        "no_hallucination": False,
        "proper_structure": False,
        "proper_references": False
    }
    
    def check_exact_content(response_text, source_items):
        """Stricter verification of exact content matching"""
        def normalize(text):
            text = text.lower().strip()
            text = re.sub(r'\s+', ' ', text)
            return text
        
        # Split into parts more carefully
        def split_content(text):
            parts = set()
            # Split by newlines first
            for line in text.split('\n'):
                if line.strip():
                    # Then split by periods, preserving structure
                    if re.match(r'^[#\-*\d]', line.strip()):  # Preserve structured content
                        parts.add(normalize(line))
                    else:
                        parts.update(normalize(s) for s in line.split('.') if s.strip())
            return parts
        
        response_parts = split_content(response_text)
        source_parts = set()
        
        for item in source_items:
            if item.get('text'):
                source_parts.update(split_content(item['text']))
        
        # Calculate content overlap
        matching_parts = response_parts.intersection(source_parts)
        if not matching_parts:
            return False
            
        # Very strict allowed connections
        allowed_connections = {'and', 'but', 'or'}
        non_matching = response_parts - source_parts
        
        return all(
            part in allowed_connections or len(part.split()) <= 2
            for part in non_matching
        )

    def check_format_preservation(response_text, source_items):
        """Verify formatting is preserved"""
        formats = {
            'bullet_points': (r'^\s*[-*•]\s+', False),
            'numbered_lists': (r'^\s*\d+\.\s+', False),
            'code_blocks': (r'```\w*\n[\s\S]*?```', False),
            'tables': (r'\|[^|]+\|', False),
            'headings': (r'^#+\s+', False)
        }
        
        # Check source for formats
        for item in source_items:
            if not item.get('text'):
                continue
            for format_name, (pattern, _) in formats.items():
                if re.search(pattern, item['text'], re.MULTILINE):
                    formats[format_name] = (pattern, True)
        
        # Verify formats are preserved in response
        preserved = True
        for format_name, (pattern, exists) in formats.items():
            if exists and not re.search(pattern, response_text, re.MULTILINE):
                preserved = False
                break
                
        return preserved
    
    def check_no_hallucination(text):
        """Enhanced check for speculative or external information"""
        suspicious_patterns = [
            # Speculative language
            r'(?:may|might|could|probably|possibly|perhaps|maybe|likely)',
            # Assumptive phrases
            r'(?:I think|I believe|I assume|presumably|seems to|appears to)',
            # External references
            r'(?:typically|generally|usually|commonly|often|in most cases)',
            # Uncertainty
            r'(?:unclear|unknown|not specified|depending on|varies|various)',
            # Suggestions
            r'(?:recommended|suggested|advised|consider|try|attempt)',
            # Future/conditional
            r'(?:would|should|can be|will be|expected to)',
            # External knowledge markers
            r'(?:standard practice|best practice|common approach|popular|widely used)',
            # Opinion indicators
            r'(?:better|worse|preferred|ideal|optimal|recommended)',
            # Generalizations
            r'(?:always|never|all|none|every|most|many)',
            # Assumptions
            r'(?:assuming|presumably|if we|suppose|let\'s say)',
            # External references
            r'(?:documentation|manual|guide|reference|according to)',
            # Uncertainty qualifiers
            r'(?:somewhat|relatively|fairly|quite|rather|approximately)'
        ]
        
        # Check each pattern
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True

    def check_references(response_text, source_items):
        """Verify references are properly formatted and accurate"""
        if "References" not in response_text:
            return False
            
        ref_section = response_text.split("References")[-1]
        refs = re.findall(r'\*\*\[Source: ([^,]+), page (\d+)\]\*\*', ref_section)
        
        if not refs:
            return False
            
        # Verify each reference exists
        for source_file, page in refs:
            source_exists = any(
                os.path.basename(item['path']).startswith(source_file) and
                str(item['page'] + 1) == page
                for item in source_items
            )
            if not source_exists:
                return False
                
        return True
    
    def check_structure_preservation(response_text, source_items):
        """Verify document structure is preserved"""
        patterns = {
            'headings': r'^#+\s+.+$',
            'bullet_lists': r'^\s*[-*]\s+.+$',
            'numbered_lists': r'^\s*\d+\.\s+.+$',
            'tables': r'\|[^|]+\|',
            'code_blocks': r'```[\s\S]*?```'
        }
        
        source_structure = {k: [] for k in patterns}
        for item in source_items:
            if item.get('text'):
                lines = item['text'].split('\n')
                for line in lines:
                    for struct_type, pattern in patterns.items():
                        if re.match(pattern, line):
                            source_structure[struct_type].append(line)
        
        response_lines = response_text.split('\n')
        preserved_count = 0
        total_count = 0
        
        for struct_type, source_lines in source_structure.items():
            if source_lines:
                total_count += 1
                pattern = patterns[struct_type]
                response_matches = [line for line in response_lines 
                                  if re.match(pattern, line)]
                if response_matches:
                    preserved_count += 1
        
        return preserved_count / total_count if total_count > 0 else True

    # Remove references section for content checking
    main_content = response.split("References")[0] if "References" in response else response
    
    # Verify exact content matching
    verification_results["exact_content"] = check_exact_content(main_content, matched_items)
    
    # Verify format preservation
    verification_results["format_preserved"] = check_format_preservation(main_content, matched_items)
    
    # Enhanced hallucination check
    verification_results["no_hallucination"] = check_no_hallucination(main_content)
    
    # Verify structure
    verification_results["proper_structure"] = check_structure_preservation(main_content, matched_items)
    
    # Verify references
    verification_results["proper_references"] = check_references(response, matched_items)
    
    # Calculate overall validity (require 90% of checks to pass)
    is_valid = (sum(verification_results.values()) / len(verification_results)) >= 0.9
    
    # Generate specific feedback
    feedback = []
    if not verification_results["exact_content"]:
        feedback.append("Response must use exact content from documents without modification")
    if not verification_results["format_preserved"]:
        feedback.append("Original document formatting must be preserved (lists, code blocks, tables)")
    if not verification_results["no_hallucination"]:
        feedback.append("Remove all speculative language and external information - use only document facts")
    if not verification_results["proper_structure"]:
        feedback.append("Document structure and headings must be maintained exactly")
    if not verification_results["proper_references"]:
        feedback.append("All content must have proper reference citations")
        
    return is_valid, feedback if feedback else ["Response meets all criteria"]

def extract_numbers_and_dates(text):
    """Extract numerical values and dates with better pattern matching"""
    # Ensure UTF-8 encoding
    text = text.encode('utf-8').decode('utf-8', errors='replace')
    
    # Find numbers (including decimals and negative numbers)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    
    # Convert to float values
    return [float(num) for num in numbers]

def check_factual_consistency(response, matched_items):
    """Check numerical consistency with source material"""
    # Extract numbers from response
    response_numbers = extract_numbers_and_dates(response)
    
    # Extract numbers from source material
    source_numbers = []
    for item in matched_items:
        if item.get('text'):
            source_numbers.extend(extract_numbers_and_dates(item['text']))
    
    # Skip check if no numbers found
    if not response_numbers:
        return True
        
    # Check each response number exists in source
    return all(
        any(abs(rnum - snum) < 0.01 for snum in source_numbers)
        for rnum in response_numbers
    )