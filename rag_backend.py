import boto3
import tabula
import faiss
import json
import base64
import fitz as pymupdf
import os
import logging
import warnings
from botocore.exceptions import ClientError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock
import pickle
import re

# Constants
BASE_DIR = "data"
VECTOR_STORE = "vector_store"
FAISS_INDEX = "faiss.index"
ITEMS_PICKLE = "items.pkl"
QUERY_EMBEDDINGS_CACHE = "query_embeddings.pkl"

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def create_directories():
    """Create necessary directories for storing data"""
    dirs = [BASE_DIR, VECTOR_STORE]
    subdirs = ["images", "text", "tables", "page_images"]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(BASE_DIR, subdir), exist_ok=True)

def process_tables(doc, page_num, items, filepath):
    """Process tables with better table handling"""
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True)
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            # Skip empty tables
            if table.empty:
                continue
                
            # Clean table data
            table = table.fillna('')  # Handle NaN values
            
            # Create a more readable markdown table
            headers = table.columns.tolist()
            markdown_rows = []
            
            # Add headers
            markdown_rows.append("| " + " | ".join(str(h) for h in headers) + " |")
            markdown_rows.append("| " + " | ".join(['---' for _ in headers]) + " |")
            
            # Add data rows
            for _, row in table.iterrows():
                markdown_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            table_text = f"### Table {table_idx + 1}\n" + "\n".join(markdown_rows)
            
            table_file_name = os.path.join(BASE_DIR, "tables", 
                f"{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt")
                
            with open(table_file_name, 'w', encoding='utf-8') as f:
                f.write(table_text)
                
            items.append({
                "page": page_num,
                "type": "table",
                "text": table_text,
                "path": table_file_name,
                "raw_table": table.to_dict('records')
            })
    except Exception as e:
        logger.warning(f"Table processing error on page {page_num + 1}: {str(e)}")

def process_text_chunks(text, text_splitter, page_num, items, filepath):
    """Enhanced text processing with better structure preservation"""
    import re
    
    # Document structure patterns
    patterns = {
        'heading': re.compile(r'^(?:#{1,6}|\d+\.|[A-Z][^.]+:)\s*(.+)$', re.MULTILINE),
        'bullet_list': re.compile(r'^\s*[-*•]\s+(.+)$', re.MULTILINE),
        'numbered_list': re.compile(r'^\s*\d+\.\s+(.+)$', re.MULTILINE),
        'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
        'table': re.compile(r'^\s*\|(?:[^|]+\|)+\s*$', re.MULTILINE)
    }
    
    def extract_structure(text_block):
        """Extract structural elements while preserving their exact format"""
        elements = []
        current_element = {'type': 'text', 'content': []}
        lines = text_block.split('\n')
        code_block = False
        
        for line in lines:
            # Handle code blocks
            if line.startswith('```'):
                if code_block:
                    current_element['content'].append(line)
                    elements.append(current_element)
                    current_element = {'type': 'text', 'content': []}
                    code_block = False
                else:
                    if current_element['content']:
                        elements.append(current_element)
                    current_element = {'type': 'code', 'content': [line]}
                    code_block = True
                continue
                
            if code_block:
                current_element['content'].append(line)
                continue
                
            # Check for structural elements
            for elem_type, pattern in patterns.items():
                if pattern.match(line):
                    if current_element['content']:
                        elements.append(current_element)
                    current_element = {'type': elem_type, 'content': [line]}
                    break
            else:
                if line.strip():
                    current_element['content'].append(line)
                elif current_element['content']:
                    elements.append(current_element)
                    current_element = {'type': 'text', 'content': []}
        
        if current_element['content']:
            elements.append(current_element)
            
        return elements
    
    def save_element(element, section_num):
        """Save a structural element while preserving its format"""
        content = '\n'.join(element['content'])
        file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_{element['type']}_{page_num}_{section_num}.txt"
        
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)
            
        metadata = {
            'type': element['type'],
            'has_code': element['type'] == 'code',
            'has_list': element['type'] in ['bullet_list', 'numbered_list'],
            'has_table': element['type'] == 'table',
            'is_heading': element['type'] == 'heading'
        }
        
        return {
            "page": page_num,
            "type": "text",
            "text": content,
            "path": file_name,
            "metadata": metadata
        }
    
    try:
        # First extract structural elements
        elements = extract_structure(text)
        
        # Save elements while preserving their structure
        for i, element in enumerate(elements):
            item = save_element(element, i)
            items.append(item)
        
        # Process any remaining text traditionally
        remaining_text = text_splitter.split_text(text)
        for i, chunk in enumerate(remaining_text):
            # Only save chunks that aren't part of structural elements
            if not any(chunk in elem['content'] for elem in elements):
                text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
                with open(text_file_name, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                items.append({
                    "page": page_num,
                    "type": "text",
                    "text": chunk,
                    "path": text_file_name,
                    "metadata": {'type': 'text'}
                })
                
    except Exception as e:
        logger.error(f"Error processing text chunks on page {page_num}: {str(e)}")
        # Fall back to basic processing
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
            with open(text_file_name, 'w', encoding='utf-8') as f:
                f.write(chunk)
            items.append({
                "page": page_num,
                "type": "text",
                "text": chunk,
                "path": text_file_name
            })

def process_images(page, page_num, items, filepath, doc):
    """Process images from PDF pages"""
    images = page.get_images()
    for idx, image in enumerate(images):
        try:
            xref = image[0]
            pix = pymupdf.Pixmap(doc, xref)
            
            # Improve image quality by converting to RGB if needed
            if pix.n - pix.alpha < 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                
            image_name = os.path.join(BASE_DIR, "images", 
                f"{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png")
            
            # Save image without quality parameter
            pix.save(image_name)
            
            with open(image_name, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf8')
            items.append({
                "page": page_num,
                "type": "image",
                "path": image_name,
                "image": encoded_image
            })
        except Exception as e:
            logger.warning(f"Image processing error on page {page_num + 1}, image {idx}: {str(e)}")
            continue

def process_page_images(page, page_num, items, filepath):
    """Process full page images"""
    pix = page.get_pixmap()
    page_path = os.path.join(BASE_DIR, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

def process_pdf(uploaded_file):
    """Process uploaded PDF file and extract content"""
    if uploaded_file is None:
        return None, None
    
    filepath = os.path.join(BASE_DIR, uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    doc = pymupdf.open(filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
    items = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        process_tables(doc, page_num, items, filepath)
        process_text_chunks(text, text_splitter, page_num, items, filepath)
        process_images(page, page_num, items, filepath, doc)
        process_page_images(page, page_num, items, filepath)

    return items, filepath

def generate_multimodal_embeddings(prompt=None, image=None, output_embedding_length=384):
    """Generate embeddings using AWS Bedrock"""
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt, base64 image, or both as input")
    
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    model_id = "amazon.titan-embed-image-v1"
    
    body = {"embeddingConfig": {"outputEmbeddingLength": output_embedding_length}}
    if prompt:
        body["inputText"] = prompt
    if image:
        body["inputImage"] = image

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(response.get("body").read())
        return result.get("embedding")
    except ClientError as err:
        logger.error(f"Error generating embeddings: {str(err)}")
        return None

def load_or_initialize_stores():
    """Load or initialize vector store and cache with UTF-8 support"""
    embedding_vector_dimension = 384
    
    if os.path.exists(os.path.join(VECTOR_STORE, FAISS_INDEX)):
        index = faiss.read_index(os.path.join(VECTOR_STORE, FAISS_INDEX))
        with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), 'rb') as f:
            # Load with UTF-8 encoding handling
            all_items = pickle.load(f)
            # Ensure all text content is UTF-8
            for item in all_items:
                if 'text' in item:
                    item['text'] = item['text'].encode('utf-8').decode('utf-8', errors='replace')
    else:
        index = faiss.IndexFlatL2(embedding_vector_dimension)
        all_items = []
    
    query_cache_path = os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE)
    if os.path.exists(query_cache_path):
        with open(query_cache_path, 'rb') as f:
            query_embeddings_cache = pickle.load(f)
            # Ensure queries are UTF-8 encoded
            query_embeddings_cache = {
                k.encode('utf-8').decode('utf-8', errors='replace'): v 
                for k, v in query_embeddings_cache.items()
            }
    else:
        query_embeddings_cache = {}
    
    return index, all_items, query_embeddings_cache

def save_stores(index, all_items, query_embeddings_cache):
    """Save vector store and cache with UTF-8 support"""
    os.makedirs(VECTOR_STORE, exist_ok=True)
    
    faiss.write_index(index, os.path.join(VECTOR_STORE, FAISS_INDEX))
    
    # Ensure UTF-8 encoding for text content before saving
    items_to_save = []
    for item in all_items:
        item_copy = item.copy()
        if 'text' in item_copy:
            item_copy['text'] = item_copy['text'].encode('utf-8').decode('utf-8', errors='replace')
        items_to_save.append(item_copy)
    
    with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), 'wb') as f:
        pickle.dump(items_to_save, f)
    
    # Ensure UTF-8 encoding for cache before saving
    cache_to_save = {
        k.encode('utf-8').decode('utf-8', errors='replace'): v 
        for k, v in query_embeddings_cache.items()
    }
    
    with open(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE), 'wb') as f:
        pickle.dump(cache_to_save, f)

def invoke_claude_3_multimodal(prompt, matched_items):
    """Generate response using Claude 3 with strict document-only answers"""
    try:
        system_msg = [{
            "text": """You are a specialized documentation assistant that ONLY answers using the EXACT content from provided documents. Follow these rules with NO EXCEPTIONS:

1. Document Content:
   - ONLY use information explicitly present in the provided documents
   - DO NOT add any external information or knowledge
   - DO NOT make assumptions or generalizations
   - If information is not in the documents, clearly state that
   - Copy relevant text exactly as it appears in the documents

2. Document Structure:
   - Preserve ALL original formatting
   - Keep headings with their exact numbering and levels
   - Maintain bullet points and list numbering exactly
   - Preserve table formatting in markdown
   - Keep code blocks with their syntax and indentation

3. Response Structure:
   - Start with the most relevant section heading from the document
   - Quote the exact text under that heading
   - Use clear section breaks between different parts
   - Start lists and code blocks on new lines
   - Maintain original paragraph breaks

4. When Information is Missing:
   - State clearly: "I cannot find information about [specific topic] in the provided documents"
   - Do not speculate or suggest possibilities
   - Do not add qualifiers like "may", "might", or "probably"
   - Do not reference external sources

5. Technical Content:
   - Keep exact command syntax and parameters
   - Preserve all code formatting and comments
   - Maintain exact error messages and outputs
   - Keep file paths and configuration exactly as shown

6. References:
   - End with a "References" section
   - Format: "- **[Source: filename, page X]**"
   - Never cite sources inline in the text
   - List all documents used in the response

7. Formatting Rules:
   - Code blocks: Use ```language_name and ``` 
   - Lists: Keep original markers (-, *, numbers)
   - Tables: Preserve | and - formatting
   - Quotes: Use exact text inside quotation marks

8. Absolutely Forbidden:
   - Adding any information not in documents
   - Modifying code examples
   - Changing technical parameters
   - Paraphrasing technical instructions
   - Making assumptions about missing information
   - Using speculative language
   - Combining information in ways that alter meaning"""
        }]
        
        if not matched_items:
            return "I cannot find any information about this topic in the provided documents."
            
        # Organize matched items by section and relevance
        organized_content = []
        for item in matched_items:
            source_file = os.path.basename(item['path']).split('_')[0]
            source_info = f"[Source: {source_file}, page {item['page']+1}]"
            
            content_entry = {
                "source": source_info,
                "type": item['type']
            }
            
            if item['type'] == 'text':
                # Preserve section structure if present
                lines = item['text'].split('\n')
                heading_match = re.match(r'^(#+|\d+\.|\w+:)\s+(.+)$', lines[0]) if lines else None
                
                if heading_match:
                    content_entry["section"] = heading_match.group(2)
                    content_entry["heading_level"] = len(heading_match.group(1)) if '#' in heading_match.group(1) else 1
                    content_entry["text"] = '\n'.join(lines[1:])
                else:
                    content_entry["text"] = item['text']
                    
                # Check for special formatting
                content_entry["has_lists"] = any(line.strip().startswith(('- ', '* ', '• ', '1. ')) for line in lines)
                content_entry["has_code"] = bool(re.search(r'```[\s\S]*?```', item['text']))
                content_entry["has_tables"] = bool(re.search(r'\|(?:[^|]+\|)+', item['text']))
                
                organized_content.append(content_entry)
                
            elif item['type'] == 'table':
                content_entry["text"] = item['text']
                organized_content.append(content_entry)
                
            elif item['type'] in ['image', 'page']:
                content_entry["image"] = {
                    "format": "png",
                    "source": {"bytes": item['image']}
                }
                organized_content.append(content_entry)

        # Build message content preserving structure
        message_content = []
        
        # Group by sections
        sections = {}
        for content in organized_content:
            if 'section' in content:
                section_name = content['section']
                if section_name not in sections:
                    sections[section_name] = []
                sections[section_name].append(content)
            else:
                if 'General' not in sections:
                    sections['General'] = []
                sections['General'].append(content)

        # Add content by section
        for section_name, contents in sections.items():
            # Add section header
            if section_name != 'General':
                message_content.append({
                    "text": f"# {section_name}\n"
                })
            
            # Add section contents preserving format
            for content in contents:
                if 'text' in content:
                    # Preserve special formatting
                    text_with_format = content['text']
                    if content.get('has_code') or content.get('has_lists') or content.get('has_tables'):
                        text_with_format = '\n' + text_with_format + '\n'
                    
                    message_content.append({
                        "text": f"{text_with_format}\n{content['source']}"
                    })
                elif 'image' in content:
                    message_content.append({"image": content["image"]})
                    message_content.append({"text": f"[Image: {content['source']}]"})

        enhanced_prompt = f"""Question: {prompt}

Requirements for your response:
1. Use ONLY information from the provided documents
2. Use EXACT quotes and preserve ALL formatting
3. If information is missing, state that clearly
4. NEVER add external information or assumptions
5. Keep ALL original structure (lists, code blocks, tables)

Available sections: {', '.join(sections.keys())}"""

        inference_params = {
            "max_new_tokens": 1000,
            "top_p": 0.9,
            "top_k": 20,
            "temperature": 0.1,  # Very low temperature for deterministic responses
            "stop_sequences": []
        }
        
        message_list = [
            {"role": "user", "content": message_content},
            {"role": "user", "content": [{"text": enhanced_prompt}]}
        ]
        
        request_body = {
            "messages": message_list,
            "system": system_msg,
            "inferenceConfig": inference_params,
        }
        
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        client = ChatBedrock(model_id=model_id)
        
        response = client.invoke(json.dumps(request_body))
        return response.content
        
    except Exception as e:
        logger.error(f"Error invoking Claude-3: {str(e)}")
        return f"### Error\n{str(e)}\n\nPlease try again or contact support if the problem persists."

def clear_vector_store():
    """Clear all stored vectors and caches"""
    try:
        if os.path.exists(VECTOR_STORE):
            import shutil
            shutil.rmtree(VECTOR_STORE)
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")

def clear_history():
    """Clear the query history and cached responses"""
    try:
        if os.path.exists(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE)):
            os.remove(os.path.join(VECTOR_STORE, QUERY_EMBEDDINGS_CACHE))
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")

def check_aws_credentials():
    """Verify AWS credentials are properly configured"""
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            return False
        return True
    except Exception as e:
        logger.error(f"AWS configuration error: {str(e)}")
        return False