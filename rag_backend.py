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
    """Process text content from PDF pages with UTF-8 support"""
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w', encoding='utf-8') as f: 
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": text_file_name})

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
        # More restrictive system message
        system_msg = [{
            "text": """You are a document-focused question answering assistant. Follow these rules STRICTLY:
                    1. Answer questions ONLY using information found in the provided document context
                    2. If the answer cannot be found in the provided context, respond ONLY with: "I cannot answer this question based on the provided documents."
                    3. Do not make assumptions or add information beyond what's in the documents
                    4. Do not use any external knowledge
                    5. Keep responses focused and precise, using only facts from the documents
                    6. Use proper markdown formatting for any tables
                    7. NEVER include source citations within the response text
                    8. Include only a single "References" section at the very end listing all sources used
                       Format: References\\n- **[Source: filename, page X]**
                    9. If only partial information is available, specify what aspects of the question you can and cannot answer based on the documents"""
        }]
        
        # Check if we have any matched items
        if not matched_items:
            return "I cannot answer this question based on the provided documents."
            
        message_content = []
        for item in matched_items:
            source_file = os.path.basename(item['path']).split('_')[0]
            source_info = f"[Source: {source_file}, page {item['page']+1}]"
            
            if item['type'] == 'text':
                message_content.append({
                    "text": f"Text content: {item['text']}\n{source_info}"
                })
            elif item['type'] == 'table':
                message_content.append({
                    "text": f"Table content: {item['text']}\n{source_info}"
                })
            elif item['type'] in ['image', 'page']:
                message_content.append({
                    "image": {
                        "format": "png",
                        "source": {"bytes": item['image']},
                    }
                })
                message_content.append({
                    "text": f"[Image reference: {source_info}]"
                })

        enhanced_prompt = f"""Question: {prompt}

Answer the question using ONLY the information provided in the context. Follow these requirements strictly:
1. If you cannot find a complete answer in the provided context, state that clearly
2. Do not add any information beyond what's in the documents
3. Format the response in clear paragraphs with markdown
4. Include relevant quotes without source citations
5. NEVER include any source citations within the text of your response
6. Add ONLY ONE "References" section at the very end listing all sources used
   Format:
   References
   - **[Source: filename, page X]**"""

        inference_params = {
            "max_new_tokens": 1000,
            "top_p": 0.9,
            "top_k": 20,
            "temperature": 0.2,
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