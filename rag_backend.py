import logging
import warnings

# Configure root logger
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=Warning)

import tabula
import faiss
import base64
import fitz as pymupdf
import os
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io

# Load environment variables
load_dotenv()

# Constants
BASE_DIR = "data"
VECTOR_STORE = "vector_store"
FAISS_INDEX = "faiss.index"
ITEMS_PICKLE = "items.pkl"
QUERY_EMBEDDINGS_CACHE = "query_embeddings.pkl"

# Initialize embedding models (they will be loaded when first needed)
embedding_models = {
    'text_model': None,
    'image_model': None,
    'image_processor': None
}

def initialize_embedding_models():
    """Initialize the embedding models if they haven't been loaded yet"""
    if embedding_models['text_model'] is None:
        embedding_models['text_model'] = SentenceTransformer('all-mpnet-base-v2')
    
    if embedding_models['image_model'] is None:
        embedding_models['image_model'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        embedding_models['image_processor'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Move CLIP to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_models['image_model'].to(device)

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
            table = table.fillna('')  
            
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
        logger.warning(f"Error processing table: {str(e)}")

def process_text_chunks(text, text_splitter, page_num, items, filepath):
    """Process text content from PDF pages with UTF-8 support"""
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{BASE_DIR}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w', encoding='utf-8') as f:  # Add UTF-8 encoding
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
            logger.error(f"Error processing image: {str(e)}")
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

def preprocess_image(image_pil):
    """Preprocess image for CLIP"""
    # Ensure image is in RGB mode
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    return image_pil

def generate_multimodal_embeddings(prompt=None, image=None, output_embedding_length=768):
    """Generate embeddings using open source models"""
    if not prompt and not image:
        raise ValueError("Please provide either a text prompt, base64 image, or both as input")
    
    try:
        # Initialize models if needed
        initialize_embedding_models()
        
        if prompt and not image:
            # Text-only embedding using Sentence-BERT
            embedding = embedding_models['text_model'].encode(prompt, convert_to_numpy=True)
            
        elif image and not prompt:
            try:
                # Decode base64 image
                image_bytes = base64.b64decode(image)
                image_pil = Image.open(io.BytesIO(image_bytes))
                
                # Preprocess image
                image_pil = preprocess_image(image_pil)
                
                # Process image for CLIP
                device = next(embedding_models['image_model'].parameters()).device
                inputs = embedding_models['image_processor'](
                    images=image_pil, 
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": 224, "width": 224}
                ).to(device)
                
                # Generate embedding
                with torch.no_grad():
                    embedding = embedding_models['image_model'].get_image_features(**inputs)
                    embedding = embedding.cpu().numpy()[0]
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return None
                
        else:
            # Combined text and image embedding
            try:
                text_embedding = embedding_models['text_model'].encode(prompt, convert_to_numpy=True)
                
                # Process image
                image_bytes = base64.b64decode(image)
                image_pil = Image.open(io.BytesIO(image_bytes))
                image_pil = preprocess_image(image_pil)
                
                device = next(embedding_models['image_model'].parameters()).device
                inputs = embedding_models['image_processor'](
                    images=image_pil, 
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": 224, "width": 224}
                ).to(device)
                
                with torch.no_grad():
                    image_embedding = embedding_models['image_model'].get_image_features(**inputs)
                    image_embedding = image_embedding.cpu().numpy()[0]
                
                # Average the embeddings
                embedding = (text_embedding + image_embedding) / 2
            except Exception as e:
                logger.error(f"Error processing combined embedding: {str(e)}")
                return None
        
        # Ensure embedding is the correct shape and type
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (output_embedding_length,):
            logger.warning(f"Unexpected embedding shape: {embedding.shape}")
            return None
        
        # Normalize the embedding
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            normalized_embedding = embedding / embedding_norm
        else:
            logger.warning("Zero norm embedding encountered")
            return None
        
        return normalized_embedding
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return None

def load_or_initialize_stores():
    """Load or initialize vector store and cache with updated embedding dimension"""
    embedding_vector_dimension = 768  # New dimension for SBERT/CLIP embeddings
    
    if os.path.exists(os.path.join(VECTOR_STORE, FAISS_INDEX)):
        index = faiss.read_index(os.path.join(VECTOR_STORE, FAISS_INDEX))
        with open(os.path.join(VECTOR_STORE, ITEMS_PICKLE), 'rb') as f:
            all_items = pickle.load(f)
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
    """
    Generate response using GPT-4 with strict document-only answers
    
    Args:
        prompt (str): User's question or query
        matched_items (list): List of relevant document chunks with their metadata
        
    Returns:
        str: Generated response based on document context
    """
    try:
        # Initialize OpenAI chat model with specific parameters
        chat = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            presence_penalty=0,
            frequency_penalty=0
        )
        
        # Comprehensive system message for strict document-based responses
        system_msg = SystemMessage(content="""You are a document-focused question answering assistant. Follow these rules STRICTLY:
                1. Answer questions ONLY using information found in the provided document context
                2. If the answer cannot be found in the provided context, respond ONLY with: "I cannot answer this question based on the provided documents."
                3. Do not make assumptions or add information beyond what's in the documents
                4. Do not use any external knowledge
                5. Keep responses focused and precise, using only facts from the documents
                6. Use proper markdown formatting for any tables
                7. NEVER include source citations within the response text
                8. Include only a single "References" section at the very end listing all sources used
                   Format: References\\n- **[Source: filename, page X]**
                9. If only partial information is available, specify what aspects of the question you can and cannot answer based on the documents
                10. For tables, maintain their structure using markdown table formatting
                11. For numerical data, maintain precise values as shown in the documents
                12. If documents contain conflicting information, acknowledge the discrepancy and cite all sources""")
        
        # Handle case with no matched items
        if not matched_items:
            logger.info("No matching documents found for query: %s", prompt)
            return "I cannot answer this question based on the provided documents."
        
        # Process and structure matched items
        context_parts = []
        image_references = []
        table_contents = []
        
        for item in matched_items:
            try:
                source_file = os.path.basename(item['path']).split('_')[0]
                source_info = f"[Source: {source_file}, page {item['page']+1}]"
                
                if item['type'] == 'text':
                    # Process text content with UTF-8 encoding
                    text_content = item['text'].encode('utf-8').decode('utf-8', errors='replace')
                    context_parts.append(f"Text content: {text_content}\n{source_info}")
                    
                elif item['type'] == 'table':
                    # Handle table content separately to maintain structure
                    table_contents.append(f"Table content:\n{item['text']}\n{source_info}")
                    
                elif item['type'] in ['image', 'page']:
                    # Keep track of image references
                    image_references.append(f"[Image reference: {source_info}]")
                    
            except Exception as e:
                logger.error(f"Error processing item {item.get('path', 'unknown')}: {str(e)}")
                continue
        
        # Combine all context parts with proper separation
        full_context = "\n\n".join([
            *context_parts,
            *table_contents,
            *image_references
        ])
        
        # Construct enhanced prompt with comprehensive instructions
        enhanced_prompt = f"""Question: {prompt}

Context:
{full_context}

Answer the question using ONLY the information provided in the context above. Follow these requirements strictly:
1. If you cannot find a complete answer in the provided context, state that clearly
2. Do not add any information beyond what's in the documents
3. Format the response in clear paragraphs with markdown
4. Include relevant quotes without source citations
5. NEVER include any source citations within the text of your response
6. Add ONLY ONE "References" section at the very end listing all sources used
   Format:
   References
   - **[Source: filename, page X]**
7. For tables, maintain the exact structure and data
8. For numerical values, use exact figures from the documents
9. If you find conflicting information, acknowledge it and cite all relevant sources"""

        # Create message list for the chat model
        messages = [
            system_msg,
            HumanMessage(content=enhanced_prompt)
        ]
        
        # Log the request (excluding sensitive content)
        logger.info(f"Sending request to GPT-4 for query: {prompt[:100]}...")
        
        # Get response from OpenAI with timeout handling
        try:
            response = chat.invoke(messages)
            
            # Log successful response (length only)
            logger.info(f"Received response of length: {len(response.content)}")
            
            # Post-process response to ensure proper formatting
            processed_response = response.content.strip()
            
            # Ensure response ends with References section if sources were used
            if any(source_info in processed_response for item in matched_items 
                  for source_info in [os.path.basename(item['path']).split('_')[0]]):
                if "References" not in processed_response:
                    processed_response += "\n\nReferences\n"
                    for item in matched_items:
                        source_file = os.path.basename(item['path']).split('_')[0]
                        processed_response += f"- **[Source: {source_file}, page {item['page']+1}]**\n"
            
            return processed_response
            
        except Exception as e:
            logger.error(f"Error during GPT-4 API call: {str(e)}")
            return (f"### Error\nAn error occurred while generating the response: {str(e)}\n\n"
                   "Please try again or contact support if the problem persists.")
        
    except Exception as e:
        logger.error(f"Fatal error in invoke_claude_3_multimodal: {str(e)}")
        return "### Error\nA critical error occurred. Please try again later or contact support."

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
    """This function is kept for compatibility but always returns True now"""
    return True