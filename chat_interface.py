import os
import streamlit as st
import logging
from response_verifier import verify_response
from rag_backend import (
    check_aws_credentials,
    load_or_initialize_stores,
    generate_multimodal_embeddings,
    invoke_claude_3_multimodal,
    save_stores
)
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def handle_clear_chat():
    """Clear only the chat history from current session"""
    st.session_state.chat_history = []
    st.rerun()

def generate_and_verify_response(query, matched_items):
    """Generate response and verify its quality, logging verification details"""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        # Generate initial response
        response = invoke_claude_3_multimodal(query, matched_items)
        
        # Verify response quality
        is_valid, feedback = verify_response(response, matched_items, query)
        
        # Log verification results
        logger.info(f"\n{'='*50}")
        logger.info(f"Response Verification - Attempt {attempt + 1}/{max_attempts}")
        logger.info(f"Query: {query}")
        logger.info(f"Validation Result: {'PASSED' if is_valid else 'FAILED'}")
        if not is_valid:
            logger.info("Feedback for improvement:")
            for point in feedback:
                logger.info(f"- {point}")
        logger.info(f"{'='*50}\n")
        
        if is_valid:
            return response
        
        # If invalid, regenerate with feedback
        attempt += 1
        if attempt < max_attempts:
            # Add feedback to the prompt for improvement but don't show in chat
            enhanced_query = f"{query}\n\nImprove the response considering: {'; '.join(feedback)}"
            query = enhanced_query
    
    # If all attempts failed, log warning but return best effort response
    logger.warning("Maximum verification attempts reached. Returning best effort response.")
    return response

def main():
    st.set_page_config(layout="wide", page_title="Chat Interface")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e6f3ff;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #f0f0f0;
            margin-right: 2rem;
        }
        
        /* Hide default Streamlit elements */
        div[data-testid="stToolbar"] {
            display: none;
        }
        button[title="View fullscreen"] {
            display: none;
        }
        
        /* Style for clear button */
        button[data-testid="clear_chat"] {
            background-color: #FF0000 !important;
            color: white !important;
        }
        button[data-testid="clear_chat"]:hover {
            background-color: #CC0000 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Handle clear chat from URL parameter
    if 'clear_chat' in st.query_params:
        handle_clear_chat()
        st.query_params.clear()
    
    # Title and header area with clear button
    col1, col2 = st.columns([6,1])
    with col1:
        st.title("Chat Interface")
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_chat", help="Clear chat history"):
            handle_clear_chat()
    
    # Check AWS credentials
    if not check_aws_credentials():
        st.error("AWS credentials not properly configured.")
        st.stop()
    
    # Load stores (but don't initialize new ones)
    index, all_items, query_embeddings_cache = load_or_initialize_stores()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
    
    # Chat input and processing
    if query := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if all_items:
                    # Generate query embedding
                    if query in query_embeddings_cache:
                        query_embedding = query_embeddings_cache[query]
                    else:
                        query_embedding = generate_multimodal_embeddings(prompt=query)
                        query_embeddings_cache[query] = query_embedding
                        save_stores(index, all_items, query_embeddings_cache)
                    
                    # Search for relevant content
                    distances, result = index.search(
                        np.array(query_embedding, dtype=np.float32).reshape(1,-1), 
                        k=5
                    )
                    
                    # Get matched items
                    matched_items = [{k: v for k, v in all_items[idx].items() if k != 'embedding'} 
                                   for idx in result.flatten()]
                    
                    # Generate and verify response
                    response = generate_and_verify_response(query, matched_items)
                    st.write(response)
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": response
                    })
                else:
                    st.warning("Please upload some documents first.")

if __name__ == "__main__":
    main()