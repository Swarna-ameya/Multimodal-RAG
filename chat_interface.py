import streamlit as st
from rag_backend import (
    check_aws_credentials,
    load_or_initialize_stores,
    generate_multimodal_embeddings,
    invoke_claude_3_multimodal,
    save_stores,
    clear_history
)
import numpy as np

def main():
    # Set page configuration
    st.set_page_config(layout="wide")
    
    # Add custom CSS for chat styling
    st.markdown(
        """
        <style>
        /* Chat message styles */
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
        .message-header {
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        /* Clear History button styles */
        .clear-button {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background-color: #ff4b4b;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: background-color 0.3s;
        }
        
        .clear-button:hover {
            background-color: #ff3333;
        }
        
        /* Hide default Streamlit elements */
        div[data-testid="stToolbar"] {
            display: none;
        }
        button[title="View fullscreen"] {
            display: none;
        }
        
        /* Title styles */
        div h1 {
            color: #31333F;
            margin: 0 !important;
        }
        </style>
        
        <!-- Clear History Button with Trash Icon -->
        <button class="clear-button" onclick="window.location.href='?clear=1'">
            <svg viewBox="0 0 24 24" width="16" height="16" style="fill: currentColor;">
                <path d="M19 6h-3.5l-1-1h-5l-1 1H5v2h14V6zM6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V8H6v11z"/>
            </svg>
            Clear History
        </button>
        """,
        unsafe_allow_html=True,
    )

    # Title with icon
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px;">
            <svg width="40" height="40" viewBox="0 0 24 24" style="fill: #0066cc;">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
            </svg>
            <h1>Chat Interface</h1>
        </div>
        """, unsafe_allow_html=True)
    
    # Check AWS credentials
    if not check_aws_credentials():
        st.error("AWS credentials not properly configured.")
        st.stop()
    
    # Handle clear history
    if 'clear' in st.query_params:
        clear_history()
        st.session_state.chat_history = []
        st.query_params.clear()
        st.rerun()

    # Load or initialize stores
    index, all_items, query_embeddings_cache = load_or_initialize_stores()

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.write(query)

        # Generate and display assistant response
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
                    
                    # Get matched items without embeddings
                    matched_items = [{k: v for k, v in all_items[idx].items() if k != 'embedding'} 
                                   for idx in result.flatten()]
                    
                    # Get response from Claude
                    response = invoke_claude_3_multimodal(query, matched_items)
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