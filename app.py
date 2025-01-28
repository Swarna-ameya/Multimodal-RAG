import streamlit as st
from rag_backend import (
    check_aws_credentials,
    create_directories,
    process_pdf,
    load_or_initialize_stores,
    generate_multimodal_embeddings,
    invoke_claude_3_multimodal,
    save_stores,
    clear_history
)
import numpy as np
import os
from pathlib import Path
import base64

def get_pdf_download_link(pdf_path):
    """Generate a download link for a PDF file"""
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    return f'data:application/pdf;base64,{b64_pdf}'

def add_searchable_pdf_viewer():
    """Add a searchable PDF viewer to the sidebar"""
    with st.sidebar:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                <svg width="24" height="24" viewBox="0 0 24 24" style="fill: #0066cc;">
                    <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                </svg>
                <h3 style="margin: 0;">Document Viewer</h3>
            </div>
        """, unsafe_allow_html=True)

        # Initialize session state for pagination
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        if 'items_per_page' not in st.session_state:
            st.session_state.items_per_page = 10

        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.endswith('.pdf')]
            if files:
                # Search box
                search_term = st.text_input("🔍 Search documents", key="doc_search").lower()
                
                # Filter files based on search
                filtered_files = [f for f in files if search_term in f.lower()] if search_term else files
                
                # Pagination
                total_pages = len(filtered_files) // st.session_state.items_per_page + (1 if len(filtered_files) % st.session_state.items_per_page > 0 else 0)
                
                # Display total count and current range
                start_idx = st.session_state.current_page * st.session_state.items_per_page
                end_idx = min(start_idx + st.session_state.items_per_page, len(filtered_files))
                
                st.markdown(f"""
                    <div style="margin: 10px 0;">
                        Showing {start_idx + 1}-{end_idx} of {len(filtered_files)} documents
                    </div>
                """, unsafe_allow_html=True)

                # Display paginated files with previews
                current_files = filtered_files[start_idx:end_idx]
                
                # Create columns for the file list and preview
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    selected_file = None
                    for file in current_files:
                        if st.button(
                            file, 
                            key=f"btn_{file}",
                            help="Click to view PDF",
                            use_container_width=True
                        ):
                            selected_file = file
                
                # Pagination controls
                col_prev, col_page, col_next = st.columns([1, 2, 1])
                
                with col_prev:
                    if st.button("← Previous", disabled=st.session_state.current_page == 0):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with col_page:
                    page_input = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state.current_page + 1,
                        key="page_input"
                    )
                    if page_input != st.session_state.current_page + 1:
                        st.session_state.current_page = page_input - 1
                        st.rerun()
                
                with col_next:
                    if st.button("Next →", disabled=st.session_state.current_page >= total_pages - 1):
                        st.session_state.current_page += 1
                        st.rerun()
                
                # PDF Preview
                with col2:
                    if selected_file:
                        st.markdown(f"**Preview:** {selected_file}")
                        with open(os.path.join("data", selected_file), "rb") as f:
                            pdf_bytes = f.read()
                        pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
                        pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_b64}" width="100%" height="600px" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    else:
                        st.info("Select a document to preview")

                # Items per page selector at the bottom
                st.selectbox(
                    "Items per page",
                    options=[10, 20, 50, 100],
                    key="items_per_page_selector",
                    index=[10, 20, 50, 100].index(st.session_state.items_per_page)
                )
                if st.session_state.items_per_page != st.session_state.items_per_page_selector:
                    st.session_state.items_per_page = st.session_state.items_per_page_selector
                    st.session_state.current_page = 0
                    st.rerun()
            else:
                st.info("No documents uploaded yet")
                    
def main():
    # Set page configuration for wider layout
    st.set_page_config(layout="wide")
    
    # Handle PDF file requests
    if 'data' in st.query_params:
        pdf_name = st.query_params['data']
        pdf_path = os.path.join("data", pdf_name)
        if os.path.exists(pdf_path):
            pdf_link = get_pdf_download_link(pdf_path)
            st.markdown(f'<iframe src="{pdf_link}" width="100%" height="800px"></iframe>', unsafe_allow_html=True)
            return
    
    # CSS for better layout and button styling
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
        
        /* Trash icon */
        .trash-icon {
            width: 16px;
            height: 16px;
            fill: currentColor;
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
        
        div h2 {
            color: #31333F;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        /* Icon container styles */
        div[style*="display: flex"] {
            margin: 1rem 0;
        }
        </style>
        
        <!-- Clear History Button with Trash Icon -->
        <button class="clear-button" onclick="window.location.href='?clear=1'">
            <svg class="trash-icon" viewBox="0 0 24 24">
                <path d="M19 6h-3.5l-1-1h-5l-1 1H5v2h14V6zM6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V8H6v11zm3.5-6.8l1.8 1.8L13 12.2l1.8 1.8 1.4-1.4-1.8-1.8 1.8-1.8-1.4-1.4L13 10.2l-1.8-1.8-1.4 1.4 1.8 1.8-1.8 1.8z"/>
            </svg>
            Clear History
        </button>
        """,
        unsafe_allow_html=True,
    )

    # Handle clear history
    if 'clear' in st.query_params:
        clear_history()
        st.session_state.chat_history = []
        st.query_params.clear()
        st.rerun()

    # Title with icon
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px;">
            <svg width="40" height="40" viewBox="0 0 24 24" style="fill: #0066cc;">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
            </svg>
            <h1 style="margin: 0; display: inline;">Multimodal RAG System</h1>
        </div>
        """, unsafe_allow_html=True)
    
    if not check_aws_credentials():
        st.stop()
    
    create_directories()

    # Sidebar for document management
    with st.sidebar:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px;">
                <svg width="30" height="30" viewBox="0 0 24 24" style="fill: #0066cc;">
                    <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/>
                </svg>
                <h2 style="margin: 0; display: inline;">Document Management</h2>
            </div>
            """, unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files"
        )
        
        if os.path.exists("data"):
            files = [f for f in os.listdir("data") if f.endswith('.pdf')]
            if files:
                st.subheader("Uploaded Documents:")
                for file in files:
                    st.text(file)

        add_searchable_pdf_viewer()

    # Load or initialize stores
    index, all_items, query_embeddings_cache = load_or_initialize_stores()

    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            items, filepath = process_pdf(uploaded_file)
            if items:
                st.success(f"Processed {uploaded_file.name}")
                
                with st.spinner("Generating embeddings..."):
                    for item in items:
                        if item['type'] in ['text', 'table']:
                            item['embedding'] = generate_multimodal_embeddings(prompt=item['text'])
                        else:
                            item['embedding'] = generate_multimodal_embeddings(image=item['image'])
                
                new_embeddings = np.array([item['embedding'] for item in items])
                index.add(np.array(new_embeddings, dtype=np.float32))
                all_items.extend(items)
                save_stores(index, all_items, query_embeddings_cache)

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            # User message
            with st.chat_message("user"):
                st.write(message["question"])
            # Assistant message
            with st.chat_message("assistant"):
                st.write(message["answer"])

    # Chat input
    if query := st.chat_input("Ask a question about your documents..."):
        # User message
        with st.chat_message("user"):
            st.write(query)

        # Generate response
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
                    
                    matched_items = [{k: v for k, v in all_items[idx].items() if k != 'embedding'} 
                                   for idx in result.flatten()]
                    
                    # Get response
                    response = invoke_claude_3_multimodal(query, matched_items)
                    st.write(response)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": response
                    })
                else:
                    st.warning("Please upload some documents first.")

if __name__ == "__main__":
    main()