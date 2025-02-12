# Document Manager and Chat Interface with Multimodal Embeddings

This project comprises two key Streamlit applications:
1. `document_manager.py`: A document management system that allows users to upload, view, and process PDF files for information retrieval.
2. `chat_interface.py`: A chat-based interface where users can query processed documents and get answers based on embeddings and multimodal AI capabilities.

## Features

### Document Manager (`document_manager.py`)
- **Upload & Process PDFs**: Upload multiple PDF files to extract text, tables, and images.
- **Document Search**: Search uploaded documents by name.
- **View PDFs**: Display PDFs within the Streamlit app.
- **Multimodal Embedding Generation**: Generate text and image embeddings for content using AWS Bedrock.
- **Vector Store Management**: Store embeddings using FAISS for efficient similarity search.
- **Automatic Directory Management**: Creates necessary folders and files to store processed data.

### Chat Interface (`chat_interface.py`)
- **Query Documents**: Ask questions about the uploaded documents.
- **Chat History**: Retain and display chat history across sessions.
- **Multimodal Query Handling**: Generate embeddings for text-based queries and retrieve the most relevant sections from the document store.
- **AI-Powered Responses**: Use Claude AI to generate detailed and contextually accurate answers.
- **Session State Management**: Efficiently manage session states for chat and query caches.
- **Clear History**: Option to clear chat history and query cache.

---

## Setup and Installation

### Installation
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>


## Requirements

The following Python packages are required:
- boto3
- botocore
- pypdfium2
- faiss-cpu
- PyMuPDF
- langchain
- langchain-community
- langchain-aws
- tabula-py
- tqdm
- streamlit
- jpype1
- requests
- numpy
- ipython

### Install them using:
```bash
pip install -r requirements.txt
```

## Configure AWS credentials:
```bash
aws configure
```

## Run the applications:

### start document manager (terminal-1)
Upload files and preview document
```bash
streamlit run document_manager.py --server.fileWatcherType none
```

### start the chat interface (terminal-2)
```bash
streamlit run chat_interface.py --server.fileWatcherType none
```