# import os
# import uuid
# import time
# import streamlit as st
# import chromadb
# from dotenv import load_dotenv
# from pypdf import PdfReader
# import google.generativeai as genai
# from chromadb import Documents, EmbeddingFunction, Embeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from typing import List, Tuple, Dict
# import re
# from langchain.document_loaders import PyPDFLoader

# # Load environment variables from .env file
# load_dotenv()

# # Set page config for a wider layout
# st.set_page_config(
#     page_title="PDF RAG Assistant",
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
# <style>
#     .main {
#         background-color: #f5f7f9;
#     }
#     .stApp {
#         max-width: 1200px;
#         margin: 0 auto;
#     }
#     .chat-message {
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         margin-bottom: 1rem;
#         display: flex;
#         box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
#     }
#     .chat-message.user {
#         background-color: #e6f3ff;
#         border-left: 5px solid #2e86de;
#     }
#     .chat-message.bot {
#         background-color: #f0f5ff;
#         border-left: 5px solid #5c7cfa;
#     }
#     .stButton > button {
#         background-color: #2e86de;
#         color: white;
#         border: none;
#         border-radius: 4px;
#         padding: 0.5rem 1rem;
#         font-weight: 600;
#     }
#     .stButton > button:hover {
#         background-color: #1c6dc9;
#     }
#     div[data-testid="stFileUploader"] {
#         border: 2px dashed #ccc;
#         border-radius: 8px;
#         padding: 20px;
#         margin-bottom: 15px;
#     }
#     div[data-testid="stFileUploader"]:hover {
#         border-color: #2e86de;
#     }
#     .stProgress > div > div {
#         background-color: #2e86de;
#     }
#     .title-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         margin-bottom: 2rem;
#     }
#     .title-text {
#         font-size: 2.5rem;
#         font-weight: 700;
#         color: #2e86de;
#         text-align: center;
#     }
#     .subtitle-text {
#         font-size: 1.2rem;
#         color: #777;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Custom GeminiEmbeddingFunction for ChromaDB
# class GeminiEmbeddingFunction(EmbeddingFunction):
#     """
#     Custom embedding function using the Gemini AI API for document retrieval.
#     """
#     def __call__(self, input: Documents) -> Embeddings:
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         if not gemini_api_key:
#             raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
#         genai.configure(api_key=gemini_api_key)
#         model = "models/embedding-001"
#         title = "Custom query"
#         return genai.embed_content(model=model,
#                                   content=input,
#                                   task_type="retrieval_document",
#                                   title=title)["embedding"]

# def clean_pdf(pdf_content: bytes, min_word_count: int = 20) -> str:
#     """
#     Reads a PDF content, removes pages with fewer than the specified word count,
#     and returns the cleaned content.
    
#     Args:
#         pdf_content (bytes): The uploaded PDF content
#         min_word_count (int): Minimum word count for a page to be included (default: 20)
        
#     Returns:
#         str: The cleaned content from the PDF
#     """
#     # Create a temporary file path
#     temp_file_path = f"temp_{uuid.uuid4()}.pdf"
    
#     try:
#         # Write the uploaded content to a temporary file
#         with open(temp_file_path, "wb") as f:
#             f.write(pdf_content)
        
#         # Load the PDF using PyPDFLoader
#         loader = PyPDFLoader(temp_file_path)
#         pages = loader.load_and_split()
        
#         # Process each page
#         cleaned_content = ""
#         for page in pages:
#             page_content = page.page_content.strip()
#             words = re.findall(r'\b\w+\b', page_content)
#             word_count = len(words)
            
#             # Decide whether to keep or remove the page
#             if word_count >= min_word_count:
#                 cleaned_content += page_content + "\n\n"
        
#         return cleaned_content
    
#     finally:
#         # Clean up the temporary file
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)

# def create_chroma_db(documents: List[str], collection_name: str) -> chromadb.Collection:
#     """
#     Creates a Chroma database with the provided documents.
    
#     Args:
#         documents (List[str]): List of text chunks to embed.
#         collection_name (str): Name of the Chroma collection.
    
#     Returns:
#         chromadb.Collection: Chroma collection object.
#     """
#     # Create a temporary directory for the Chroma DB
#     db_path = "vectordb"
#     os.makedirs(db_path, exist_ok=True)
    
#     # Create the client and collection
#     chroma_client = chromadb.PersistentClient(path=db_path)
    
#     # Try to get the collection if it exists, otherwise create it
#     try:
#         collection = chroma_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())
#         # If collection exists, delete all documents
#         all_ids = collection.get()['ids']
#         if all_ids:
#             collection.delete(ids=all_ids)
#     except Exception:
#         # Create new collection if it doesn't exist
#         collection = chroma_client.create_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())
    
#     # Add documents to the collection
#     for i, d in enumerate(documents):
#         collection.add(documents=[d], ids=str(i))
    
#     return collection

# def embed_query(query: str) -> List[float]:
#     """
#     Embeds the query using Gemini AI with task_type="retrieval_query".
    
#     Args:
#         query (str): The query text.
    
#     Returns:
#         List[float]: The embedded query vector.
#     """
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         raise ValueError("Gemini API Key not provided.")
#     genai.configure(api_key=gemini_api_key)
#     model = "models/embedding-001"
#     return genai.embed_content(model=model,
#                               content=query,
#                               task_type="retrieval_query")["embedding"]

# def get_relevant_passage(query: str, collection, n_results: int = 3) -> List[str]:
#     """
#     Retrieves the most relevant text chunks from ChromaDB based on the query.
    
#     Args:
#         query (str): User's question.
#         collection: Chroma collection object.
#         n_results (int): Number of top results to retrieve.
    
#     Returns:
#         List[str]: List of relevant text chunks.
#     """
#     query_embedding = embed_query(query)
#     results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
#     return results['documents'][0]

# def make_rag_prompt(query: str, relevant_passage: str) -> str:
#     """
#     Creates a prompt for the Gemini model using the query and relevant text.
    
#     Args:
#         query (str): User's question.
#         relevant_passage (str): Retrieved text to include in the prompt.
    
#     Returns:
#         str: Formatted prompt string.
#     """
#     escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
#     prompt = f"""You are a helpful and informative assistant that answers questions using text from the reference passage below. 
#     Respond in complete sentences, be comprehensive, and include all relevant information. 
#     Use a friendly, conversational tone and break down complicated concepts.
#     If the passage doesn't contain enough information to answer the question, say you don't have enough info to provide a full answer.
    
#     QUESTION: '{query}'
#     PASSAGE: '{escaped}'

#     ANSWER:
#     """
#     return prompt

# def gemini_answer(prompt: str) -> str:
#     """
#     Generates an answer using the Gemini AI model.
    
#     Args:
#         prompt (str): Formatted prompt string.
    
#     Returns:
#         str: Generated answer text.
#     """
#     gemini_api_key = os.getenv("GEMINI_API_KEY")
#     if not gemini_api_key:
#         raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
#     genai.configure(api_key=gemini_api_key)
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     answer = model.generate_content(prompt)
#     return answer.text

# def generate_answer(collection, query: str) -> str:
#     """
#     Generates an answer to the query using the RAG pipeline.
    
#     Args:
#         collection: Chroma collection object.
#         query (str): User's question.
    
#     Returns:
#         str: Generated answer.
#     """
#     relevant_text = get_relevant_passage(query, collection, n_results=3)
#     prompt = make_rag_prompt(query, " ".join(relevant_text))
#     answer = gemini_answer(prompt)
#     return answer

# def main():
#     # Header
#     st.markdown('<div class="title-container"><span class="title-text">📚 PDF RAG Assistant</span></div>', unsafe_allow_html=True)
#     st.markdown('<p class="subtitle-text">Upload a PDF and ask questions about its content</p>', unsafe_allow_html=True)
    
#     # Initialize session state for storing conversation history
#     if 'conversation' not in st.session_state:
#         st.session_state.conversation = []
#     if 'pdf_processed' not in st.session_state:
#         st.session_state.pdf_processed = False
#     if 'collection' not in st.session_state:
#         st.session_state.collection = None
#     if 'pdf_name' not in st.session_state:
#         st.session_state.pdf_name = None
    
#     # Create two columns for the main layout
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.subheader("📄 Upload PDF")
        
#         # File uploader
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
#         # Process the uploaded PDF
#         if uploaded_file is not None and (not st.session_state.pdf_processed or uploaded_file.name != st.session_state.pdf_name):
#             st.session_state.pdf_name = uploaded_file.name
            
#             # Display processing status
#             with st.status("Processing PDF...", expanded=True) as status:
#                 st.write("Reading PDF...")
#                 pdf_content = uploaded_file.read()
                
#                 st.write("Cleaning PDF content...")
#                 pdf_text = clean_pdf(pdf_content)
                
#                 st.write("Splitting into chunks...")
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size=1000,
#                     chunk_overlap=100,
#                     length_function=len,
#                 )
#                 chunked_text = text_splitter.split_text(pdf_text)
#                 chunked_text = [chunk for chunk in chunked_text if len(chunk.strip()) > 50]
#                 st.write(f"Created {len(chunked_text)} chunks.")
                
#                 st.write("Creating embeddings...")
#                 collection_name = f"pdf_{uuid.uuid4().hex[:8]}"
#                 st.session_state.collection = create_chroma_db(documents=chunked_text, collection_name=collection_name)
                
#                 st.write("PDF processed successfully!")
#                 status.update(label="PDF processed successfully! ✅", state="complete")
                
#                 # Reset conversation when a new PDF is uploaded
#                 st.session_state.conversation = []
#                 st.session_state.pdf_processed = True
        
#         # Display current PDF info
#         if st.session_state.pdf_processed and st.session_state.pdf_name:
#             st.success(f"Active PDF: {st.session_state.pdf_name}")
            
#             # Clear PDF button
#             if st.button("Clear PDF"):
#                 st.session_state.pdf_processed = False
#                 st.session_state.collection = None
#                 st.session_state.pdf_name = None
#                 st.session_state.conversation = []
#                 st.rerun()
    
#     with col2:
#         st.subheader("💬 Ask Questions")
        
#         # Display conversation history
#         for i, (role, text) in enumerate(st.session_state.conversation):
#             if role == "user":
#                 st.markdown(f'<div class="chat-message user"><div>{text}</div></div>', unsafe_allow_html=True)
#             else:
#                 st.markdown(f'<div class="chat-message bot"><div>{text}</div></div>', unsafe_allow_html=True)
        
#         # Input for user question
#         question = st.text_input("Type your question here:", key="user_question", disabled=not st.session_state.pdf_processed)
        
#         # Generate answer when user submits a question
#         if question and st.session_state.pdf_processed:
#             # Add user question to conversation
#             st.session_state.conversation.append(("user", question))
            
#             # Display typing indicator
#             with st.status("Generating answer...", expanded=True) as status:
#                 # Get answer from RAG pipeline
#                 answer = generate_answer(st.session_state.collection, question)
#                 time.sleep(0.5)  # Small delay for better UX
#                 status.update(label="Answer generated! ✅", state="complete")
            
#             # Add assistant response to conversation
#             st.session_state.conversation.append(("assistant", answer))
            
#             # Clear the input field
#             st.rerun()
        
#         # Show instruction if no PDF is uploaded
#         if not st.session_state.pdf_processed:
#             st.info("Please upload a PDF document to start asking questions.")
        
#         # Clear conversation button
#         if st.session_state.conversation:
#             if st.button("Clear Conversation"):
#                 st.session_state.conversation = []
#                 st.rerun()

# if __name__ == "__main__":
#     main()

import os
import uuid
import time
import streamlit as st
import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple, Dict
import re
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

# Set page config for a wider layout with dark theme
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/yourusername',
        'Report a bug': "https://github.com/yourusername/issues",
        'About': "# PDF RAG Assistant\nThis app allows you to ask questions about your PDF documents."
    }
)

# Custom CSS for styling with dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.2);
    }
    .chat-message.user {
        background-color: #2c3e50;
        border-left: 5px solid #3498db;
    }
    .chat-message.bot {
        background-color: #1c2833;
        border-left: 5px solid #2980b9;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #34495e;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #3498db;
    }
    .stProgress > div > div {
        background-color: #3498db;
    }
    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 2rem;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3498db;
        text-align: center;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #bdc3c7;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Make input textbox stand out */
    div[data-baseweb="input"] {
        background-color: #1c2833;
        border-radius: 8px;
        border: 1px solid #34495e;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #3498db;
    }
    /* Status indicator styling */
    div[data-testid="stStatus"] {
        background-color: #1c2833;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Custom GeminiEmbeddingFunction for ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.
    """
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                  content=input,
                                  task_type="retrieval_document",
                                  title=title)["embedding"]

def clean_pdf(pdf_content: bytes, min_word_count: int = 20) -> str:
    """
    Reads a PDF content, removes pages with fewer than the specified word count,
    and returns the cleaned content.
    
    Args:
        pdf_content (bytes): The uploaded PDF content
        min_word_count (int): Minimum word count for a page to be included (default: 20)
        
    Returns:
        str: The cleaned content from the PDF
    """
    # Create a temporary file path
    temp_file_path = f"temp_{uuid.uuid4()}.pdf"
    
    try:
        # Write the uploaded content to a temporary file
        with open(temp_file_path, "wb") as f:
            f.write(pdf_content)
        
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load_and_split()
        
        # Process each page
        cleaned_content = ""
        for page in pages:
            page_content = page.page_content.strip()
            words = re.findall(r'\b\w+\b', page_content)
            word_count = len(words)
            
            # Decide whether to keep or remove the page
            if word_count >= min_word_count:
                cleaned_content += page_content + "\n\n"
        
        return cleaned_content
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def create_chroma_db(documents: List[str], collection_name: str) -> chromadb.Collection:
    """
    Creates a Chroma database with the provided documents.
    
    Args:
        documents (List[str]): List of text chunks to embed.
        collection_name (str): Name of the Chroma collection.
    
    Returns:
        chromadb.Collection: Chroma collection object.
    """
    # Create a temporary directory for the Chroma DB
    db_path = "vectordb"
    os.makedirs(db_path, exist_ok=True)
    
    # Create the client and collection
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Try to get the collection if it exists, otherwise create it
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())
        # If collection exists, delete all documents
        all_ids = collection.get()['ids']
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception:
        # Create new collection if it doesn't exist
        collection = chroma_client.create_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction())
    
    # Add documents to the collection in batches to avoid timeouts
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_ids = [str(j) for j in range(i, i+len(batch_docs))]
        collection.add(documents=batch_docs, ids=batch_ids)
        # Small delay to avoid overloading the API
        time.sleep(0.5)
    
    return collection

def embed_query(query: str) -> List[float]:
    """
    Embeds the query using Gemini AI with task_type="retrieval_query".
    
    Args:
        query (str): The query text.
    
    Returns:
        List[float]: The embedded query vector.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided.")
    genai.configure(api_key=gemini_api_key)
    model = "models/embedding-001"
    return genai.embed_content(model=model,
                              content=query,
                              task_type="retrieval_query")["embedding"]

def get_relevant_passage(query: str, collection, n_results: int = 3) -> List[str]:
    """
    Retrieves the most relevant text chunks from ChromaDB based on the query.
    
    Args:
        query (str): User's question.
        collection: Chroma collection object.
        n_results (int): Number of top results to retrieve.
    
    Returns:
        List[str]: List of relevant text chunks.
    """
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results['documents'][0]

def make_rag_prompt(query: str, relevant_passage: str) -> str:
    """
    Creates a prompt for the Gemini model using the query and relevant text.
    
    Args:
        query (str): User's question.
        relevant_passage (str): Retrieved text to include in the prompt.
    
    Returns:
        str: Formatted prompt string.
    """
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative assistant that answers questions using text from the reference passage below. 
    Respond in complete sentences, be comprehensive, and include all relevant information. 
    Use a friendly, conversational tone and break down complicated concepts.
    If the passage doesn't contain enough information to answer the question, say you don't have enough info to provide a full answer.
    
    QUESTION: '{query}'
    PASSAGE: '{escaped}'

    ANSWER:
    """
    return prompt

def gemini_answer(prompt: str) -> str:
    """
    Generates an answer using the Gemini AI model.
    
    Args:
        prompt (str): Formatted prompt string.
    
    Returns:
        str: Generated answer text.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    
    genai.configure(api_key=gemini_api_key)
    
    # Safety measure to prevent infinite loops
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')  # Using stable model version
        response = model.generate_content(prompt, timeout=30)  # Adding timeout
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try a different question or check your API key."

def generate_answer(collection, query: str) -> str:
    """
    Generates an answer to the query using the RAG pipeline.
    
    Args:
        collection: Chroma collection object.
        query (str): User's question.
    
    Returns:
        str: Generated answer.
    """
    try:
        relevant_text = get_relevant_passage(query, collection, n_results=3)
        prompt = make_rag_prompt(query, " ".join(relevant_text))
        answer = gemini_answer(prompt)
        return answer
    except Exception as e:
        return f"Error processing your question: {str(e)}. Please try again with a different question."

def main():
    # Header
    st.markdown('<div class="title-container"><span class="title-text">📚 PDF RAG Assistant</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Upload a PDF and ask questions about its content</p>', unsafe_allow_html=True)
    
    # Initialize session state for storing conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Create a two-column layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown("### 📄 Upload PDF")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", 
                                        help="Upload your PDF document (Max size: 200MB)")
        
        # Process the uploaded PDF
        if uploaded_file is not None and (not st.session_state.pdf_processed or uploaded_file.name != st.session_state.pdf_name) and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.pdf_name = uploaded_file.name
            
            # Display processing status
            with st.status("Processing PDF...", expanded=True) as status:
                try:
                    st.write("Reading PDF...")
                    pdf_content = uploaded_file.read()
                    
                    st.write("Cleaning PDF content...")
                    pdf_text = clean_pdf(pdf_content)
                    
                    st.write("Splitting into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100,
                        length_function=len,
                    )
                    chunked_text = text_splitter.split_text(pdf_text)
                    chunked_text = [chunk for chunk in chunked_text if len(chunk.strip()) > 50]
                    st.write(f"Created {len(chunked_text)} chunks.")
                    
                    st.write("Creating embeddings...")
                    collection_name = f"pdf_{uuid.uuid4().hex[:8]}"
                    st.session_state.collection = create_chroma_db(documents=chunked_text, collection_name=collection_name)
                    
                    st.write("PDF processed successfully!")
                    status.update(label="PDF processed successfully! ✅", state="complete")
                    
                    # Reset conversation when a new PDF is uploaded
                    st.session_state.conversation = []
                    st.session_state.pdf_processed = True
                    st.session_state.processing = False
                except Exception as e:
                    status.update(label=f"Error: {str(e)}", state="error")
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state.processing = False
        
        # Display current PDF info
        if st.session_state.pdf_processed and st.session_state.pdf_name:
            st.success(f"Active PDF: {st.session_state.pdf_name}")
            
            # PDF info container
            with st.container():
                st.markdown(f"""
                <div style="background-color: #1c2833; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 5px solid #27ae60;">
                    <h4 style="color: #2ecc71;">PDF Ready for Questions</h4>
                    <p style="color: #bdc3c7; font-size: 0.9rem;">You can now ask questions about this document in the chat panel.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clear PDF button
            if st.button("Clear PDF"):
                st.session_state.pdf_processed = False
                st.session_state.collection = None
                st.session_state.pdf_name = None
                st.session_state.conversation = []
                st.rerun()
    
    with right_col:
        st.markdown("### 💬 Ask Questions")
        
        # Chat container
        chat_container = st.container(height=500)
        
        with chat_container:
            # Display conversation history
            for i, (role, text) in enumerate(st.session_state.conversation):
                if role == "user":
                    st.markdown(f'<div class="chat-message user"><div>{text}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot"><div>{text}</div></div>', unsafe_allow_html=True)
        
        # Input for user question
        with st.container():
            question = st.text_input(
                "Type your question here:", 
                key="user_question", 
                disabled=not st.session_state.pdf_processed,
                placeholder="What would you like to know about this document?"
            )
            
            # Generate answer when user submits a question
            if question and st.session_state.pdf_processed and not st.session_state.processing:
                # Add user question to conversation
                st.session_state.conversation.append(("user", question))
                
                # Display typing indicator
                with st.status("Generating answer...", expanded=True) as status:
                    try:
                        # Get answer from RAG pipeline with a timeout
                        answer = generate_answer(st.session_state.collection, question)
                        time.sleep(0.5)  # Small delay for better UX
                        status.update(label="Answer generated! ✅", state="complete")
                        
                        # Add assistant response to conversation
                        st.session_state.conversation.append(("assistant", answer))
                        
                        # Clear the input field
                        st.session_state.user_question = ""
                        st.experimental_rerun()
                    except Exception as e:
                        status.update(label=f"Error: {str(e)}", state="error")
                        st.error(f"Something went wrong: {str(e)}")
                        # Still clear the input
                        st.session_state.user_question = ""
            
        # Show instruction if no PDF is uploaded
        if not st.session_state.pdf_processed:
            st.info("Please upload a PDF document to start asking questions.")
        
        # Clear conversation button
        col1, col2 = st.columns([2, 1])
        with col2:
            if st.session_state.conversation:
                if st.button("Clear Chat"):
                    st.session_state.conversation = []
                    st.rerun()

if __name__ == "__main__":
    main()