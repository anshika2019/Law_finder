"""
ChromaDB connection and initialization
Automatically creates database if it doesn't exist
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
import csv
import re
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, 'ipc_sections.csv')
DB_DIR = os.path.join(BASE_DIR, 'chroma_langchain_db')

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def create_database(silent=False):
    """Create the ChromaDB database from CSV file"""
    if not silent:
        print("Creating ChromaDB database from CSV file...")
    
    # First, read CSV to get Section values mapped by row number
    section_data = {}
    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=0):
            section_data[idx] = row.get('Section', '').strip()

    # Load documents using CSVLoader
    loader = CSVLoader(
        file_path=CSV_FILE_PATH,
        csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['Description', 'Offense', 'Punishment', 'Section']
        }
    ) 
    section_loader = loader.load()

    # Add Section to metadata for each document
    for doc in section_loader:
        row_num = doc.metadata.get('row', -1)
        
        # Skip header row (row 0)
        if row_num == 0:
            continue
        
        # Try to get Section from CSV data first
        if row_num > 0 and (row_num - 1) in section_data:
            section_value = section_data[row_num - 1]
            if section_value and section_value.strip() and section_value != 'Section':
                doc.metadata['Section'] = section_value
                continue
        
        # Fallback: Extract Section from page_content
        patterns = [
            r'IPC\s*Section\s*(\d+)',
            r'section\s*(\d+)\s*of',
            r'IPC[_\s]*(\d+)',
            r'Section:\s*(IPC[_\s]*\d+)'
        ]
        
        for pattern in patterns:
            section_match = re.search(pattern, doc.page_content, re.IGNORECASE)
            if section_match:
                section_num = section_match.group(1)
                doc.metadata['Section'] = f'IPC_{section_num}'
                break

    # Filter out header row (row 0) before storing in database
    filtered_docs = [doc for doc in section_loader if doc.metadata.get('row', -1) != 0]

    # Create ChromaDB - use from_documents which handles persistence automatically
    chroma_vector_database = Chroma.from_documents(
        documents=filtered_docs,
        embedding=embeddings,
        collection_name="section_collection",
        persist_directory=DB_DIR,
    )
    if not silent:
        print(f"Database created successfully with {len(filtered_docs)} documents!")
    return chroma_vector_database

def get_database(silent=False):
    """Get the ChromaDB database, creating it if it doesn't exist"""
    # Check if database exists
    db_file = os.path.join(DB_DIR, 'chroma.sqlite3')
    if not os.path.exists(DB_DIR) or not os.path.exists(db_file):
        if not silent:
            print("Database not found. Creating new database...")
        return create_database(silent=silent)
    
    # Try to load existing database
    try:
        vector_store = Chroma(
            collection_name="section_collection",
            embedding_function=embeddings,
            persist_directory=DB_DIR,
        )
        # Test if database is accessible by trying to get collection
        try:
            _ = vector_store.get()
        except:
            # If get() fails, try a simple query
            retriever = vector_store.as_retriever(search_kwargs={"k": 1})
            _ = retriever.invoke("test")
        return vector_store
    except Exception as e:
        if not silent:
            print(f"Error loading existing database: {e}")
            print("Recreating database...")
        return create_database(silent=silent)

# Initialize database (lazy loading - only when imported)
chroma_vector_database = None

def _init_database():
    """Initialize database on first access"""
    global chroma_vector_database
    if chroma_vector_database is None:
        chroma_vector_database = get_database()
    return chroma_vector_database

# For backward compatibility, initialize on import
chroma_vector_database = get_database()

if __name__ == "__main__":
    # If run directly, just create/verify database
    print("Database initialized successfully!")
    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": 1})
    test_docs = retriever.invoke("test")
    print(f"Database contains {len(test_docs)} retrievable documents (sample query)")
