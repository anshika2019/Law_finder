# IPC Section Query System

A RAG (Retrieval-Augmented Generation) system for querying Indian Penal Code (IPC) sections using ChromaDB vector database and ZenMux API with Gemini 2.5 Flash.

## Features

- **Vector Search**: Uses ChromaDB to store and retrieve relevant IPC sections
- **LLM Integration**: Uses ZenMux API with Gemini 2.5 Flash for intelligent answer generation
- **Semantic Search**: Finds relevant IPC sections based on query meaning, not just keywords
- **Template-based Fallback**: Provides structured answers even without API key
- **Automatic Setup**: Database is created automatically if it doesn't exist
- **Minimal Configuration**: Just install dependencies and run!

## Quick Start (Minimal Setup)

1. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

2. **(Optional) Add ZenMux API key to `.env` file for AI-powered answers:**
```bash
ZENMUX_API_KEY="your-api-key-here"
# or
ZenMux_KEY="your-api-key-here"
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

That's it! The database will be created automatically on first run.

## Manual Setup (Optional)

If you want to create the database manually before running the app:

```bash
python3 setup.py
```

Or run the connection script directly:

```bash
python3 chroma_conn.py
```

## Usage

### Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

The app will:
- Automatically create the database if it doesn't exist
- Load API key from `.env` file (no frontend input needed)
- Provide interactive chat interface
- Show source documents with proper Section numbers

### Command Line

```bash
# Simple Q&A
python3 simple_qa.py

# Main query interface
python3 relevant_doc.py

# Test ZenMux integration
python3 test_gemini.py
```

## Project Structure

- `app.py`: Streamlit web application (main interface)
- `chroma_conn.py`: Database initialization (auto-creates if missing)
- `simple_qa.py`: Q&A system with ZenMux integration
- `relevant_doc.py`: Command-line query interface
- `setup.py`: Manual setup script (optional)
- `ipc_sections.csv`: IPC sections data file
- `.env`: Environment variables (create this file with your API key)

## How It Works

1. **Automatic Database Creation**: On first run, the system automatically:
   - Checks if `chroma_langchain_db` exists
   - If not, loads CSV file and creates vector database
   - Extracts Section metadata from CSV
   - Stores documents with embeddings

2. **Query Processing**: 
   - User query is embedded using HuggingFace embeddings
   - Matched against stored documents using semantic search
   - Retrieves top K relevant IPC sections

3. **Answer Generation**: 
   - **With API Key**: Uses ZenMux API with Gemini 2.5 Flash for intelligent answers
   - **Without API Key**: Uses template-based extraction for structured answers

## Requirements

See `requirements.txt` for full list. Key dependencies:
- langchain-community
- langchain-chroma
- langchain-huggingface (for embeddings)
- chromadb
- sentence-transformers
- openai (for ZenMux client)
- python-dotenv
- streamlit

## Environment Variables

Create a `.env` file in the project root (optional, for AI-powered answers):

```
ZENMUX_API_KEY="your-zenmux-api-key"
# or
ZenMux_KEY="your-zenmux-api-key"
```

**Note**: The API key is only loaded from `.env` file. There is no frontend input for security.

## Sharing the Project

When sharing this project folder:

1. **Include these files:**
   - `ipc_sections.csv` (required)
   - All Python files
   - `requirements.txt`
   - `README.md`
   - `.gitignore`

2. **Exclude these (auto-generated):**
   - `chroma_langchain_db/` (will be created automatically)
   - `.env` (user should create their own)
   - `__pycache__/`

3. **Recipient setup:**
   ```bash
   pip3 install -r requirements.txt
   streamlit run app.py
   ```
   The database will be created automatically on first run!

## Troubleshooting

- **Database not found**: The app will automatically create it on first run
- **Section shows as N/A**: Delete `chroma_langchain_db` folder and restart the app to regenerate
- **API key not working**: Make sure it's in `.env` file as `ZENMUX_API_KEY` or `ZenMux_KEY`
