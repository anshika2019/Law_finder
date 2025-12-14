"""
Streamlit frontend for IPC Section Query System
"""
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if database exists, create if not
try:
    from chroma_conn import get_database
    # Ensure database is initialized (will create if doesn't exist, silently)
    chroma_vector_database = get_database(silent=True)
except Exception as e:
    st.error(f"Error loading database: {e}")
    st.info("Please ensure ipc_sections.csv exists in the project directory.")
    st.stop()

from simple_qa import answer_with_llm, answer_with_template

# Page configuration
st.set_page_config(
    page_title="IPC Section Query System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stMarkdown {
        line-height: 1.8;
    }
    .stMarkdown strong {
        color: #1f77b4;
        font-weight: 700;
    }
    .source-doc {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# Get API key from environment only (no frontend input)
zenmux_key = os.getenv("ZENMUX_API_KEY") or os.getenv("ZenMux_KEY")

# Header
st.markdown('<h1 class="main-header">⚖️ IPC Section Query System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about Indian Penal Code sections</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key status
    if zenmux_key:
        st.success("✅ API Key loaded from .env file")
    else:
        st.warning("⚠️ No API key found")
        st.info("""
        To use AI-powered answers, add your ZenMux API key to `.env` file:
        
        ```
        ZENMUX_API_KEY="your-api-key-here"
        ```
        
        Or use template-based answers (no API key needed).
        """)
    
    st.divider()
    
    # Model selection
    model_option = st.selectbox(
        "Select Model",
        options=[
            "google/gemini-2.5-flash",
            "google/gemini-2.0-flash-exp",
            "google/gemini-1.5-flash",
            "openai/gpt-4",
            "openai/gpt-3.5-turbo",
            "openai/gpt-4-turbo",
        ],
        index=0,
        help="Choose the LLM model to use for answering questions"
    )
    
    st.divider()
    
    # Number of documents to retrieve
    num_docs = st.slider(
        "Number of Documents to Retrieve",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of relevant IPC sections to retrieve for context"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info section
    st.info("""
    **How to use:**
    1. Add ZenMux API key to `.env` file (optional)
    2. Select a model
    3. Type your question about IPC sections
    4. Get intelligent answers with source documents
    """)
    
    st.divider()
    
    # Database status
    import os
    db_exists = os.path.exists("chroma_langchain_db/chroma.sqlite3")
    if db_exists:
        st.success("✅ Database ready")
    else:
        st.warning("⚠️ Database not found")
        if st.button("🔄 Create Database"):
            with st.spinner("Creating database... This may take a minute."):
                try:
                    from chroma_conn import create_database
                    create_database()
                    st.success("Database created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating database: {e}")

# Main chat interface
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show source documents if available
        if "source_docs" in message and message["source_docs"]:
            with st.expander("📄 View Source Documents"):
                for i, doc in enumerate(message["source_docs"], 1):
                    st.markdown(f"**Document {i}:**")
                    st.markdown(f"**Section:** {doc.metadata.get('Section', 'N/A')}")
                    st.markdown(f"**Row:** {doc.metadata.get('row', 'N/A')}")
                    st.markdown(f"**Content:**")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about IPC sections..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if API key is available (from .env file only)
                if zenmux_key:
                    # Use LLM for answer
                    answer, source_docs = answer_with_llm(prompt, model=model_option, k=num_docs)
                    
                    if answer:
                        # Display answer with proper markdown rendering
                        st.markdown("---")
                        st.markdown(answer)
                        st.markdown("---")
                        
                        # Display source documents
                        if source_docs:
                            with st.expander(f"📄 View {len(source_docs)} Source Documents"):
                                for i, doc in enumerate(source_docs, 1):
                                    st.markdown(f'<div class="source-doc">', unsafe_allow_html=True)
                                    st.markdown(f"**Document {i}:**")
                                    st.markdown(f"**Section:** {doc.metadata.get('Section', 'N/A')}")
                                    st.markdown(f"**Row:** {doc.metadata.get('row', 'N/A')}")
                                    st.markdown(f"**Content:**")
                                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                    st.divider()
                                    st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "source_docs": source_docs
                        })
                    else:
                        st.error("Failed to generate answer. Please check your API key and try again.")
                else:
                    # Use template-based answer
                    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": num_docs})
                    answer, sections = answer_with_template(prompt, retriever)
                    
                    st.markdown("---")
                    st.markdown(answer)
                    st.markdown("---")
                    
                    # Convert sections to document format for display
                    source_docs = []
                    for sec in sections:
                        # Create a simple document-like object
                        class SimpleDoc:
                            def __init__(self, content, metadata):
                                self.page_content = content
                                self.metadata = metadata
                        
                        doc = SimpleDoc(
                            sec['content'],
                            {'Section': sec['section'], 'row': sec['row']}
                        )
                        source_docs.append(doc)
                    
                    if source_docs:
                        with st.expander(f"📄 View {len(source_docs)} Source Documents"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f'<div class="source-doc">', unsafe_allow_html=True)
                                st.markdown(f"**Document {i}:**")
                                st.markdown(f"**Section:** {doc.metadata.get('Section', 'N/A')}")
                                st.markdown(f"**Row:** {doc.metadata.get('row', 'N/A')}")
                                st.markdown(f"**Content:**")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.divider()
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_docs": source_docs
                    })
                    
                    st.info("💡 Tip: Add your ZenMux API key in the sidebar to get AI-powered answers!")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Falling back to template-based answer...")
                
                try:
                    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": num_docs})
                    answer, sections = answer_with_template(prompt, retriever)
                    st.markdown("---")
                    st.markdown(answer)
                    st.markdown("---")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                except Exception as e2:
                    st.error(f"Error generating answer: {str(e2)}")
                    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": num_docs})
                    answer, sections = answer_with_template(prompt, retriever)
                    st.markdown("---")
                    st.markdown(answer)
                    st.markdown("---")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>IPC Section Query System</p>
    </div>
    """,
    unsafe_allow_html=True
)

