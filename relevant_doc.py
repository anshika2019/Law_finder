from chroma_conn import chroma_vector_database
from simple_qa import answer_with_llm, answer_with_template
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get retriever
retriever = chroma_vector_database.as_retriever(search_kwargs={"k":5})

# Your query
query = 'A person stabs someone with a knife. What IPC section applies and what is the punishment?'

print("="*70)
print("IPC Section Query")
print("="*70)
print(f"\nQuestion: {query}\n")
print("-"*70)

# Check if ZenMux API key is available
has_zenmux_key = os.getenv("ZENMUX_API_KEY") is not None or os.getenv("ZenMux_KEY") is not None

if has_zenmux_key:
    print("Using ZenMux API (Gemini 2.5 Flash) for intelligent answer generation...\n")
    try:
        answer, source_docs = answer_with_llm(query, model="google/gemini-2.5-flash")
        if answer:
            print("Answer:")
            print(answer)
            print("\n" + "-"*70)
            print("\nSource Documents:")
            for i, doc in enumerate(source_docs, 1):
                section = doc.metadata.get('Section', 'N/A')
                print(f"\n{i}. Section: {section}")
                print(f"   Row: {doc.metadata.get('row', 'N/A')}")
                print(f"   Content: {doc.page_content[:250]}...")
    except Exception as e:
        print(f"Error with ZenMux LLM: {e}")
        print("\nFalling back to template-based answer...\n")
        answer, sections = answer_with_template(query, retriever)
        print(answer)
else:
    print("ZenMux API key not found!")
    print("Please set it in your .env file:")
    print("  ZENMUX_API_KEY='your-api-key-here'")
    print("  or")
    print("  ZenMux_KEY='your-api-key-here'")
    print("\nUsing template-based answer (no API key required)...\n")
    answer, sections = answer_with_template(query, retriever)
    print(answer)
    
    print("\n" + "-"*70)
    print("\nRetrieved Documents:")
    for i, doc in enumerate(retriever.invoke(query), 1):
        print(f"\n{i}. {doc.metadata}")
        print(f"   {doc.page_content[:300]}...")
