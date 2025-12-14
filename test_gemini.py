"""
Test script for ZenMux integration
"""
from dotenv import load_dotenv
import os
from chroma_conn import chroma_vector_database
from simple_qa import answer_with_llm, answer_with_template

# Load environment variables from .env file
load_dotenv()

query = "A person stabs someone with a knife. What IPC section applies and what is the punishment?"

print("="*70)
print("Testing ZenMux API Integration")
print("="*70)
print(f"\nQuestion: {query}\n")
print("-"*70)

# Check if API key is set (try both formats)
zenmux_key = os.getenv("ZENMUX_API_KEY") or os.getenv("ZenMux_KEY")
if zenmux_key:
    print("ZenMux API key found! Using ZenMux with Gemini 2.5 Flash...\n")
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
                print(f"   Content: {doc.page_content[:200]}...")
    except Exception as e:
        print(f"Error: {e}")
        print("\nFalling back to template-based answer...\n")
        retriever = chroma_vector_database.as_retriever(search_kwargs={"k": 5})
        answer, sections = answer_with_template(query, retriever)
        print(answer)
else:
    print("ZenMux API key not found!")
    print("Please set it in your .env file:")
    print("  ZENMUX_API_KEY='your-api-key-here'")
    print("  or")
    print("  ZenMux_KEY='your-api-key-here'")
    print("\nUsing template-based answer instead...\n")
    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": 5})
    answer, sections = answer_with_template(query, retriever)
    print(answer)

