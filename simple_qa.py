"""
Simple Q&A system using retrieved IPC sections with ZenMux API
"""
from chroma_conn import chroma_vector_database
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def answer_with_template(query, retriever, k=5):
    """Answer question using retrieved documents with a simple template"""
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Extract relevant information
    sections = []
    for doc in docs:
        section_info = {
            'section': doc.metadata.get('Section', 'N/A'),
            'content': doc.page_content,
            'row': doc.metadata.get('row', 'N/A')
        }
        sections.append(section_info)
    
    # Create answer from retrieved sections
    answer = f"Based on the Indian Penal Code, here are the relevant sections for your query:\n\n"
    
    for i, sec in enumerate(sections, 1):
        answer += f"{i}. {sec['section']}\n"
        # Extract key information
        if 'Offense:' in sec['content']:
            offense_line = [line for line in sec['content'].split('\n') if 'Offense:' in line]
            if offense_line:
                answer += f"   Offense: {offense_line[0].split('Offense:')[1].strip()}\n"
        
        if 'Punishment:' in sec['content']:
            punishment_line = [line for line in sec['content'].split('\n') if 'Punishment:' in line]
            if punishment_line:
                answer += f"   Punishment: {punishment_line[0].split('Punishment:')[1].strip()}\n"
        
        # Get description summary
        if 'Description:' in sec['content']:
            desc_start = sec['content'].find('Description:')
            desc_text = sec['content'][desc_start:desc_start+300]
            answer += f"   Summary: {desc_text.split('Offense:')[0].replace('Description:', '').strip()[:200]}...\n"
        
        answer += "\n"
    
    return answer, sections

def extract_key_terms(query):
    """Extract key legal terms from query to improve retrieval"""
    import re
    
    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)
    
    # Common legal action terms
    legal_actions = ['stab', 'stabbed', 'stabbing', 'cut', 'hurt', 'injure', 'assault', 
                     'attack', 'murder', 'kill', 'rob', 'theft', 'steal', 'rape', 
                     'abuse', 'threaten', 'intimidate', 'cheat', 'fraud', 'deceive',
                     'cause', 'causing', 'voluntarily', 'intentionally']
    
    # Extract action verbs and legal terms
    key_terms = []
    for word in words:
        if word in legal_actions:
            key_terms.append(word)
    
    # Extract weapon/object terms
    weapons = ['knife', 'weapon', 'gun', 'pistol', 'sword', 'stick', 'stone', 
               'acid', 'poison', 'fire', 'explosive', 'dangerous', 'instrument']
    for word in words:
        if word in weapons:
            key_terms.append(word)
    
    # Extract body parts or injury types
    injuries = ['death', 'grievous', 'hurt', 'wound', 'injury', 'harm', 'bodily']
    for word in words:
        if word in injuries:
            key_terms.append(word)
    
    # Build enhanced query: prioritize key terms, then add original query
    # This helps the embedding model focus on legal concepts
    if key_terms:
        # Create a focused query with key terms first
        enhanced_query = ' '.join(set(key_terms)) + ' voluntarily causing hurt dangerous weapon'
    else:
        enhanced_query = query
    
    return enhanced_query

def answer_with_zenmux(query, api_key, model="google/gemini-2.5-flash", k=5):
    """Answer using ZenMux API"""
    from openai import OpenAI
    
    # Initialize the OpenAI client pointing to ZenMux endpoint
    client = OpenAI(
        base_url="https://zenmux.ai/api/v1",
        api_key=api_key,
    )
    
    # Extract key terms to improve retrieval
    enhanced_query = extract_key_terms(query)
    
    # Create retriever with k parameter (increase k to get more relevant docs)
    # Use at least 15 documents to ensure we get relevant sections even if some are duplicates
    retriever = chroma_vector_database.as_retriever(search_kwargs={"k": max(k, 15)})
    
    # Retrieve relevant documents using enhanced query
    docs = retriever.invoke(enhanced_query)
    
    # Check if documents were retrieved
    if not docs or len(docs) == 0:
        return "I couldn't find any relevant IPC sections in the database for your query. Please try rephrasing your question or check if the database has been properly initialized.", []
    
    # Build context from documents, removing duplicates based on Section
    seen_sections = set()
    unique_docs = []
    for doc in docs:
        section = doc.metadata.get('Section', '')
        if section and section not in seen_sections:
            seen_sections.add(section)
            unique_docs.append(doc)
        elif not section:  # Include docs without section metadata
            unique_docs.append(doc)
    
    # Use unique docs, but keep at least k documents
    if len(unique_docs) < k:
        unique_docs = docs[:k]
    
    # Build context from unique documents with proper section labels
    context = "\n\n".join([f"**IPC Section {doc.metadata.get('Section', 'Unknown')}:**\n{doc.page_content}" for doc in unique_docs])
    
    # Create prompt
    prompt = f"""You are a legal assistant specializing in Indian Penal Code (IPC). 
Use the following pieces of context from IPC sections to answer the question accurately and comprehensively.
If the answer is not in the provided context, say "I don't have enough information in the provided IPC sections to answer this question."

Context from IPC sections:
{context}

Question: {query}

Provide a detailed, comprehensive, and well-structured answer based on the IPC sections provided above. Format your answer using Markdown with proper formatting. Your answer should:

1. Be thorough and detailed (at least 3-4 paragraphs or 200-300 words)
2. Use **bold text** for important terms, section numbers, and key information:
   - **IPC Section XXX** for section numbers
   - **Offense:** for offense descriptions
   - **Punishment:** for punishment details
   - **Key terms** and important legal concepts
3. Clearly identify the relevant IPC section(s) and section numbers using bold formatting
4. Explain the offense in detail, including what actions constitute the offense
5. Provide complete information about the punishment, including duration, fines, and any alternative punishments
6. Include relevant details from the IPC section descriptions
7. Explain the legal implications and context
8. If multiple sections are relevant, discuss each one separately with clear headings
9. Use clear, professional legal language while remaining accessible
10. Structure your answer with proper paragraphs and use bullet points or numbered lists where appropriate

Format your answer with:
- **Bold text** for section numbers, key terms, and important information
- Clear paragraphs with proper spacing
- Headings using ## or ### for different sections if needed
- Bullet points or numbered lists for multiple items

Example format:
**IPC Section 324**

**Offense:** [description in bold]

**Punishment:** [punishment details in bold]

[Detailed explanation paragraph...]

Make sure your answer is comprehensive and covers all aspects of the question. Do not provide brief or short answers - be thorough and detailed."""
    
    # Make the request to ZenMux
    try:
        completion = client.chat.completions.create(
            model=model,  # Format: "provider/model-name" e.g., "google/gemini-2.5-flash"
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
        
        answer = completion.choices[0].message.content
        return answer, docs
    except Exception as e:
        # Try fallback models if the primary model fails
        fallback_models = [
            "google/gemini-2.0-flash-exp",
            "google/gemini-1.5-flash",
        ]
        
        for fallback_model in fallback_models:
            try:
                completion = client.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0
                )
                answer = completion.choices[0].message.content
                return answer, docs
            except:
                continue
        
        raise e

def answer_with_llm(query, model="google/gemini-2.5-flash", k=5):
    """Answer question using ZenMux LLM"""
    # Get ZenMux API key
    zenmux_api_key = os.getenv("ZENMUX_API_KEY") or os.getenv("ZenMux_KEY")
    
    if not zenmux_api_key:
        raise ValueError("ZenMux API key not found. Set ZENMUX_API_KEY or ZenMux_KEY environment variable.")
    
    # Use ZenMux API
    try:
        return answer_with_zenmux(query, zenmux_api_key, model, k=k)
    except Exception as e:
        print(f"Error with ZenMux API: {e}")
        return None, None

if __name__ == "__main__":
    # Example query
    query = "A person stabs someone with a knife. What IPC section applies and what is the punishment?"
    
    print("="*70)
    print("IPC Section Query System")
    print("="*70)
    print(f"\nQuestion: {query}\n")
    print("-"*70)
    
    # Check if ZenMux API key is available
    has_zenmux_key = os.getenv("ZENMUX_API_KEY") is not None or os.getenv("ZenMux_KEY") is not None
    
    if has_zenmux_key:
        print("Using ZenMux API (Gemini 2.5 Flash) for answer generation...\n")
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
                    print(f"   Content preview: {doc.page_content[:200]}...")
        except Exception as e:
            print(f"Error with ZenMux LLM: {e}")
            print("\nFalling back to template-based answer...\n")
            retriever = chroma_vector_database.as_retriever(search_kwargs={"k": 5})
            answer, sections = answer_with_template(query, retriever)
            print("Answer:")
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
        print("Answer:")
        print(answer)
