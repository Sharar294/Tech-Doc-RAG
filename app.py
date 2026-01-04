import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
DB_PATH = "./chroma_langchain_db"

st.set_page_config(page_title="Local RAG Assistant", layout="centered")
st.title("🤖 Technical Doc Assistant")
st.caption("Running locally with Llama 3 & ChromaDB")

# --- 2. CACHING THE ENGINE ---
# We use @st.cache_resource so the model stays in memory and doesn't reload on every click
@st.cache_resource
def load_rag_chain():
    if not os.path.exists(DB_PATH):
        return None
    
    embeddings = OllamaEmbeddings(model="llama3")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm = OllamaLLM(model="llama3")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    system_prompt = (
        "You are an expert technical assistant. "
        "Use the following context to answer the question. "
        "If you don't know, say you don't know. Keep it technical.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Initialize the chain
rag_chain = load_rag_chain()

# --- 3. CHAT HISTORY (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. CHAT INTERFACE ---
if prompt_input := st.chat_input("Ask about your datasheet..."):
    
    # Check if database exists
    if rag_chain is None:
        st.error("Vector database not found! Please run ingest.py first.")
        st.stop()

    # Add user message to UI and State
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt_input})
            answer = response["answer"]
            
            # Show the answer
            st.markdown(answer)
            
            # Optional: Show which pages were referenced
            with st.expander("View Source Context"):
                for doc in response["context"]:
                    st.write(f"**Page {doc.metadata.get('page', 'N/A')}:** {doc.page_content[:200]}...")

    # Save AI response to State
    st.session_state.messages.append({"role": "assistant", "content": answer})