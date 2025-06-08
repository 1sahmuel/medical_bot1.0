import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="centered"
)

# --- RAG Pipeline Functions ---
@st.cache_resource(show_spinner=False)
def initialize_rag_pipeline():
    """Initialize and cache the RAG pipeline"""
    with st.spinner("Initializing medical knowledge base..."):
        # Load environment variables
        load_dotenv()
        required_vars = ["GROQ_API_KEY", "LANGCHAIN_API_KEY", "HF_TOKEN"]
        if not all(os.getenv(var) for var in required_vars):
            st.error("Missing environment variables. Check your .env file")
            st.stop()

        # Initialize LLM
        llm = ChatGroq(
            model_name="Gemma2-9b-It",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # Load and process documents
        try:
            loader = JSONLoader(
                file_path='./data/who_data.json',
                jq_schema='.diseases[]',
                content_key='full_content',
                metadata_func=lambda record, _: {'disease_name': record.get('name', '')}  # ✅ Fixed
            )
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
        except Exception as e:
            st.error(f"Failed to load documents: {str(e)}")
            st.stop()

        # Create vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Define prompt template
        prompt_template = """
        You are a medical information assistant 3 to 5 sentences when needed. If user enter ther names, greet them. Use ONLY the following context to answer.
        Context contains WHO-approved disease information. Never provide medical advice.

        If unsure, say: "This information is not available in WHO records."

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create the RetrievalQA chain
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

# --- Streamlit UI ---
def display_chat():
    """Handle chat interface and history"""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me about disease information from WHO records."}
        ]

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input prompt (user input)
    if prompt := st.chat_input("Ask about a disease..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user input (right side)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Consulting WHO database..."):
                try:
                    qa_chain = st.session_state.qa_chain
                    result = qa_chain({"query": prompt})
                    response = result["result"]

                    # Show answer
                    st.markdown(response)

                    # Show sources
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.subheader(f"Source {i}: {doc.metadata['disease_name']}")
                            st.caption(f"Relevance score: {doc.metadata.get('score', 'N/A')}")
                            st.text(doc.page_content[:300] + "...")
                except Exception as e:
                    response = "Sorry, I encountered an error."
                    st.error(f"Error processing query: {str(e)}")

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Main App ---
def main():
    st.title("🩺 WHO Medical Information Assistant")
    st.caption("Retrieval-Augmented Generation from WHO disease records")

    # Disclaimer
    st.warning("""
    ⚠️ This tool provides general health information from WHO records. 
    It does NOT provide medical advice. Consult a healthcare professional for medical concerns.
    """)

    # Initialize RAG pipeline once
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = initialize_rag_pipeline()

    # Display chat interface
    display_chat()

    # Optional: Debug info
    with st.sidebar:
        st.header("Configuration")
        st.code("Model: Gemma2-9b-It\nEmbeddings: BAAI/bge-small-en-v1.5")
        if st.checkbox("Show session state"):
            st.json(st.session_state)

# Run the app
if __name__ == "__main__":
    main()
