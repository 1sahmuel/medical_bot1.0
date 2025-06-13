# 🩺 WHO Medical Information Assistant (MIH\_Chatbot)

A **Retrieval-Augmented Generation (RAG)** chatbot that provides information on diseases using official WHO records. Powered by **LangChain**, **Groq's Gemma2-9b-It** LLM, **HuggingFace embeddings**, and **ChromaDB** for vector search. The app runs on **Streamlit** and supports natural language queries.

> ⚠️ **Disclaimer**: This tool provides general health information from WHO records. It does **NOT** give medical advice. Always consult a healthcare professional for any health concerns.

---

## 🔧 Features

* 💬 Chat interface built with Streamlit
* 🧠 LLM: Groq-hosted **Gemma2-9b-It** via LangChain
* 📚 Retrieval from local JSON WHO dataset using ChromaDB
* 🔍 Embedding model: **BAAI/bge-small-en-v1.5** (HuggingFace)
* 📄 Custom prompt template with WHO-specific rules
* 🧾 Shows source document info for each answer

---

## 📁 Project Structure

```
medical_bot1.0/
├── data/
│   └── who_data.json             # WHO disease records                 
├── app.py                       # Streamlit app
├── .env                         # API keys (not tracked)
├── .gitignore                   # Excludes chroma_db, .env, etc.
└── README.md                    # You're here
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/1sahmuel/medical_bot1.0.git
cd medical_bot1.0
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following:

```ini
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_key
HF_TOKEN=your_huggingface_token
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will load the WHO JSON dataset, embed documents, and allow you to query via chat.

---

## 📝 Prompt Template (LLM Behavior)

> "You are a helpful medical assistant. Use the following context exactly as it is written in the database to answer the user's question. Do not give any medical advice.
>
> If the context doesn't contain the exact answer, provide a partial answer based on the available information and explain what additional information would be needed.
>
> Only say "I don't know based on the provided information" if no relevant content is found."

---

## 🧠 Example Questions

* "What are the symptoms of malaria?"
* "Is cholera spread through water?"
* "Tell me about the treatment for tuberculosis."

---
![image](https://github.com/user-attachments/assets/21f4c555-7b9b-49c2-9c15-dd0e25f26337)



