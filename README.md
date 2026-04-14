# 📄 DocChat — Sənədlərimlə Danış

**DocChat** is a high-performance, privacy-focused PDF RAG (Retrieval-Augmented Generation) chatbot. It allows you to upload multiple PDF documents and chat with them using a combination of local vector embeddings and the powerful LLaMA 3.3 model via Groq API.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f3f3f3?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-blue?style=for-the-badge)

## ✨ Features
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` locally for lightning-fast text vectorization without sending your data to external embedding providers.
- **Fast Generation**: Powered by **LLaMA 3.3 70B** through the Groq Cloud API for near-instant responses.
- **Modern UI**: A premium, dark-themed Streamlit interface with smooth animations and responsive design.
- **Secure**: API keys are processed only in the session state and never stored permanently.
- **Multilingual Support**: Optimized for Azerbaijani and English queries.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A Groq API Key (Get it for free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/docchat.git
   cd docchat
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🛠️ How it Works
1. **PDF Text Extraction**: Uses `PyPDF2` to read and extract text from uploaded documents.
2. **Chunking**: Splits the text into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Vector Database**: Generates embeddings locally and stores them in a memory-efficient `FAISS` index.
4. **Retrieval**: When a question is asked, the system finds the most relevant chunks using similarity search.
5. **RAG Pipeline**: The relevant context + chat history is sent to Groq's LLaMA 3.3 model to generate a precise answer.

## 🎨 UI Customization
The app features a custom CSS theme defined in `app.py`, providing:
- Glassmorphism effects
- Custom typography (Syne & DM Sans)
- Elegant chat bubbles and status badges

## 📄 License
This project is open-source and available under the MIT License.

---
*Developed with ❤️ for the community.*
