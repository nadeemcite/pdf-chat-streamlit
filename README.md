# PDF Chat App

This is a **PDF Chat App** built with **Streamlit** that allows users to upload a PDF file, process its content using LangChain, and interact with the data through a conversational AI powered by Groq's Mixtral-8x7b model.

**URL of the deployed app:** [PDF Chat App](https://pdf-chat-appgit-qfxbmr8vteevvqjappgxxs4.streamlit.app/)

---

## Features
- **PDF Upload:** Upload a PDF file for analysis.
- **PDF Parsing:** Parses PDF content using `PyPDFLoader`.
- **Text Embeddings:** Embeds the PDF content using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Search:** Retrieves relevant information using `InMemoryVectorStore`.
- **Conversational AI:** Answers user queries about the uploaded document using Groq's `mixtral-8x7b` model.

---

## Technologies Used
- **Streamlit:** For the user interface.
- **LangChain:** For AI model interaction and document handling.
- **LangChain-Community:** For community plugins like `PyPDFLoader`.
- **LangChain-Groq:** For using Groq's large language models.
- **LangChain-HuggingFace:** For integrating Hugging Face sentence transformers.
- **LangGraph:** For creating a graph-based conversational flow.
- **Sentence-Transformers:** For creating text embeddings.
- **PyPDF:** For PDF parsing.

---

## How to Use
1. **Run the App:**
   ```bash
   streamlit run app.py
   ```
2. **Upload a PDF File:**
   - Click on the file uploader and select a PDF.
3. **Enter Groq API Key:**
   - Enter your Groq API Key in the sidebar.
4. **Ask a Question:**
   - Type a question related to the uploaded document.
5. **Get Answers:**
   - The app will provide answers based on the PDF content.

---

## Requirements

You can install them using:

```bash
pip install -r requirements.txt
```

---

## Project Structure
```
- app.py                   # Main application script
- requirements.txt         # List of required Python packages
```

---

## Environment Variables
- **Groq API Key:** Required for using the `mixtral-8x7b` model. Enter it via the Streamlit sidebar.

---

## License
This project is licensed under the MIT License.

---

**Happy Chatting! 🎉**
