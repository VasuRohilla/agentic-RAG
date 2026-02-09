# Offline Doc Q&A Buddy

Chat with your documents — **100% offline LLM, zero API cost.**

## What It Does

* Upload **PDF / Word** documents
* Ask questions in chat
* AI answers **only from your document**
* No internet, no OpenAI, fully local

## Tech Used

* Ollama (Mistral + nomic-embed-text)
* Pinecone Vector DB
* LangChain
* Streamlit UI

## Setup

### 1. Install models

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add environment variable

Create `.env`:

```
PINECONE_API_KEY=your_key_here
```

Create Pinecone index named:

```
local-index
```

### 4. Run

```bash
streamlit run agenticRAG.py
```

## Supported Files

* PDF
* DOCX / DOC

## Behavior

* Uses **only uploaded document**
* No general knowledge
* If not found →

  > I don't find this in the document.

---

**Private • Local • Free**
