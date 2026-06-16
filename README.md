# On-Premises Agentic RAG Application for Legal Analytics

This repository contains an air-gapped, fully open-source Agentic Retrieval-Augmented Generation (RAG) system engineered for solicitors and legal professionals. The application runs entirely on local, on-premises servers to guarantee absolute client data privacy and strict compliance without hitting the public internet.

## 🛠️ On-Premises Technology Stack
*   **User Interface (UI)**: Streamlit or Gradio (python-native local web interfaces).
*   **Orchestration Framework**: LangGraph or LlamaIndex (for multi-step legal reasoning and graph-based relationships).
*   **Local LLM Engine**: Ollama or vLLM hosting open-source models (e.g., Llama 3) on local hardware.
*   **Vector Database**: ChromaDB or Qdrant (deployed locally on your private network).
*   **Local Embeddings**: Hugging Face `sentence-transformers` running completely offline.
*   **Document Processor**: `PyMuPDF` or `Unstructured` for coordinate mapping and text extraction.

## 🤖 Legal Agentic Features & Architecture
1.  **Multi-Document Relationship Mapping**: Tracks shared metadata and entities to find connected legal cases, contracts, or cross-references.
2.  **Autonomous Query Expansion**: Rewrites complex legal phrasing into multiple targeted queries.
3.  **Strict Source Verification**: A self-correction loop screens retrieved chunks for relevance before passing them to the LLM.
4.  **UI Text Highlight Syncing**: Stores character coordinates during parsing. Clicking a document in the UI opens it and highlights the exact matching text.

## 📂 Project Roadmap & Milestones
* [ ] Phase 1: Local environment setup & installing offline dependencies.
* [ ] Phase 2: Document Processing Pipeline (PDF parsing, metadata extraction, character-coordinate mapping, local embedding generation).
* [ ] Phase 3: Building the multi-document Agentic retrieval loop.
* [ ] Phase 4: Constructing the UI with a document viewer that supports active text highlighting.
