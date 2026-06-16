# Phase 3: Agentic Graph & Multi-Document Relationship Analyzer
import chromadb
import json
import re
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM

class LegalAgent:
    def __init__(self, db_dir="./local_vector_db", collection_name="legal_documents"):
        # Initialize on-premises Vector DB and Embedding model
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.local_emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        # Use get_or_create_collection to ensure the collection exists
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name, embedding_function=self.local_emb_fn)
        # Initialize local Ollama LLM
        self.llm = OllamaLLM(model="llama3")

    def analyze_query(self, user_query):
        print(f"[Agent] Starting legal search for: '{user_query}'")

        # --- LOOP 1: Primary Search ---
        primary_results = self.collection.query(query_texts=[user_query], n_results=2)

        retrieved_contexts = []
        source_documents = []

        # Check if any results were returned for the first (and only) query
        if primary_results['documents'] and primary_results['metadatas'] and len(primary_results['documents'][0]) > 0:
            # Iterate through the results of the first query
            # primary_results['documents'][0] is the list of documents for the first query
            # primary_results['metadatas'][0] is the list of metadatas for the first query
            for i in range(len(primary_results['documents'][0])):
                text = primary_results['documents'][0][i]
                meta = primary_results['metadatas'][0][i]
                retrieved_contexts.append(text)
                source_documents.append({
                    "file": meta['source_file'],
                    "page": meta['page'],
                    "bbox": json.loads(meta['bbox']),
                    "type": "Primary Match"
                })

        combined_text = "\n".join(retrieved_contexts)
        print(f"[Agent] Found {len(source_documents)} primary reference blocks.")

        # --- LOOP 2: Agentic Relationship Discovery ---
        # Look for references to other legal attachments (e.g., Exhibit B, Schedule 1)
        relationship_pattern = r"(Exhibit\s+[A-Z]|Schedule\s+\d+|Appendix\s+[A-Z0-9])"
        found_relationships = re.findall(relationship_pattern, combined_text, re.IGNORECASE)

        if found_relationships:
            # Clean duplicates
            unique_relations = list(set(found_relationships))
            print(f"[Agent Warning] Found cross-document references: {unique_relations}")

            for relation in unique_relations:
                print(f"[Agent Execution] Branching search to retrieve related text for: '{relation}'")
                # Auto-expand search to grab the related exhibit document contents
                related_results = self.collection.query(query_texts=[relation], n_results=1)

                # Check if any results were returned for the first (and only) related query
                if related_results['documents'] and related_results['metadatas'] and len(related_results['documents'][0]) > 0:
                    # Iterate through the results of the first related query
                    for j in range(len(related_results['documents'][0])):
                        r_text = related_results['documents'][0][j]
                        r_meta = related_results['metadatas'][0][j]
                        retrieved_contexts.append(r_text)
                        source_documents.append({
                            "file": r_meta['source_file'],
                            "page": r_meta['page'],
                            "bbox": json.loads(r_meta['bbox']),
                            "type": f"Related Document ({relation})"
                        })

        # --- LOOP 3: Legal Synthesis ---
        final_context = "\n\n".join(retrieved_contexts)
        system_prompt = f"""You are an expert on-premises legal assistant.
Synthesize an answer for the solicitor using the provided text blocks.
Highlight how different documents or sections relate to each other if relevant.

CONTEXT BLOCKS:
{final_context}

QUESTION:
{user_query}

SYNTHESIZED LEGAL RESPONSE:"""

        print("[Agent] Synthesizing final answer using local Llama3...")
        response = self.llm.invoke(system_prompt)

        return {
            "answer": response,
            "sources": source_documents
        }

if __name__ == '__main__':
    # Test execution block when running the file standalone
    agent = LegalAgent()

    # Add some dummy data if the collection is empty for testing purposes
    if agent.collection.count() == 0:
        print("[Agent] Adding dummy data to the collection for testing.")
        agent.collection.add(
            documents=[
                "This is a legal document discussing purchase price.",
                "Exhibit B details the terms of the agreement.",
                "Schedule 1 outlines the payment plan.",
                "This document is unrelated to the query."
            ],
            metadatas=[
                {"source_file": "doc1.pdf", "page": 1, "bbox": json.dumps([10,20,30,40])},
                {"source_file": "doc2.pdf", "page": 5, "bbox": json.dumps([50,60,70,80])},
                {"source_file": "doc3.pdf", "page": 2, "bbox": json.dumps([100,110,120,130])},
                {"source_file": "doc4.pdf", "page": 1, "bbox": json.dumps([10,20,30,40])}
            ],
            ids=["doc1_chunk1", "doc2_chunk1", "doc3_chunk1", "doc4_chunk1"]
        )

    output = agent.analyze_query("What is the purchase price?")
    print("\n=== FINAL AGENT ANSWER ===")
    print(output['answer'])
