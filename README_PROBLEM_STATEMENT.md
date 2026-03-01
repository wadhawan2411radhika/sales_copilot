# I2 Coding Round – Conversational AI Copilot

---

## 1 · Problem Statement  
You are building a **chatbot** that helps sales users understand and summarise their past sales calls.  
The bot will ingest a small set of call-transcript files (samples provided), embed and store them, and let users ask questions in natural language. The response **must indicate which conversation segment(s) contributed to the answer**. 

---

## 2 · Your Main Responsibilities :contentReference[oaicite:1]{index=1}
1. **Retrieval**  
   * Parse user queries.  
   * Retrieve and compose relevant information from stored transcripts.  
   * Build prompts and call an LLM (hosted or local).  
   * Return the answer **with source snippets**.  
2. **Ingestion & Storage**  
   * Pick any datastore (DB, in-memory, FAISS, etc.) and design the schema.  
   * Store transcripts, metadata, and vector embeddings.  
   * Ensure fast retrieval for summarisation and Q&A.  
3. **CLI Chatbot**  
   * Provide a simple command-line interface.  
   * Example commands:  
     ```
     list my call ids
     summarise the last call
     Give me all negative comments when pricing was mentioned in the calls
     ingest a new call transcript from <path>
     ```

---

## 3 · What We Evaluate :contentReference[oaicite:2]{index=2}
| Area | Focus |
|------|-------|
| **Low-level design** | Folder/class structure, clarity, modularity |
| **Storage design** | Schema, entity relations, retrieval efficiency (justify your choice) |
| **Prompt engineering** | Quality of prompts, handling of varied intents |
| **GenAI skills** | Effective use of LLM + RAG concepts |
| **Working solution** | Runs end-to-end with clear setup instructions |
| **Git** | commit hygiene, PR etiquette [use git from the start] |

---

## 4 · Deliverables Checklist ✅
- [ ] All storage and retrieval code  
- [ ] CLI chatbot (`python cli.py` or similar)  
- [ ] **README** with setup & run steps and any assumptions  
- [ ] `.env.example` (if you rely on env variables)  
- [ ] *(Good to have)* basic tests for ingestion and retrieval  
- [ ] Github/Gitlab URL

---

## 5 · Time Limit & Tools  
* You have **24 hours** after receiving the assignment link. Most candidates finish in 3 – 4 focused hours. :contentReference[oaicite:3]{index=3}  
* Any IDE, Copilot, code-gen tool, or LLM is allowed during the take-home. :contentReference[oaicite:4]{index=4}  

---

## 6 · Evaluation Note  
In the review round you may be asked to **extend your code live**; we’ll judge how well you understand and evolve your design, not just initial correctness. 

---

## 7 · Assumptions  
Making reasonable assumptions is fine—just document them in your README and keep them aligned with this problem statement.