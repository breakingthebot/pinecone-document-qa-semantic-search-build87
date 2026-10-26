# Build 87 — Pinecone: Enterprise Document Q&A & Hybrid RAG Engine

> **Category**: Databases - Vector/Search > **Summary**: Enterprise Document Q&A RAG engine featuring dense vector embeddings, BM25 sparse vectors, Pinecone hybrid index search, conversational session memory with query reformulation, faithfulness & hallucination guardrails, and an interactive web dashboard.

## Stack

- **Primary Technology**: Pinecone (Databases - Vector/Search)
- **Python Packages**: `﻿annotated-doc`, `annotated-types`, `anyio`, `certifi`, `charset-normalizer`, `click`

## Setup & Environment

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` if you want to override default local configuration.

## Usage

```powershell
python -m unittest discover -v
```

## Data & Privacy

- Processes data locally and stores state in gitignored local storage or sandboxed databases.
- Zero secrets, credentials, or personally identifiable information (PII) are hardcoded or committed.
