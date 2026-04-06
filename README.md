# SEC 10-K Research Advisor

This project is an agentic prototype for Milestone 3. It answers questions over a local corpus of SEC 10-K filings using a ReAct-style LangChain agent and local tools.

## What is included

- `agent.py` - main agent code with registered tools
- `app.py` - Streamlit chat interface
- `batch_run.py` - batch query runner that saves structured outputs
- `pipeline.py` - earlier ingestion and indexing pipeline
- `query.py` - earlier retrieval test script
- `chunks.jsonl`, `docs.jsonl`, `faiss.index`, `id_map.json` - processed corpus artifacts
- source PDF filings for Apple, Tesla, Delta, Home Depot, and Coca-Cola for 2023 and 2024

## Agent design

The system uses a single LLM-based agent with multiple tools:

1. `search_sec_filings` - semantic vector search over the FAISS index
2. `lookup_company_filing` - targeted lookup for a specific company and optional year/keyword
3. `list_available_filings` - lists the filings currently loaded into the corpus

The agent decides which tool to use based on the user question, then produces a concise answer grounded in retrieved filing excerpts.

## Installation

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
```

You also need Ollama running locally with the model used by the agent:

```bash
ollama pull llama3:8b
```

## Run the chat interface

```bash
streamlit run app.py
```

## Run the command-line agent

Interactive mode:

```bash
python agent.py
```

Single question mode:

```bash
python agent.py --query "Compare Apple and Tesla 2024 filings."
```

## Run batch queries

Use the built-in demo questions:

```bash
python batch_run.py
```

Use your own text file or JSON list:

```bash
python batch_run.py --input queries.txt --output results.json
python batch_run.py --input queries.json --output results.csv
```

## Example questions

- What cybersecurity risks appear across the 2024 filings?
- Compare Apple and Tesla 2024 business focus areas.
- What supply chain or operational risks are mentioned most often?
- What filings are available in the corpus?

## Notes

- The local corpus artifacts are loaded from the same folder as `agent.py` by default.
- The agent is designed for Milestone 3 demonstration purposes, so answers are grounded in retrieved filing chunks rather than full-document reasoning.
- If you want to rebuild the index from PDFs, use `pipeline.py` from the earlier milestone work.


# Update: The system uses an agentic tool-based workflow over a local SEC 10-K corpus. We registered multiple tools for semantic vector search, company-specific filing lookup, and corpus inspection. To improve reliability with local Ollama execution, we used a controlled reasoning flow in ask_agent() that routes questions to the appropriate tool and then uses the LLM to synthesize a grounded answer from retrieved evidence.