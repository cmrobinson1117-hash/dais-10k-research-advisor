import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.1:8b"
TOP_K = 5


class FilingStore:
    """Load chunk metadata and the FAISS index used by the agent tools."""

    def __init__(self, data_dir: Path, embed_model: str = EMBED_MODEL) -> None:
        self.data_dir = Path(data_dir)
        self.embedder = SentenceTransformer(embed_model)
        self.chunks = self._load_chunks(self.data_dir / "chunks.jsonl")
        self.index = faiss.read_index(str(self.data_dir / "faiss.index"))
        self.available_filings = self._build_available_filings()

    @staticmethod
    def _load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def _build_available_filings(self) -> List[str]:
        combos = sorted(
            {
                (str(c.get("company", "Unknown")), str(c.get("year", "Unknown")))
                for c in self.chunks
            }
        )
        return [f"{company} ({year})" for company, year in combos]

    def semantic_search(self, query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        query_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(query_emb, k)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
            if idx == -1:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "company": str(chunk.get("company", "Unknown")),
                    "year": str(chunk.get("year", "Unknown")),
                    "section": str(chunk.get("section", "Unknown")),
                    "chunk_index": int(chunk.get("chunk_index", -1)),
                    "text": str(chunk.get("text", "")),
                }
            )
        return results

    def filter_chunks(
        self,
        company: str,
        year: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        company_lower = company.lower().strip()

        matches: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            chunk_company = str(chunk.get("company", "")).lower().strip()
            chunk_year = str(chunk.get("year", "")).strip()
            text = str(chunk.get("text", ""))

            if company_lower not in chunk_company:
                continue
            if year and str(year).strip() != chunk_year:
                continue

            matches.append(
                {
                    "company": str(chunk.get("company", "Unknown")),
                    "year": str(chunk.get("year", "Unknown")),
                    "section": str(chunk.get("section", "Unknown")),
                    "chunk_index": int(chunk.get("chunk_index", -1)),
                    "text": text,
                }
            )

            if len(matches) >= limit:
                break

        # Soft keyword preference after collecting company/year matches
        if keyword:
            keyword_lower = keyword.lower().strip()
            prioritized = [m for m in matches if keyword_lower in m["text"].lower()]
            if prioritized:
                return prioritized[:limit]

        return matches


STORE = FilingStore(DEFAULT_DATA_DIR)
chat_model = ChatOllama(model=OLLAMA_MODEL, temperature=0)


@tool
def search_sec_filings(query: str, k: int = TOP_K) -> str:
    """
    Search the SEC 10-K vector database for the most relevant filing excerpts.

    Use this tool for general factual, analytical, or comparison questions.

    Args:
        query: User question or search phrase.
        k: Number of top chunks to retrieve.

    Returns:
        A formatted string containing ranked excerpts and metadata.
    """
    results = STORE.semantic_search(query=query, k=k)
    if not results:
        return "No relevant filing excerpts were found."

    formatted = []
    for r in results:
        formatted.append(
            f"[Result {r['rank']}]\n"
            f"Company: {r['company']}\n"
            f"Year: {r['year']}\n"
            f"Section: {r['section']}\n"
            f"Chunk: {r['chunk_index']}\n"
            f"Similarity Score: {r['score']:.4f}\n"
            f"Excerpt: {r['text'][:800]}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def lookup_company_filing(company: str, year: Optional[str] = None, keyword: Optional[str] = None) -> str:
    """
    Retrieve excerpts from a specific company's filing, optionally filtered by year and keyword.

    Use this when the user explicitly names a company or asks for company-specific evidence.

    Args:
        company: Company name such as Apple, Tesla, Delta, cocacola, or homedepot.
        year: Optional year such as 2023 or 2024.
        keyword: Optional keyword such as cybersecurity, products, segments, or strategy.

    Returns:
        A formatted string containing matching excerpts.
    """
    matches = STORE.filter_chunks(company=company, year=year, keyword=keyword, limit=5)
    if not matches:
        year_text = f" for {year}" if year else ""
        keyword_text = f" with keyword '{keyword}'" if keyword else ""
        return f"No excerpts found for {company}{year_text}{keyword_text}."

    formatted = []
    for i, m in enumerate(matches, start=1):
        formatted.append(
            f"[Match {i}]\n"
            f"Company: {m['company']}\n"
            f"Year: {m['year']}\n"
            f"Section: {m['section']}\n"
            f"Chunk: {m['chunk_index']}\n"
            f"Excerpt: {m['text'][:800]}"
        )
    return "\n\n---\n\n".join(formatted)


@tool
def list_available_filings() -> str:
    """
    List all company-year filings currently loaded in the local corpus.

    Use this when the user asks what filings are available.

    Returns:
        A newline-separated list of available filings.
    """
    return "Available filings:\n" + "\n".join(STORE.available_filings)


def _normalize_company_name(raw_company: str) -> str:
    """Map natural user company names to the company names used in chunk metadata."""
    mapping = {
        "apple": "apple",
        "tesla": "tesla",
        "delta": "delta",
        "delta air lines": "delta",
        "coca-cola": "cocacola",
        "coca cola": "cocacola",
        "coke": "cocacola",
        "home depot": "homedepot",
        "the home depot": "homedepot",
    }
    return mapping.get(raw_company.lower().strip(), raw_company.lower().strip())


def ask_agent(user_query: str) -> str:
    """
    Run a controlled tool-using agent flow.

    This preserves the milestone's tool-based requirement while making outputs
    more reliable than free-form tool-calling with the local model.
    """
    q = user_query.lower().strip()

    if any(
        phrase in q
        for phrase in [
            "what filings are available",
            "available filings",
            "which filings are available",
            "what companies do you have",
            "what filings do you have",
            "list available filings",
        ]
    ):
        return list_available_filings.invoke({})

    known_companies = [
        "apple",
        "tesla",
        "delta air lines",
        "delta",
        "coca-cola",
        "coca cola",
        "coke",
        "the home depot",
        "home depot",
    ]

    detected_company: Optional[str] = None
    for company in known_companies:
        if company in q:
            detected_company = company
            break

    detected_year: Optional[str] = None
    for year in ["2023", "2024"]:
        if year in q:
            detected_year = year
            break

    candidate_keywords = [
        "cybersecurity",
        "risk factors",
        "risk",
        "business segments",
        "segments",
        "products",
        "strategy",
        "business",
        "supply chain",
        "operations",
    ]
    detected_keyword: Optional[str] = None
    for keyword in candidate_keywords:
        if keyword in q:
            detected_keyword = keyword
            break

    # First retrieval pass
    if detected_company:
        normalized_company = _normalize_company_name(detected_company)
        context = lookup_company_filing.invoke(
            {
                "company": normalized_company,
                "year": detected_year,
                "keyword": detected_keyword,
            }
        )
    else:
        context = search_sec_filings.invoke(
            {
                "query": user_query,
                "k": TOP_K,
            }
        )

    # Fallback retrieval if company lookup is weak
    if "No excerpts found" in context or len(context.strip()) < 200:
        enhanced_query = f"{user_query} SEC 10-K business segments products strategy risk"
        context = search_sec_filings.invoke(
            {
                "query": enhanced_query,
                "k": TOP_K,
            }
        )

    synthesis_prompt = f"""
You are a financial research assistant analyzing SEC 10-K filings.

Answer the user's question using ONLY the context below.
Do not use outside knowledge.
Do not mention tools, tool calls, JSON, or internal reasoning.
If the retrieved evidence is weak, try to infer cautiously based on partial evidence.
If still unclear, say the evidence is limited.
Mention company names and years when supported by the evidence.
When the answer is a short list, format it clearly as bullet points.
If the context contains partial mentions (e.g., "business", "segments"), extract and interpret them carefully.

Question:
{user_query}

Context:
{context}

Final answer:
""".strip()

    return chat_model.invoke(synthesis_prompt).content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="SEC 10-K Research Advisor")
    parser.add_argument("--query", type=str, help="Single question to ask the agent")
    args = parser.parse_args()

    if args.query:
        print(ask_agent(args.query))
        return

    print("SEC 10-K Research Advisor")
    print("Type a question and press Enter. Press Ctrl+C to exit.\n")

    while True:
        try:
            user_query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_query:
            continue

        try:
            answer = ask_agent(user_query)
        except Exception as exc:
            answer = f"Error: {exc}"

        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()