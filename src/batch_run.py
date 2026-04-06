import argparse
import csv
import json
from pathlib import Path
from typing import List

from agent import ask_agent


DEFAULT_QUERIES = [
    "What cybersecurity risks appear across the 2024 filings?",
    "Compare Apple and Tesla 2024 business focus areas.",
    "What supply chain or operational risks are mentioned most often?",
]


def load_queries(path: Path) -> List[str]:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("JSON query file must contain a list of questions.")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def save_json(results: List[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_csv(results: List[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "answer", "status", "error"])
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch query runner for the SEC 10-K agent")
    parser.add_argument("--input", type=str, help="Optional .txt or .json file containing queries")
    parser.add_argument("--output", type=str, default="batch_results.json", help="Output file path")
    args = parser.parse_args()

    queries = load_queries(Path(args.input)) if args.input else DEFAULT_QUERIES
    results = []

    for query in queries:
        try:
            answer = ask_agent(query)
            results.append({"query": query, "answer": answer, "status": "success", "error": ""})
        except Exception as exc:  # pragma: no cover - runtime protection
            results.append({"query": query, "answer": "", "status": "error", "error": str(exc)})

    output_path = Path(args.output)
    if output_path.suffix.lower() == ".csv":
        save_csv(results, output_path)
    else:
        save_json(results, output_path)

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
