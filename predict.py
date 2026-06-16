import argparse
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Retrieve the most similar medical questions using a fine-tuned BERT classifier.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--candidate_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


def score_pair(question: str, candidate: str, tokenizer, model, max_length: int) -> float:
    encoded = tokenizer(
        question,
        candidate,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)
    return float(probabilities[0, 1])


def load_candidates(candidate_file: str) -> List[str]:
    df = pd.read_csv(candidate_file)
    candidates = pd.concat([df["question_1"], df["question_2"]]).dropna().drop_duplicates().tolist()
    return candidates


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    candidates = load_candidates(args.candidate_file)
    ranked_results: List[Tuple[str, float]] = []

    for candidate in candidates:
        score = score_pair(args.question, candidate, tokenizer, model, args.max_length)
        ranked_results.append((candidate, score))

    ranked_results = sorted(ranked_results, key=lambda item: item[1], reverse=True)

    for candidate, score in ranked_results[: args.top_k]:
        print(f"{score:.4f}\t{candidate}")


if __name__ == "__main__":
    main()
