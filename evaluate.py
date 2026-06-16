import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from dataset import MedicalQuestionPairDataset
from utils import compute_metrics, load_and_split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned BERT model for medical SQR.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    _, test_df = load_and_split_data(args.data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    test_dataset = MedicalQuestionPairDataset(test_df, tokenizer, max_length=args.max_length)
    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)

    metrics = trainer.evaluate(test_dataset)
    print(metrics)


if __name__ == "__main__":
    main()
