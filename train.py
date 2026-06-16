import argparse

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from dataset import MedicalQuestionPairDataset
from utils import compute_metrics, load_and_split_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for medical question similarity retrieval.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="models/bert-medical-sqr")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_df, eval_df = load_and_split_data(args.data_path, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = MedicalQuestionPairDataset(train_df, tokenizer, max_length=args.max_length)
    eval_dataset = MedicalQuestionPairDataset(eval_df, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
