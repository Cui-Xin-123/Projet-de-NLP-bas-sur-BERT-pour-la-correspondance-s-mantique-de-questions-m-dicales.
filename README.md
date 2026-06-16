# Projet-de-NLP-bas-sur-BERT-pour-la-correspondance-s-mantique-de-questions-m-dicales.
L’objectif de ce projet est d’identifier automatiquement si deux questions médicales expriment la même intention ou portent sur le même problème de santé. Le modèle repose sur BERT, un modèle Transformer pré-entraîné, puis affiné (fine-tuning) sur un jeu de données annoté.
# SQR médical avec BERT

Ce projet reproduit un pipeline simple de **Question Answering Retrieval / Similar Question Retrieval (SQR)** dans le domaine médical.  
L'objectif est de retrouver, pour une question patient, la question médicale la plus proche dans une base de connaissances, en utilisant des représentations textuelles issues de BERT.

## Objectifs

- Préparer un jeu de paires question-question avec un label binaire (`1 = similaires`, `0 = non similaires`).
- Fine-tuner un modèle BERT pour la classification de similarité sémantique.
- Évaluer le modèle avec Accuracy, Precision, Recall et F1-score.
- Utiliser le modèle entraîné pour retrouver les questions médicales les plus proches.

## Stack technique

- Python
- PyTorch
- Transformers / BERT
- scikit-learn
- pandas
- NumPy

## Structure du projet

```text
bert-medical-sqr/
├── data/
│   └── sample/
│       └── medical_question_pairs.csv
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── dataset.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Données

Le fichier `data/sample/medical_question_pairs.csv` contient un petit exemple de données au format :

```csv
question_1,question_2,label
"What are the symptoms of diabetes?","How can I know if I have diabetes?",1
"What are the symptoms of diabetes?","How to treat a broken arm?",0
```

Dans un vrai contexte, ce fichier peut être remplacé par un corpus médical plus large, par exemple un corpus de FAQ santé ou un dataset de questions médicales annotées.

## Installation

```bash
pip install -r requirements.txt
```

## Entraînement

```bash
python src/train.py \
  --data_path data/sample/medical_question_pairs.csv \
  --model_name bert-base-uncased \
  --output_dir models/bert-medical-sqr \
  --epochs 3 \
  --batch_size 8 \
  --max_length 128
```

## Évaluation

```bash
python src/evaluate.py \
  --data_path data/sample/medical_question_pairs.csv \
  --model_dir models/bert-medical-sqr
```

## Prédiction

```bash
python src/predict.py \
  --model_dir models/bert-medical-sqr \
  --question "What are the signs of high blood pressure?" \
  --candidate_file data/sample/medical_question_pairs.csv
```

## Résultat attendu

Le modèle renvoie un score de similarité pour chaque paire de questions.  
Les questions candidates peuvent ensuite être classées afin de proposer la réponse associée à la question médicale la plus proche.

## Remarque

Ce dépôt est une version démonstrative destinée à montrer la logique du projet : préparation des données, fine-tuning BERT, évaluation et inférence. Pour un usage réel dans le domaine médical, il faudrait utiliser un corpus validé, vérifier la qualité des annotations et intégrer des règles strictes de validation médicale.
