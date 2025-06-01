# JibJobRecSys

A gig/job recommendation system using Heterogeneous Graph Neural Networks and BERT Embeddings, with a fully self-generated dataset.

## Directory Structure

```
JibJobRecSys/
├── data/
│   ├── raw/
│   └── generated/
│       ├── categories.csv
│       ├── professionals.csv
│       ├── professional_selected_categories.csv
│       ├── clients.csv
│       ├── jobs.csv
│       ├── job_required_categories.csv
│       └── interactions.csv
├── notebooks/
│   ├── 00_dataset_generation.ipynb
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_feature_engineering_bert.ipynb
│   ├── 03_graph_construction.ipynb
│   ├── 04_model_development_hetgcn.ipynb
│   ├── 05_training_and_evaluation.ipynb
│   └── 06_recommendation_generation.ipynb
├── src/
│   ├── dataset_generator/
│   │   ├── __init__.py
│   │   ├── categories_generator.py
│   │   ├── users_generator.py
│   │   ├── jobs_generator.py
│   │   └── interactions_generator.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── text_embedders.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graph_builder.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── hetgcn_recommender.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── negative_sampler.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── common.py
│   └── main.py
├── tests/
├── requirements.txt
└── README.md
```

## How to Use

1. **Generate the dataset**: Run the scripts in `src/dataset_generator/` or use `00_dataset_generation.ipynb`.
2. **Explore and preprocess**: Use `01_eda_and_preprocessing.ipynb`.
3. **Generate BERT embeddings**: Use `02_feature_engineering_bert.ipynb` and `src/features/text_embedders.py`.
4. **Build the graph**: Use `03_graph_construction.ipynb` and `src/graph/graph_builder.py`.
5. **Develop and train the model**: Use `04_model_development_hetgcn.ipynb`, `src/models/hetgcn_recommender.py`, and `src/training/`.
6. **Evaluate**: Use `05_training_and_evaluation.ipynb` and `src/evaluation/metrics.py`.
7. **Generate recommendations**: Use `06_recommendation_generation.ipynb`.

## Requirements

See `requirements.txt` for all dependencies.

## Project Highlights

- Fully synthetic, reproducible dataset generation
- Heterogeneous GNN (HGTConv preferred) and BERT for text
- Modular, extensible, and scalable design
- Extensive evaluation and critical analysis

---

For detailed design, see the global prompt and code documentation in each module.
