# Exécution du Système de Recommandation JibJob

Ce guide vous explique comment exécuter et tester le système de recommandation JibJob sur votre machine locale.

## Prérequis

Avant de commencer, assurez-vous d'avoir :
- Python 3.8 ou supérieur
- Accès à un GPU (recommandé pour l'entraînement mais non obligatoire)
- 8 Go de RAM minimum

## Installation

1. Clonez le dépôt (ou téléchargez le code source)
2. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Exécution du pipeline complet

Le pipeline complet exécute les étapes suivantes :
1. Simulation des données
2. Ingénierie des caractéristiques
3. Entraînement du modèle

Pour exécuter l'ensemble du pipeline, utilisez la commande :

```bash
python src/pipeline.py
```

## Exécution des étapes individuelles

Vous pouvez également exécuter chaque étape séparément :

### 1. Génération des données de simulation

```bash
python src/data_simulation.py
```

Cela créera trois fichiers CSV dans le dossier `data/` :
- `users_df.csv` : données des utilisateurs
- `jobs_df.csv` : données des emplois
- `interactions_df.csv` : interactions entre utilisateurs et emplois

### 2. Ingénierie des caractéristiques

```bash
python src/feature_engineering.py
```

Cette étape effectue l'analyse de sentiment sur les commentaires et génère des embeddings BERT pour les descriptions d'emploi. Elle crée :
- `processed_interactions.csv` : interactions avec scores de sentiment
- `job_embeddings.pkl` : embeddings vectoriels des emplois

### 3. Entraînement du modèle GCN

```bash
python src/train_gcn.py
```

Cette étape entraîne le modèle GCN et sauvegarde le meilleur modèle dans le dossier `models/`.

## Démarrer le serveur API

Une fois le modèle entraîné, vous pouvez démarrer le serveur API pour faire des recommandations :

```bash
python src/api.py
```

Le serveur API sera accessible à l'adresse http://localhost:8000.

## Tester avec la démo interactive

Pour tester le système avec une interface utilisateur simple, exécutez :

```bash
python src/demo.py
```

Cela vous permettra de :
- Obtenir des recommandations pour différents utilisateurs
- Voir les détails des emplois
- Simuler des interactions utilisateur

## Notes importantes

- La première exécution téléchargera les modèles BERT, ce qui peut prendre un certain temps
- L'entraînement du modèle peut être lent sur CPU
- Pour les grandes instances, considérez d'ajuster les paramètres dans les fichiers source
