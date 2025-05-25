# JibJob Recommendation System

## 🚀 Overview
JibJob est un système intelligent de recommandation pour une plateforme mobile qui connecte les particuliers ayant besoin d’aide pour des tâches ponctuelles (déménagement, jardinage, réparations…) avec des travailleurs flexibles, principalement pour le marché algérien.

Ce projet utilise des techniques avancées de machine learning (BERT, GCN, analyse de sentiment) pour fournir des recommandations personnalisées et améliorer l’expérience utilisateur.

---

## ✨ Features
- **Compréhension sémantique (BERT)** : analyse avancée des descriptions de jobs.
- **Analyse de sentiment** : prise en compte des commentaires utilisateurs.
- **Recommandation par graphe (GCN)** : modélisation des relations complexes utilisateurs-jobs.
- **API rapide (FastAPI)** : intégration facile avec des applications mobiles/web.
- **Support multilingue** : arabe, français, anglais.
- **Génération de données synthétiques** : pour le développement et les tests.

---

## 🖥️ Installation

### Prérequis
- Python 3.8+
- PyTorch 1.9+
- (Optionnel) GPU compatible CUDA

### Installation rapide
```powershell
# Clonez le dépôt
git clone https://github.com/yourusername/jibjob-recommendation.git
cd jibjob-recommendation

# Créez un environnement virtuel
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Installez les dépendances
pip install -r requirements.txt
```

---

## 🏁 Démarrage rapide

### Générer des données d’exemple
```powershell
python examples/sample_data_demo.py
```

### Entraîner un modèle sur les données d’exemple
```powershell
python examples/train_with_sample_data.py
```

### Lancer l’API (FastAPI)
```powershell
uvicorn src.api.main:app --reload
```

### Tester les recommandations
```powershell
python examples/basic_recommendation_demo.py
```

---

## 📁 Structure du projet

```
jibjob-recommendation/
├── src/           # Code source principal
│   ├── models/    # Modèles ML (BERT, GCN, etc.)
│   ├── data/      # Prétraitement et gestion des données
│   ├── utils/     # Outils divers (metrics, visualisation...)
│   └── api/       # Service API (FastAPI)
├── examples/      # Scripts d’exemple et de test
├── tests/         # Suite de tests unitaires
├── sample_data/   # Données d’exemple
└── docs/          # Documentation
```

---

## 🧑‍💻 Utilisation API

### Lancer le serveur
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Documentation interactive
- Swagger UI : [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc : [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints principaux
- `GET /` : Infos API
- `GET /health` : Statut API
- `POST /recommend` : Recommandations pour un utilisateur
- `POST /recommend-batch` : Recommandations pour plusieurs utilisateurs
- `GET /job-similarity/{job_id}` : Jobs similaires
- `POST /analyze-sentiment` : Analyse de sentiment
- `POST /load-model` : Charger un modèle

### Exemple d’appel API (Python)
```python
import requests

# Recommandations pour un utilisateur
resp = requests.post(
    "http://localhost:8000/recommend",
    json={"user_id": "user_1", "top_k": 5, "exclude_rated": True}
)
print(resp.json())

# Jobs similaires
resp = requests.get("http://localhost:8000/job-similarity/job_10?top_k=5")
print(resp.json())
```

---

## 📊 Évaluation & Métriques
Le système est évalué sur :
- **Precision@K** : proportion de jobs recommandés pertinents
- **Recall@K** : proportion de jobs pertinents recommandés
- **NDCG@K** : qualité du classement des recommandations
- **MAE** : erreur absolue moyenne sur les notes
- **RMSE** : racine de l’erreur quadratique moyenne

Exemple :
```python
metrics = recommender.evaluate(
    test_interactions=test_interactions,
    user_id_col='user_id',
    job_id_col='job_id',
    rating_col='rating',
    top_k=10
)
print(metrics)
```

---

## 🧪 Tests
```powershell
pytest
```

---

## 🤝 Contribution
Les contributions sont les bienvenues !
1. Forkez le repo
2. Créez une branche : `git checkout -b feature-ma-branche`
3. Commitez : `git commit -am 'Ajout fonctionnalité'`
4. Pushez : `git push origin feature-ma-branche`
5. Ouvrez une Pull Request

---

## 📄 Licence
MIT License. Voir le fichier LICENSE.

---

## 📬 Contact
Pour toute question ou suggestion : [your-email@example.com](mailto:your-email@example.com)

---

*JibJob Recommendation System – Connecter les talents et les besoins en Algérie.*
