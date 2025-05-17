# Guide Technique : Architecture et Fonctionnement 

Ce document fournit une explication technique détaillée du système de recommandation JibJob pour les développeurs et data scientists.

## Architecture générale

Le système de recommandation JibJob utilise une architecture hybride combinant :
- Traitement du langage naturel (NLP) avec BERT
- Analyse de sentiment
- Réseaux de neurones convolutifs sur graphes (GCN)

### Dépendances principales

- **PyTorch** : Framework d'apprentissage profond
- **PyTorch Geometric** : Extension pour le traitement de graphes
- **Transformers** : Bibliothèque pour les modèles BERT
- **FastAPI** : Pour l'API REST

## Composants clés

### 1. Module de simulation de données (`data_simulation.py`)

Génère trois ensembles de données synthétiques :
- Informations utilisateurs
- Informations emplois
- Interactions utilisateur-emploi

Le générateur crée des données qui imitent les distributions réelles avec différents niveaux d'engagement et de satisfaction.

### 2. Ingénierie des caractéristiques (`feature_engineering.py`)

#### Classe `FeatureEngineer`
- **Initialisation**: Charge le modèle BERT et l'analyseur de sentiment
- **get_bert_embedding()**: Génère des vecteurs d'embedding pour le texte
- **process_interactions()**: Ajoute des scores de sentiment et calcule les ratings améliorés
- **process_jobs()**: Génère des embeddings pour les descriptions d'emploi

#### Points techniques importants :
- Utilise la classe `SentimentAnalyzer` pour le traitement des commentaires
- Normalise les valeurs pour assurer la cohérence
- Combine les notes explicites et les scores de sentiment avec des poids configurables

### 3. Analyse de sentiment (`sentiment_analysis_module.py`)

Utilise un modèle DistilBERT fine-tuné pour la classification de sentiment.

#### Points techniques importants :
- Convertit les étiquettes POSITIF/NÉGATIF en scores continus entre -1 et +1
- Gère les cas de commentaires manquants ou vides

### 4. Construction de graphes (`graph_construction.py`)

Transforme les données tabulaires en une structure de graphe hétérogène pour l'apprentissage GCN.

#### Points techniques importants :
- Crée une structure `HeteroData` de PyTorch Geometric 
- Ajoute des arêtes bidirectionnelles pour faciliter la propagation des messages
- Maintient des mappings entre IDs et indices pour la récupération des recommandations

### 5. Modèle GCN (`gcn_model.py`)

Implémente un réseau neuronal GCN spécialisé pour la prédiction de liens dans un graphe hétérogène.

#### Classe `HeteroGCNLinkPredictor`:
- **encode()**: Transforme les caractéristiques des nœuds à travers les couches GCN
- **decode()**: Prédit les scores de préférence pour les paires utilisateur-emploi
- **forward()**: Combine encode et decode pour l'entraînement
- **get_embeddings()**: Obtient les embeddings finaux pour tous les nœuds

#### Architecture du modèle :
- Couches GCN multiples avec normalisation par lots
- Propagation de messages entre différents types de nœuds
- Réseau MLP final pour la prédiction de liens

### 6. Entraînement du modèle (`train_gcn.py`)

#### Points techniques importants :
- Division des données en ensembles d'entraînement, validation et test
- Optimisation avec Adam et fonction de perte BCE
- Early stopping basé sur la perte de validation
- Évaluation avec AUC-ROC, précision et rappel

### 7. Module de recommandation (`recommender.py`)

#### Classe `JobRecommender`:
- **get_recommendations()**: Génère des recommandations personnalisées pour un utilisateur
- Utilise les embeddings précalculés pour des inférences rapides
- Peut filtrer les emplois déjà consultés

#### Stratégie de recommandation :
- Calcule des scores pour toutes les paires utilisateur-emploi possibles
- Trie les emplois par score de préférence prédit
- Retourne les N meilleurs emplois comme recommandations

### 8. API de service (`api.py`)

Expose les fonctionnalités de recommandation via une API REST utilisant FastAPI.

## Flux de données

1. Les données brutes sont transformées en caractéristiques enrichies
2. Les caractéristiques sont utilisées pour construire un graphe hétérogène
3. Le graphe est divisé pour l'entraînement et l'évaluation
4. Le modèle GCN est entraîné pour prédire les liens manquants
5. Les embeddings du modèle entraîné sont utilisés pour les recommandations

## Optimisation et mise à l'échelle

Le système est conçu pour être performant même avec un grand nombre d'utilisateurs et d'emplois :

- Les embeddings sont précalculés et mis en cache
- Le modèle encode les utilisateurs et les emplois séparément pour une inférence rapide
- L'API permet des requêtes parallèles pour servir plusieurs utilisateurs

## Extensions possibles

1. **Intégration de caractéristiques temporelles**: Pour capturer les tendances saisonnières et l'évolution des préférences
2. **Filtrage par contexte**: Ajouter des filtres géographiques ou de disponibilité
3. **Apprentissage continu**: Mise à jour incrémentale du modèle à mesure que de nouvelles interactions sont enregistrées
4. **Explications des recommandations**: Ajouter un module qui explique pourquoi un emploi spécifique a été recommandé
