# Exemples concrets pour JibJob

## Exemple d'entrées de données

### Utilisateurs
| user_id  | description_profil_utilisateur_anglais |
|----------|----------------------------------------|
| user_123 | Experienced plumber with 5 years of work in residential and commercial settings. |
| user_456 | College student available for part-time work, experienced in customer service. |
| user_789 | Retired teacher looking for occasional gardening and tutoring opportunities. |

### Emplois
| job_id  | description_mission_anglais | categorie_mission |
|---------|---------------------------|------------------|
| job_101 | Need help fixing a leaking kitchen sink and installing a new faucet. | Plumbing |
| job_202 | Looking for someone to tutor my 10-year-old in mathematics twice a week. | Teaching |
| job_303 | Need assistance with garden maintenance including pruning and planting new flowers. | Gardening |

### Interactions
| user_id  | job_id  | rating_explicite | commentaire_texte_anglais |
|----------|---------|-----------------|---------------------------|
| user_123 | job_101 | 5.0 | Very straightforward plumbing job, good pay and nice homeowner. |
| user_456 | job_202 | 4.0 | The child was attentive and the parents were supportive. |
| user_789 | job_303 | 3.0 | Garden was larger than described but the work was satisfying. |

## Exemple d'analyse de sentiment

| Commentaire | Score calculé | Étiquette |
|-------------|---------------|-----------|
| "Very straightforward plumbing job, good pay and nice homeowner." | +0.92 | POSITIF |
| "The child was attentive and the parents were supportive." | +0.78 | POSITIF |
| "Garden was larger than described but the work was satisfying." | +0.25 | POSITIF |
| "The job took much longer than expected and the payment was delayed." | -0.65 | NÉGATIF |
| "Instructions were unclear and the working conditions were poor." | -0.88 | NÉGATIF |

## Exemple de calcul de score amélioré

Pour l'utilisateur user_123 et l'emploi job_101:
- Note explicite: 5.0 (sur 5) → Note normalisée: 1.0
- Score de sentiment: +0.92 → Sentiment normalisé: 0.96
- Poids de la note: 0.7
- Poids du sentiment: 0.3

Score amélioré = (0.7 × 1.0) + (0.3 × 0.96) = 0.7 + 0.288 = 0.988

## Exemple de recommandations générées

Pour l'utilisateur user_123 (plombier expérimenté):

| Job ID | Description | Score prédit |
|--------|-------------|--------------|
| job_105 | Bathroom renovation including installing new shower and toilet. | 0.95 |
| job_107 | Fix multiple water leaks in basement plumbing. | 0.87 |
| job_110 | Install new water heater in residential home. | 0.82 |
| job_112 | Repair outdoor irrigation system with multiple broken pipes. | 0.76 |
| job_115 | Help with kitchen remodeling including sink and dishwasher installation. | 0.71 |

Pour l'utilisateur user_456 (étudiant à temps partiel):

| Job ID | Description | Score prédit |
|--------|-------------|--------------|
| job_209 | Weekend cashier position at local bookstore. | 0.88 |
| job_215 | After-school tutoring for middle school students. | 0.85 |
| job_220 | Part-time customer service at neighborhood coffee shop. | 0.79 |
| job_225 | Social media management for small local business. | 0.72 |
| job_230 | Data entry work that can be done remotely. | 0.68 |
