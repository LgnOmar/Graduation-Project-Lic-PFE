# JibJob Recommendation System

A hybrid recommendation system for JibJob, an Algerian mobile platform for small jobs. The system combines sentiment analysis, BERT embeddings, and Graph Neural Networks to provide personalized job recommendations to users.

## Key Features

- **Sentiment Analysis**: BERT-based analysis of user comments to extract implicit feedback
- **Content-Based Features**: BERT embeddings for job descriptions to capture semantic meaning
- **Graph Neural Network**: Heterogeneous GCN that learns from both content and user interactions
- **API Deployment**: FastAPI server for real-time recommendation delivery
- **Comprehensive Pipeline**: End-to-end workflow from data processing to serving recommendations
- **Algerian-Specific Data**: Enhanced with real-world Algerian job market data covering all 58 wilayas

## Technical Architecture

![JibJob Architecture](https://via.placeholder.com/800x400?text=JibJob+Recommendation+System+Architecture)

### Components

- **Sentiment Analysis Module**: Uses BERT to transform qualitative user feedback into quantitative scores
- **Feature Engineering**: Creates embeddings and processes interaction data
- **Graph Construction**: Builds heterogeneous graph connecting users and jobs
- **GCN Model**: Learns representations of users and jobs for recommendation
- **Recommender Service**: Generates personalized recommendations using the trained model
- **API Layer**: Exposes the recommendation functionality via REST API
- **Algerian Context Integration**: Incorporates location, cultural, and linguistic awareness

## Algerian Data Enhancement

The system includes realistic data for the Algerian market:

- **Location Coverage**: All 58 Algerian wilayas (provinces) for location-based recommendations
- **Local Job Categories**: 10 job categories with 60+ popular services tailored to the Algerian gig economy
- **Authentic Profiles**: User profiles with realistic skills, preferences, and language abilities
- **Cultural Context**: Recommendations accounting for regional preferences and practices

For more details on the Algerian data enhancement, see [ALGERIAN_DATA_ENHANCEMENT.md](ALGERIAN_DATA_ENHANCEMENT.md).

## Demo API

For demonstration purposes, we've included a simplified version of the API that doesn't require a pre-trained model:

```bash
# Run the demo API directly
python src/demo_api.py

# Or use the launcher script for a better experience
python launch_demo_api.py
```

The launcher script:
- Checks for required dependencies
- Launches the API server
- Opens API documentation in your browser
- Shows sample commands for testing

## Quick Start

### Option 1: Using the Demo API

1. Clone this repository
2. Run the launcher script:
   ```bash
   python launch_demo_api.py
   ```
3. Access the API documentation at http://localhost:8000/docs
4. Test the recommendations:
   ```bash
   python test_recommendations.py
   ```

### Option 2: Full Training Pipeline

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate sample data (if needed):
   ```bash
   python src/data_simulation.py
   ```
4. Process features:
   ```bash
   python src/feature_engineering.py
   ```
5. Train the GCN model:
   ```bash
   python src/train_gcn.py
   ```
6. Run the API server:
   ```bash
   python src/api.py
   ```

## Project Structure

```
jibjob_recommendation/
├── data/                     # Data directory
│   ├── interactions_df.csv   # User-job interactions
│   ├── jobs_df.csv           # Job information
│   ├── users_df.csv          # User information
│   ├── job_embeddings.pkl    # BERT embeddings for jobs
│   └── processed_interactions.csv # Interactions with sentiment scores
├── models/                   # Trained models
│   └── best_model.pt         # Best GCN model checkpoint (after training)
├── src/                      # Source code
│   ├── api.py                # FastAPI application
│   ├── data_simulation.py    # Generate synthetic data
│   ├── demo_api.py           # Demo API without model requirement
│   ├── feature_engineering.py # Create features from raw data
│   ├── gcn_model.py          # GCN model architecture
│   ├── graph_construction.py # Build heterogeneous graph
│   ├── pipeline.py           # Full processing pipeline
│   ├── recommender.py        # Generate recommendations
│   ├── sentiment_analysis_module.py # BERT-based sentiment analysis
│   ├── test_modules.py       # Unit tests for modules
│   ├── test_system.py        # System integration tests
│   └── train_gcn.py          # Train the GCN model
├── launch_demo_api.py        # Launcher script for demo API
├── test_recommendations.py   # Script to test recommendations
├── TESTING_REPORT.md         # Detailed testing report
├── rapport.tex               # Technical report (French)
└── README.md                 # This file
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jibjob-recommendation.git
cd jibjob-recommendation
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate

# Linux/MacOS
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

The complete pipeline can be run with a single command:
```bash
python src/pipeline.py
```

This will:
1. Generate sample data (if needed)
2. Process features (sentiment analysis and BERT embeddings)
3. Train the GCN model
4. Ready the system for recommendations

### Individual Components

#### 1. Generate Sample Data
```bash
python src/data_simulation.py
```

#### 2. Process Features and Build Graph
```bash
python src/feature_engineering.py
```

#### 3. Train the Model
```bash
python src/train_gcn.py
```

#### 4. Start the API Server
```bash
python src/api.py
```

The API will be available at `http://localhost:8000`.

You can access the API documentation at `http://localhost:8000/docs`.

## Testing

To run the comprehensive test suite:
```bash
python src/test_system.py
```

This will test:
- Data loading and processing
- Model existence and validity
- Recommendation generation
- API endpoints
- Performance metrics

## API Endpoints

### Get Recommendations

```
GET /recommendations/{user_id}
```

Parameters:
- `user_id` (path): ID of the user to get recommendations for
- `top_n` (query, optional): Number of recommendations to return (default: 10)

Response:
```json
{
  "job_ids": ["job123", "job456", "job789"],
  "scores": [0.95, 0.87, 0.76]
}
```

## Model Performance

The system was evaluated on a test set, with the following metrics:
- AUC: ~0.85-0.90
- Precision: ~0.75-0.80
- Recall: ~0.70-0.75

Results may vary depending on the data used.

## Further Development

Future improvements could include:
- Adding temporal features to capture evolving user preferences
- Implementing A/B testing framework
- Adding explainability features for recommendations
- Scaling the system for production use with larger datasets

## License

MIT License
