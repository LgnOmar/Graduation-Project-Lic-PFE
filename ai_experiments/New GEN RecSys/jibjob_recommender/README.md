# JibJob Recommendation System

## Project Overview

This project implements a comprehensive recommendation system for the JibJob platform, which is designed to connect clients posting small jobs/gigs with professionals seeking such work.

The system uses Graph Convolutional Networks (GCNs) to provide relevant job recommendations to professional users based on their profile information, selected work categories, and location. It builds a heterogeneous graph that represents the relationships between professionals, jobs, categories, and locations, then uses graph neural networks to learn meaningful representations and make accurate recommendations.

## Key Features

- **Graph-based recommendation** using GCNs/HeteroGCNs for enhanced recommendation quality
- **BERT embeddings** for rich text content analysis of profiles and job descriptions
- **Location-aware recommendation** with distance consideration and geographic filters
- **Category-based job matching** to connect professionals with relevant opportunities
- **Heterogeneous graph construction** modeling complex relationships between entities
- **Advanced evaluation metrics** for measuring recommendation quality
- **Command-line interfaces** for training and inference
- **Comprehensive test framework** with integration, continuous integration, and A/B testing capabilities

## Project Structure

```
jibjob_recommender/
├── sample_data/                   # Sample data and data generation scripts
│   ├── generate_sample_data.py    # Script to generate synthetic data for testing
│   ├── locations.csv              # Location data with coordinates
│   ├── categories.csv             # Job categories
│   ├── users.csv                  # User profiles (professionals and employers)
│   ├── professional_categories.csv # Category interests for professionals
│   ├── jobs.csv                   # Job postings with details
│   └── job_applications.csv       # Historical job applications (for training)
│
├── jibjob_recommender_system/     # Main application package
│   ├── config/                    # Configuration files and parsers
│   │   ├── config_loader.py       # Load and validate configuration
│   │   └── settings.yaml          # System configuration parameters
│   │
│   ├── data_handling/             # Data loading, validation, preprocessing
│   │   ├── data_loader.py         # Load data from various sources
│   │   ├── data_validator.py      # Validate data format and content
│   │   └── preprocessor.py        # Clean and preprocess text data
│   │
│   ├── feature_engineering/       # Creating features for models
│   │   ├── feature_orchestrator.py # Coordinate feature generation
│   │   ├── text_embedder.py       # Generate BERT embeddings
│   │   ├── location_features.py   # Process geographical data
│   │   └── graph_features.py      # Prepare features for graph nodes/edges
│   │
│   ├── graph_construction/        # Logic to build the graph for GCNs
│   │   ├── graph_builder.py       # Construct graph data structures
│   │   └── heterogeneous_graph_def.py # Define heterogeneous graph schema
│   │
│   ├── models/                    # Model definitions
│   │   ├── base_recommender.py    # Abstract base class for recommenders
│   │   └── gcn_recommender.py     # GCN/HeteroGCN implementations
│   │
│   ├── training/                  # Training scripts and utilities
│   │   ├── trainer.py             # Model training loops and procedures
│   │   └── train_gcn.py           # Command-line script for training
│   │
│   ├── inference/                 # Logic for generating recommendations
│   │   ├── recommender_service.py # Service for generating recommendations
│   │   └── predict.py             # Command-line interface for predictions
│   │
│   ├── evaluation/                # Evaluation metrics and procedures
│   │   └── evaluation.py          # Recommendation quality metrics
│   │
│   ├── utils/                     # Common utility functions
│   │   ├── helpers.py             # General helper functions
│   │   └── logging_config.py      # Logging configuration
│   │
│   └── main.py                    # Main script to run the complete pipeline
│
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                         # Unit and integration tests
│   ├── test_integration.py        # Integration tests between components
│   ├── test_continuous_integration.py # End-to-end pipeline tests
│   └── test_ab_testing.py         # A/B testing framework and tests
│
├── run_tests.py                   # Script to discover and run all tests
└── requirements.txt               # Python package dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:

   ```
   git clone [repository_url]
   cd jibjob_recommender
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv jibjob_env
   # On Linux/Mac
   source jibjob_env/bin/activate
   # On Windows
   jibjob_env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Data Preparation

You can use your own data or generate synthetic data for testing:

1. Generate sample data:

   ```
   cd sample_data
   python generate_sample_data.py --num-users 1000 --num-jobs 500 --output-dir .
   ```

2. Or place your own data in the `sample_data` directory according to the expected schema.

## Usage

### Training a Recommendation Model

1. Train a model using the dedicated training script:

   ```
   cd jibjob_recommender_system/training
   python train_gcn.py --data-dir ../../sample_data --model-type heterogcn --epochs 100 --output-dir ../../saved_models
   ```

2. Or use the main script for the complete pipeline:
   ```
   cd jibjob_recommender_system
   python main.py --data-dir ../sample_data --output-dir ../outputs
   ```

### Testing the System

Run all tests with the provided test runner script:

```
python run_tests.py
```

This will discover and execute all test modules in the `tests` directory and report results.

### Generating Recommendations

Use the prediction script to generate recommendations:

```
cd jibjob_recommender_system/inference
python predict.py --user-id user_0001 --top-k 10 --filter-location --max-distance 50.0 --output recommendations.json
```

For a new user not in the training data:

```
python predict.py --new-user --profile-text "Experienced software developer with skills in Python and JavaScript" --categories cat_001 cat_002 --latitude 40.7128 --longitude -74.0060
```

### Command-line Options

The prediction script supports various options:

- `--user-id`: ID of an existing user
- `--new-user`: Flag to indicate recommendation for a new user
- `--profile-text`: Profile description for a new user
- `--categories`: Categories of interest for a new user
- `--latitude` and `--longitude`: Location coordinates for a new user
- `--top-k`: Number of recommendations to return
- `--filter-location`: Whether to filter by location
- `--max-distance`: Maximum distance in kilometers
- `--output`: Path to save recommendations
- `--format`: Output format ('json' or 'csv')

## Evaluation Metrics

The system includes comprehensive evaluation metrics:

- Hit Rate (HR@k)
- Precision (P@k) and Recall (R@k)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Mean Average Precision (MAP@k)
- Diversity, Serendipity, and Coverage

## Configuration

Main configuration options are in `config/settings.yaml`. Key settings include:

- Model parameters (embedding dimensions, number of layers)
- Training parameters (learning rate, batch size, epochs)
- Graph construction options
- Location-based filtering settings
- BERT embedding model selection

## Testing Framework

The JibJob recommendation system includes a comprehensive testing framework designed to ensure the reliability and performance of the entire system. The framework includes:

### Integration Testing

The `test_integration.py` module tests interactions between different components of the system:

- **Data to Feature Pipeline**: Tests the flow from data loading through feature processing
- **Feature to Model Pipeline**: Tests feature processing through model training
- **Model to Inference Pipeline**: Tests trained models in recommendation scenarios
- **Full Pipeline Testing**: End-to-end tests from raw data to recommendations

### Continuous Integration Testing

The `test_continuous_integration.py` module provides end-to-end testing for the system pipeline:

- **Main Pipeline Execution**: Tests full system under normal conditions
- **Data Change Resilience**: Tests system behavior when data changes
- **API Service Testing**: Tests the recommendation API interface
- **Incremental Training**: Tests model updating with new data

### A/B Testing Framework

The `test_ab_testing.py` module implements a framework for comparing recommendation strategies:

- **Variant Assignment**: Testing user assignment to test variants
- **Model Variant Training**: Testing training procedures for different model variants
- **Variant Evaluation**: Comparing recommendation quality between variants
- **User Feedback Simulation**: Testing system with simulated user feedback

### Running Tests

The `run_tests.py` script provides a convenient way to run all tests:

```
python run_tests.py
```

This script:

- Automatically discovers all tests in the `tests` directory
- Runs the complete test suite
- Reports test results with detailed output

### Test Classes

#### Integration Testing (`TestIntegration` class)

- `test_data_to_feature_pipeline`: Tests data loading to feature processing flow
- `test_feature_to_model_pipeline`: Tests feature processing to model training flow
- `test_model_to_inference_pipeline`: Tests model training to inference flow
- `test_full_pipeline`: Tests complete end-to-end functionality

#### Continuous Integration Testing (`TestContinuousIntegration` class)

- `test_end_to_end_pipeline`: Tests the main pipeline execution
- `test_with_data_changes`: Tests system resilience to data modifications
- `test_api_service`: Tests recommendation API functionality
- `test_incremental_training`: Tests model updating with new data

#### A/B Testing Framework (`ABTestingFramework` and `TestABTesting` classes)

- `ABTestingFramework`: Class for conducting A/B experiments

  - User variant assignment
  - Model training for different variants
  - Recommendation generation for variants
  - Evaluation and comparison of variants

- `TestABTesting`: Tests the A/B testing framework functionality
  - `test_variant_assignment`: Tests user assignment to variants
  - `test_variant_training`: Tests model training for variants
  - `test_variant_recommendations`: Tests recommendation generation
  - `test_variant_evaluation`: Tests variant comparison

## Future Work

- Integration with a web API
- Real-time recommendation updates
- Multi-objective optimization for recommendations
- User feedback integration

## License

MIT License

## Contributors

JibJob AI Team
