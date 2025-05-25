"""
Script to run the full JibJob recommendation system pipeline.
This handles the Python import issues and runs the entire system.
"""
import os
import sys
import subprocess
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run JibJob Recommendation System')
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, 
                        help='Mode: train to train the model, predict to make recommendations')
    parser.add_argument('--data-dir', type=str, default='sample_data',
                        help='Directory containing sample data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for output files')
    
    # Prediction options
    parser.add_argument('--user-id', type=str, help='User ID for recommendations')
    parser.add_argument('--top-k', type=int, default=10, 
                        help='Number of recommendations to return')
    
    # Training options
    parser.add_argument('--model-type', choices=['gcn', 'heterogcn'], default='heterogcn',
                        help='Type of GCN model to use')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    
    return parser.parse_args()

def run_train_script(args):
    """Run the training script with proper environment setup"""
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(current_dir, 'jibjob_recommender_system', 'training', 'train_gcn.py')
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create a batch script to run with proper environment setup
    batch_script = """
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, r'{project_dir}')

# Now we can import from our package
from jibjob_recommender_system.training.train_gcn import main

if __name__ == "__main__":
    # Call the main function with required arguments
    main(data_dir=r"{data_dir}", 
         model_type="{model_type}", 
         epochs={epochs}, 
         output_dir=r"{output_dir}")
    """.format(
        project_dir=current_dir,
        data_dir=data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        output_dir=output_dir
    )
    
    # Write the batch script to a temporary file
    batch_file = os.path.join(current_dir, 'run_train_temp.py')
    with open(batch_file, 'w') as f:
        f.write(batch_script)
    
    # Run the batch script
    print(f"Running training with {args.model_type} model for {args.epochs} epochs...")
    try:
        subprocess.run([sys.executable, batch_file], check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    
    # Clean up
    if os.path.exists(batch_file):
        os.remove(batch_file)

def run_predict_script(args):
    """Run the prediction script with proper environment setup"""
    # Get absolute paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(current_dir, 'jibjob_recommender_system', 'inference', 'predict.py')
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not args.user_id:
        print("Error: --user-id is required for prediction mode")
        return
    
    # Create a batch script to run with proper environment setup
    batch_script = """
import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, r'{project_dir}')

# Now we can import from our package
from jibjob_recommender_system.inference.predict import main

if __name__ == "__main__":
    # Call the main function with required arguments
    main(user_id="{user_id}", 
         top_k={top_k}, 
         data_dir=r"{data_dir}", 
         output=r"{output_dir}/recommendations.json")
    """.format(
        project_dir=current_dir,
        user_id=args.user_id,
        top_k=args.top_k,
        data_dir=data_dir,
        output_dir=output_dir
    )
    
    # Write the batch script to a temporary file
    batch_file = os.path.join(current_dir, 'run_predict_temp.py')
    with open(batch_file, 'w') as f:
        f.write(batch_script)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the batch script
    print(f"Getting recommendations for user {args.user_id}...")
    try:
        subprocess.run([sys.executable, batch_file], check=True)
        print(f"Recommendations generated successfully! Check {os.path.join(output_dir, 'recommendations.json')}")
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction: {e}")
    
    # Clean up
    if os.path.exists(batch_file):
        os.remove(batch_file)

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run the appropriate script based on the mode
    if args.mode == 'train':
        run_train_script(args)
    elif args.mode == 'predict':
        run_predict_script(args)

if __name__ == "__main__":
    main()
