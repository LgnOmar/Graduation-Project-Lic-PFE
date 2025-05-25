import os
import sys

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from jibjob_recommender_system.training.train_gcn import main

if __name__ == "__main__":
    # Call the main function with required arguments
    main(data_dir="sample_data", 
         model_type="heterogcn", 
         epochs=20, 
         output_dir="saved_models")
