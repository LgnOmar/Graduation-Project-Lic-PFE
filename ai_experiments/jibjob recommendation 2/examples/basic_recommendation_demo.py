"""
Basic demonstration of the JibJob recommendation system.

This script demonstrates how to:
1. Initialize the recommendation system
2. Load and preprocess data
3. Train the recommendation model
4. Generate recommendations for users
5. Find similar jobs
6. Analyze sentiment in comments

Usage:
    python basic_recommendation_demo.py
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.recommender import JobRecommender
from src.data.preprocessing import process_job_descriptions, normalize_ratings
from src.utils.visualization import plot_recommendation_quality


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Ensure logger is defined

def generate_sample_data(n_users=50, n_jobs=100, n_interactions=500):
    """Generate sample data with enriched job descriptions for demonstration purposes."""
    logger.info(f"Generating sample data with {n_users} users, {n_jobs} jobs, and {n_interactions} interactions (enriched descriptions)")
    
    # Create sample users
    users = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(1, n_users + 1)],
        'username': [f"User {i}" for i in range(1, n_users + 1)],
        'location': np.random.choice(['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida'], n_users)
    })
    
    job_categories = ['Plumbing', 'Painting', 'Gardening', 'Assembly', 'Tech Support', 
                      'Cleaning', 'Moving', 'Electrical', 'Tutoring', 'Delivery']
    
    job_templates = {
        "Plumbing": "Urgent need for a skilled plumber to {task}. Located in {location_type}. Key issues involve {keywords}. Previous experience with {specific_tool_or_system} is a plus.",
        "Painting": "Professional painter required for {task}. Project is for a {location_type}. Looking for a high-quality finish using {keywords}. Must be meticulous with {specific_tool_or_system}.",
        "Gardening": "Experienced gardener wanted for {task}. Property is a {location_type} with {specific_feature}. Tasks include {keywords}. Knowledge of {specific_tool_or_system} preferred.",
        "Assembly": "Help needed to assemble {task}. Item is {brand_or_type} for a {location_type}. Must bring own tools and follow instructions for {keywords}.",
        "Tech Support": "Tech support specialist to resolve {task}. Device is a {device_type} in a {location_type}. Issue relates to {keywords}. Fast diagnosis needed for {specific_tool_or_system}.",
        "Cleaning": "Reliable cleaner for {task} at a {location_type}. Focus areas are {keywords}. Using {specific_tool_or_system} products preferred.",
        "Moving": "Strong movers for {task}. Moving from {location_type} to another. Items include {keywords}. {specific_tool_or_system} such as dollies might be required. Handle with care.",
        "Electrical": "Certified electrician for {task}. Problem is with {specific_component} in a {location_type}. Safety paramount, dealing with {keywords} and {specific_tool_or_system}.",
        "Tutoring": "Knowledgeable tutor for {subject_area} for a {student_level} student. Topics include {keywords}. Goal is to {outcome}.",
        "Delivery": "Prompt delivery person for {task}. Pickup from {pickup_location_type} and deliver to {location_type}. Requires {vehicle_type_if_any} for {keywords}."
    }

    category_specifics = {
        "Plumbing": {
            "tasks": ["fix leaking sink tap", "repair bathroom toilet cistern", "install new shower head", "unclog kitchen drain pipe", "inspect water heater functionality", "replace old galvanized pipes"],
            "keywords": ["pipes, drains, water pressure, leaks, fixtures", "valves, seals, gaskets, soldering, threading", "water heater diagnosis, sump pump maintenance", "blockages, flow issues, emergency plumbing repair", "sewer line inspection"],
            "tools": ["pipe wrenches, P-trap tools, basin wrench", "soldering torch, pipe cutters, deburring tool", "hydro-jetting machine, drain auger", "plumber's snake, inspection camera", "multimeter for water heaters"]
        },
        "Painting": {
            "tasks": ["paint living room walls (2 coats emulsion)", "paint exterior of a 2-story townhouse", "repaint kitchen cabinets and doors (satin finish)", "apply feature wallpaper to one accent wall", "stain and seal wooden deck", "prime and paint new drywall"],
            "keywords": ["color matching, surface preparation, priming, sanding", "trim work, ceiling painting, precise edging", "wood staining, varnish application, polyurethane", "protective coatings, even layers, spray painting techniques", "interior decor, exterior weatherproofing"],
            "tools": ["high-quality brushes (angled, flat), rollers (various naps), paint trays", "scaffolding, extension ladders, step ladders", "electric sander, sanding blocks", "airless paint sprayer, HVLP sprayer", "masking tape, drop cloths, paint can opener"]
        },
        "Gardening": {
            "tasks": ["regular lawn mowing and precise edge trimming", "prune rose bushes and overgrown fruit trees", "design and plant new perennial flower bed", "seasonal garden cleanup, leaf removal, and mulching", "install automated drip irrigation system for vegetable patch", "weed control and soil amendment"],
            "keywords": ["soil health improvement, organic fertilizer application, integrated pest management", "landscape design principles, native plant selection", "mulching benefits, weeding strategies, advanced pruning techniques", "seasonal planting schedules, lawn aeration and overseeding", "water-wise gardening"],
            "tools": ["petrol lawnmower, electric strimmer, edger", "secateurs, loppers, pruning saw, hedge trimmer", "spade, fork, hoe, rake, wheelbarrow", "irrigation system components, timers", "compost, mulch, soil testing kit"]
        },
        "Assembly": {
            "tasks": ["assemble IKEA Malm bed frame with drawers", "put together a new ergonomic office desk chair", "build a 5-shelf Billy bookshelf unit", "install flat-pack kitchen cabinets and wall units", "set up home gym multigym equipment and weight bench", "assemble outdoor patio furniture set"],
            "keywords": ["flat-pack furniture assembly, deciphering instruction manuals", "careful handling of delicate components, systematic parts checking", "ensuring structural integrity, secure and tight fittings", "organizing screws and hardware, step-by-step methodical assembly", "furniture layout and placement"],
            "tools": ["full Allen key set, comprehensive Phillips/flathead screwdriver set", "rubber mallet, spirit level, measuring tape", "cordless power drill with multiple bit attachments", "furniture clamps, stud finder (for wall units)", "protective gloves, safety glasses"]
        },
        "Tech Support": {
            "tasks": ["diagnose and fix slow computer performance and boot issues", "setup new wireless printer and scanner connection for multiple devices", "configure home WiFi network security (WPA3) and guest network", "troubleshoot smartphone battery drain and overheating issue", "install and configure comprehensive antivirus and firewall software suite", "data migration from old PC to new PC"],
            "keywords": ["advanced software troubleshooting, hardware diagnostics and component testing", "network configuration, IP addressing, DNS, DHCP, port forwarding", "malware and virus removal, data recovery strategies", "operating system updates, driver conflict resolution", "smart home device integration (Alexa, Google Home)"],
            "tools": ["bootable diagnostic software, remote desktop access tools", "USB drives with utilities, external hard drives for backup", "network cable tester, Wi-Fi analyzer app", "anti-static wrist strap, precision screwdriver kit", "software licenses, product keys"]
        },
        "Cleaning": {
            "tasks": ["deep clean a 2-bedroom, 2-bathroom apartment after tenancy", "regular weekly house cleaning service (3 hours)", "post-construction cleanup for a newly renovated kitchen", "office cleaning including desks, common areas, and restrooms (twice weekly)", "exterior window cleaning for a ground-floor retail storefront", "specialized carpet and upholstery steam cleaning"],
            "keywords": ["thorough dusting, meticulous vacuuming, effective mopping, detailed sanitizing", "kitchen appliance degreasing, bathroom descaling and mold removal", "hard floor care, carpet spot treatment and deep cleaning", "use of eco-friendly and non-toxic cleaning products, attention to high-touch surfaces", "window cleaning without streaks"],
            "tools": ["HEPA filter vacuum cleaner, steam mop, regular mop and bucket", "microfiber cloths, sponges, squeegees", "all-purpose cleaner, glass cleaner, descaler, degreaser", "specialized stain removers, carpet cleaning machine", "extendable pole for high areas, step ladder"]
        },
        "Moving": {
            "tasks": ["help move heavy antique sofa and double-door fridge to new address", "assist with full 3-bedroom apartment moving from 2nd floor (no elevator)", "professional loading and unloading of a 15ft moving truck/van", "securely pack fragile items (glassware, electronics) for long-distance transport", "rearrange existing furniture within a large house", "transport a piano safely"],
            "keywords": ["expert heavy lifting techniques, careful and strategic handling of valuable items", "professional packing strategies, appropriate wrapping materials selection", "efficient truck loading, space optimization, weight distribution", "disassembly and reassembly of large furniture items if needed", "navigating tight spaces and stairs"],
            "tools": ["heavy-duty moving dolly, furniture sliders, appliance hand truck", "high-quality furniture blankets, stretch wrap, bubble wrap", "strong packing tape, various sizes of moving boxes", "comprehensive toolset for furniture disassembly/reassembly", "lifting straps, forearm forklifts"]
        },
        "Electrical": {
            "tasks": ["install new LED ceiling light fixture and dimmer switch", "replace faulty or outdated electrical outlets and switches", "install and wire a Nest smart thermostat system", "troubleshoot intermittent circuit breaker tripping issue", "add a new dedicated electrical socket for high-power appliance", "inspect and certify home electrical wiring"],
            "keywords": ["safe electrical wiring practices, understanding circuits, load calculation, voltage testing", "adherence to local safety protocols and national electrical codes (NEC)", "light fixtures, smart switches, GFCI/AFCI outlets installation", "advanced electrical fault finding, diagnostic procedures", "panel upgrades"],
            "tools": ["insulated voltage tester, digital multimeter, clamp meter", "professional wire strippers, crimpers, various pliers, insulated screwdriver set", "fish tape, conduit bender", "safety gloves, insulated footwear, safety glasses", "knowledge of local electrical codes and permit requirements"]
        },
        "Tutoring": {
            "tasks": ["mathematics tutoring for high school calculus student", "English language conversation practice for business professionals", "physics exam preparation for college entrance", "introductory coding lessons in Python for beginners", "French language for beginners aiming for DELF A1", "university thesis writing support"],
            "keywords": ["specific curriculum support, effective exam preparation techniques", "homework assistance, in-depth concept clarification", "development of personalized learning plans, regular progress tracking and feedback", "interactive teaching methods, building student confidence"],
            "tools": ["relevant textbooks, supplementary workbooks", "access to online educational resources, interactive learning apps", "virtual whiteboard for online sessions, physical whiteboard for in-person", "curated practice exercises and mock exams", "subject-specific software (e.g., Mathematica, IDEs)"]
        },
        "Delivery": {
            "tasks": ["urgent same-day package pickup and city-wide express dropoff", "weekly scheduled grocery shopping and home delivery service", "collect and deliver confidential legal documents between offices", "evening hot food delivery from multiple restaurant partners", "bulk leaflet distribution to residential areas", "pharmaceutical delivery with temperature control"],
            "keywords": ["punctual and timely arrival, efficient urban route optimization using GPS", "careful and secure handling of goods, temperature control for perishable items", "obtaining signature confirmation, photographic proof of delivery", "managing multi-stop delivery schedules effectively", "customer service skills"],
            "tools": ["reliable and economical vehicle (scooter, car, small van)", "smartphone with updated GPS/maps application and communication capability", "insulated delivery bags/boxes (for food or medical supplies)", "hand trolley or cart for heavier/bulkier items", "card reader for payments on delivery (if applicable)"]
        }
    }
    
    job_data = []
    for i in range(1, n_jobs + 1):
        job_id_str = f"job_{i}"
        category = job_categories[(i - 1) % len(job_categories)]
        
        # Sensible defaults
        task = f"general {category.lower()} work"
        keywords = "standard requirements and details"
        specific_tool_or_system = "appropriate tools"
        location_type = np.random.choice(["a private home", "a residential apartment", "a commercial office", "a small business unit", "an industrial warehouse"])
        
        # Use more specific details from category_specifics if available
        if category in category_specifics:
            cat_details = category_specifics[category]
            task = np.random.choice(cat_details.get("tasks", [task]))
            keywords_list = cat_details.get("keywords", [keywords])
            # Pick one or two keyword phrases
            num_keywords_to_pick = np.random.randint(1, min(3, len(keywords_list) + 1))
            keywords = ", ".join(np.random.choice(keywords_list, num_keywords_to_pick, replace=False))

            # Pick a tool/system/brand detail if present
            tool_keys = ["tools", "brands", "vehicle_type_if_any", "device_type", "specific_component", "specific_feature", "student_level", "pickup_location_type"]
            available_tool_keys = [tk for tk in tool_keys if tk in cat_details and cat_details[tk]]
            if available_tool_keys:
                chosen_tool_key = np.random.choice(available_tool_keys)
                specific_tool_or_system = np.random.choice(cat_details[chosen_tool_key])
        
        title = f"{task.capitalize()} Required - {category}"
        
        description_template_key = category if category in job_templates else np.random.choice(list(job_templates.keys())) # Fallback
        description_template = job_templates.get(description_template_key, "General help needed for {task} in {location_type}. Involves {keywords}.")
        
        # Prepare template fillers safely
        template_fillers = {
            "task": task, 
            "location_type": location_type, 
            "keywords": keywords,
            "specific_tool_or_system": specific_tool_or_system,
            "brand_or_type": specific_tool_or_system, # Generic placeholder name in template
            "device_type": specific_tool_or_system,   # Generic placeholder name in template
            "specific_feature": specific_tool_or_system, # Generic placeholder name in template
            "specific_component": specific_tool_or_system, # Generic placeholder name in template
            "subject_area": task if category == "Tutoring" else category, 
            "student_level": specific_tool_or_system if category == "Tutoring" else "any level",
            "outcome": f"successful {task.lower()}",
            "pickup_location_type": location_type if category == "Delivery" else "a local point",
            "vehicle_type_if_any": specific_tool_or_system if category == "Delivery" else "any suitable transport"
        }
        
        # Only include keys that are actually in the chosen template string
        valid_fillers = {k: v for k, v in template_fillers.items() if "{"+k+"}" in description_template}
        
        try:
            description = description_template.format(**valid_fillers)
        except KeyError as e:
            logger.error(f"KeyError formatting description for category {category} with template key {description_template_key}: {e}. Fillers: {valid_fillers}. Template: {description_template}")
            description = f"General help needed for {task} in {location_type}. Involves {keywords} and {specific_tool_or_system}."


        description += f" Reference ID for tracking: JIB{i:04d}{np.random.choice(['A','B','C'])}-{np.random.randint(10,99)}. Urgency: {np.random.choice(['High - Immediate Start Preferred', 'Medium - Flexible Start', 'Low - Planning Ahead'])}. Please outline your relevant experience when applying."

        job_data.append({
            'job_id': job_id_str,
            'title': title,
            'description': description,
            'category': category,
            'location': np.random.choice(['Algiers Central', 'Oran Portside', 'Constantine University Area', 'Annaba Corniche', 'Blida Industrial Zone', 'Outskirts Location'])
        })
    
    jobs = pd.DataFrame(job_data)
    
    # Create sample interactions (ensure this part is robust)
    interaction_data = []
    user_preferences = {}
    for user_id_val in users['user_id']:
        num_preferred_cats = np.random.randint(1, min(4, len(job_categories) + 1)) # At least 1 preferred category
        user_preferences[user_id_val] = np.random.choice(job_categories, size=num_preferred_cats, replace=False)
    
    if jobs.empty:
        logger.warning("No jobs were generated. Cannot create interactions.")
        return users, jobs, pd.DataFrame(columns=['user_id', 'job_id', 'rating', 'comment'])

    for _ in range(n_interactions):
        user_id_interaction = np.random.choice(users['user_id'])
        
        use_preferred_category = np.random.random() < 0.8
        
        if use_preferred_category and len(user_preferences[user_id_interaction]) > 0:
            category_interaction = np.random.choice(user_preferences[user_id_interaction])
            rating = np.random.uniform(4.0, 5.0)
            sentiment = "positive"
        else:
            non_preferred_cats = [c for c in job_categories if c not in user_preferences[user_id_interaction]]
            if not non_preferred_cats: # Fallback if all categories are preferred (unlikely with 1-3 preferred)
                category_interaction = np.random.choice(job_categories)
            else:
                category_interaction = np.random.choice(non_preferred_cats)
            rating = np.random.uniform(1.0, 3.0)
            sentiment = "negative" if rating < 2.0 else "neutral"
        
        # Select job from the chosen category
        category_jobs_df = jobs[jobs['category'] == category_interaction]
        if category_jobs_df.empty:
            # logger.debug(f"No jobs found for category '{category_interaction}' during interaction generation. Skipping interaction.")
            continue # Skip if no jobs for this category (e.g. if n_jobs is small)
            
        job_id_interaction = np.random.choice(category_jobs_df['job_id'].values)
        
        if sentiment == "positive":
            comments = ["Great service, very professional!", "Excellent work, completed on time.", "Very satisfied with the quality of work.", "Highly recommended, will hire again!", "Perfect job, exceeded my expectations."]
        elif sentiment == "neutral":
            comments = ["The job was done as requested.", "Acceptable service, but could be better.", "Completed the task adequately.", "Reasonable quality for the price paid.", "The work was satisfactory overall."]
        else: # negative
            comments = ["Poor service, not recommended at all.", "Did not finish the job properly as discussed.", "Disappointed with the quality and professionalism.", "Would not hire this provider again.", "Work was significantly below expectations."]
        comment = np.random.choice(comments)
        
        interaction_data.append({
            'user_id': user_id_interaction,
            'job_id': job_id_interaction,
            'rating': round(rating, 2),
            'comment': comment
        })
    
    interactions = pd.DataFrame(interaction_data)
    if interactions.empty and n_interactions > 0:
        logger.warning("No interactions generated despite requesting them. Check job/category availability and interaction logic.")
        return users, jobs, pd.DataFrame(columns=['user_id', 'job_id', 'rating', 'comment'])
    elif not interactions.empty:
        # Remove duplicate interactions if any (user interacting with same job multiple times in this synthetic data)
        interactions.drop_duplicates(subset=['user_id', 'job_id'], keep='last', inplace=True)


    return users, jobs, interactions


def main():
    # 1. Generate sample data (uses your enriched version)
    users_df, jobs_df, interactions_df_original_ratings = generate_sample_data()
    logger.info(f"Generated {len(users_df)} users, {len(jobs_df)} jobs, and {len(interactions_df_original_ratings)} interactions")

    if interactions_df_original_ratings.empty:
        logger.error("No interactions were generated by generate_sample_data. Exiting demo.")
        return

    # 2. Define "true positives" for ranking evaluation BEFORE normalization
    ORIGINAL_HIGH_RATING_THRESHOLD = 4.0 
    interactions_df_original_ratings['is_relevant_for_ranking_eval'] = \
        interactions_df_original_ratings['rating'] >= ORIGINAL_HIGH_RATING_THRESHOLD
    
    num_true_positives_total = interactions_df_original_ratings['is_relevant_for_ranking_eval'].sum()
    logger.info(f"Marked {num_true_positives_total} interactions as true positives for ranking eval (original rating >= {ORIGINAL_HIGH_RATING_THRESHOLD}).")

    # 3. Create a DataFrame for model input: normalize 'rating', keep 'is_relevant_for_ranking_eval'
    interactions_df_for_model = interactions_df_original_ratings.copy()
    interactions_df_for_model = normalize_ratings(interactions_df_for_model, rating_col='rating')
    logger.info("Normalized 'rating' column for model input.")
    # interactions_df_for_model now has 'rating' (0-1) and 'is_relevant_for_ranking_eval' (boolean)

    # 4. Initialize the recommendation system
    logger.info("Initializing recommendation system...")
    recommender = JobRecommender(
        embedding_model_name="distilbert-base-uncased",
        sentiment_model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        device="cuda" if torch.cuda.is_available() else "cpu",
        user_feature_dim=64 
    )
    
    # 5. Process job descriptions (this modifies jobs_df by adding cleaned columns)
    logger.info("Processing job descriptions...")
    jobs_df = process_job_descriptions( 
        jobs_df,
        text_columns=['title', 'description'],
        remove_stopwords=True 
    )
    
    # 6. Split data into train and test sets
    from sklearn.model_selection import train_test_split
    user_col_name = 'user_id' 

    can_stratify = False
    if user_col_name in interactions_df_for_model.columns:
        user_counts = interactions_df_for_model[user_col_name].value_counts()
        if user_counts.min() >= 2 and len(user_counts) > 1 : 
             can_stratify = True
        else:
             logger.warning(f"Cannot stratify by user for train/test split: some users may have < 2 interactions or only one unique user. Min user interactions: {user_counts.min()}, Unique users: {len(user_counts)}")

    train_interactions, test_interactions = train_test_split(
        interactions_df_for_model, # Use the df with normalized 'rating' AND 'is_relevant_for_ranking_eval'
        test_size=0.2, 
        random_state=42,
        stratify=interactions_df_for_model[user_col_name] if can_stratify else None
    )
    
    logger.info(f"Training set: {len(train_interactions)} interactions")
    num_true_positives_in_test = test_interactions['is_relevant_for_ranking_eval'].sum()
    logger.info(f"Test set: {len(test_interactions)} interactions ({num_true_positives_in_test} marked as 'is_relevant_for_ranking_eval'==True).")
    
    # 7. Train the recommendation model
    logger.info("Training recommendation model...")
    recommender.train(
        interactions_df=train_interactions, # This contains normalized 'rating' and the boolean flag
        jobs_df=jobs_df,                    
        user_col='user_id',
        job_col='job_id',
        rating_col='rating',                # Model trains on normalized 'rating'
        comment_col='comment',
        epochs=100,
        val_ratio=0.2,                      
        early_stop_patience=40,
        batch_size=32,
        learning_rate=0.001,
        gcn_embedding_dim=64,              
        hidden_dim_gcn=32,
        num_layers_gcn=2,
        dropout_gcn=0.3
    )
    
    # 8. Evaluate the model
    logger.info("Evaluating recommendation model...")
    # test_interactions has normalized 'rating' AND 'is_relevant_for_ranking_eval'
    metrics = recommender.evaluate(
        test_interactions=test_interactions, 
        jobs_df=jobs_df, 
        user_col='user_id',
        job_col='job_id',
        rating_col='rating',        # For MAE/RMSE, uses normalized 'rating'
        comment_col='comment',      
        top_k=10
        # rating_threshold_for_relevance is now handled internally by JobRecommender.evaluate 
        # using the 'is_relevant_for_ranking_eval' column.
    )
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    # 9. Plot recommendation quality
    logger.info("Plotting recommendation quality...")
    if metrics.get('actual_ratings_array_for_plot') is not None and metrics.get('predicted_ratings_array_for_plot') is not None:
        if len(metrics['actual_ratings_array_for_plot']) > 0 and len(metrics['predicted_ratings_array_for_plot']) > 0 :
            plot_recommendation_quality(
                actual_ratings=metrics['actual_ratings_array_for_plot'],
                predicted_ratings=metrics['predicted_ratings_array_for_plot'],
                title="JibJob Recommendation Quality (Normalized 0-1 Scale)"
            )
            plt.savefig("recommendation_quality.png")
            logger.info("Saved recommendation quality plot.")
        else:
            logger.info("Skipping plot: no actual/predicted ratings for MAE/RMSE returned from evaluate.")
    else:
        logger.info("Skipping plot: MAE/RMSE data not available in metrics.")

    # 10. Generate recommendations for a few sample users for display
    logger.info("Generating recommendations for sample users...")
    known_users_in_model = list(recommender.user_to_idx.keys())
    if not known_users_in_model:
        logger.warning("No users known to the model. Skipping display recommendations.")
    else:
        # Sample from users that are actually known to the model after training/mapping
        display_sample_users = np.random.choice(known_users_in_model, min(5, len(known_users_in_model)), replace=False)
        
        for user_id in display_sample_users:
            logger.info(f"Display recommendations for {user_id}:")
            
            # Get original interaction history for DISPLAY purposes
            user_interactions_for_display = interactions_df_original_ratings[
                interactions_df_original_ratings['user_id'] == user_id
            ]
            
            # Get list of all job_ids this user has interacted with from data used for MODEL MAPPINGS
            # This ensures consistency for the recommender's exclude_rated logic.
            rated_job_ids_to_exclude = interactions_df_for_model[ # Use the df that mappings were based on
                interactions_df_for_model['user_id'] == user_id
            ]['job_id'].unique().tolist()
                
            recommendations = recommender.recommend(
                user_id=user_id,
                top_k=5,
                exclude_rated=True,
                rated_job_ids_for_user=rated_job_ids_to_exclude 
            )
            
            print(f"\nUser {user_id} has rated {len(user_interactions_for_display)} jobs (showing original ratings and relevance flag):")
            if not user_interactions_for_display.empty:
                for _, row in user_interactions_for_display.iterrows():
                    try:
                        job_info = jobs_df[jobs_df['job_id'] == row['job_id']].iloc[0]
                        print(f"  - {job_info['title']} ({job_info['category']}): Original Rating = {row['rating']:.1f}/5.0 (RelevantForEval: {row.get('is_relevant_for_ranking_eval', 'N/A')})")
                    except IndexError:
                        logger.warning(f"Could not find job_id {row['job_id']} in jobs_df for display.")
            else:
                print(f"  User {user_id} had no interactions in the original generated dataset to display.")

            print(f"\nTop 5 GCN-based recommendations for {user_id}:")
            if recommendations:
                for rec in recommendations:
                    job_id_rec = rec['job_id']
                    score_rec = rec['score']
                    try:
                        job_info_rec = jobs_df[jobs_df['job_id'] == job_id_rec].iloc[0]
                        print(f"  - {job_info_rec['title']} ({job_info_rec['category']}): Predicted Score = {score_rec:.4f}")
                    except IndexError:
                        logger.warning(f"Could not find recommended job_id {job_id_rec} in jobs_df for display.")
            else:
                print("  No recommendations generated for this user.")
            print()
    
    # 11. Find similar jobs using BERT directly
    logger.info("Finding BERT-based similar jobs (content similarity)...")
    if not jobs_df.empty:
        sample_job_id_for_similarity = np.random.choice(jobs_df['job_id'].values)
        try:
            # Fetch from the current jobs_df, which should have cleaned_title/desc if process_job_descriptions ran
            sample_job_info = jobs_df[jobs_df['job_id'] == sample_job_id_for_similarity].iloc[0]
            logger.info(f"Target for BERT similarity: JOB_ID '{sample_job_id_for_similarity}': {sample_job_info['title']} ({sample_job_info['category']})")
            
            similar_jobs_bert_test = recommender.find_similar_jobs(
                job_id=sample_job_id_for_similarity, 
                jobs_df_for_similarity=jobs_df, # Pass the current (possibly processed) jobs_df
                top_k=5
            )
            
            print(f"\n[TEST] BERT-Based Similar jobs to '{sample_job_info['title']}' ({sample_job_info['category']}):")
            if similar_jobs_bert_test:
                for j_id, similarity in similar_jobs_bert_test:
                    try:
                        sim_job_info = jobs_df[jobs_df['job_id'] == j_id].iloc[0]
                        print(f"  - Job '{sim_job_info['title']}' ({sim_job_info['category']}): BERT Similarity = {similarity:.4f}")
                    except IndexError:
                         logger.warning(f"Could not find similar job_id {j_id} in jobs_df for display.")
            else:
                print("  No BERT-similar jobs found.")
        except Exception as e:
            logger.error(f"Error in BERT-based similar jobs test: {e}", exc_info=True)
    else:
        logger.warning("jobs_df is empty, skipping find_similar_jobs test.")

    # 12. Analyze sentiment
    logger.info("Analyzing sentiment in sample comments...")
    sample_comments = [
        "The service was excellent and completed on time!",
        "The job was done but not to my satisfaction.",
        "Terrible service, would not recommend.",
        "Very professional and skilled worker."
    ]
    print("\nSentiment Analysis:")
    if hasattr(recommender, 'sentiment_model') and recommender.sentiment_model is not None:
        for comment in sample_comments:
            sentiment = recommender.sentiment_model.analyze_sentiment(comment)
            print(f"  - '{comment}': {sentiment:.4f}")
    else:
        logger.warning("Sentiment model not available in recommender. Skipping sentiment analysis display.")
    
    # 13. Save the trained model
    logger.info("Saving the trained model...")
    model_save_path = "jibjob_recommender_model" # Define path
    recommender.save_model(model_save_path) 
    logger.info(f"Model saved to {model_save_path}")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    # It's good practice not to import torch at the global level if only main uses it
    # but since other parts of your project (src.models) use it, it's fine.
    main()
