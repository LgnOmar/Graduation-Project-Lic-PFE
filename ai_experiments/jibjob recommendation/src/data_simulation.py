"""
Module for generating and saving simulated data for the JibJob recommendation system
with realistic Algerian context.
"""
import pandas as pd
import numpy as np
from typing import Tuple
import random

# Algerian wilayas (provinces)
ALGERIAN_WILAYAS = [
    "Adrar", "Chlef", "Laghouat", "Oum El Bouaghi", "Batna", "Béjaïa", "Biskra", 
    "Béchar", "Blida", "Bouira", "Tamanrasset", "Tébessa", "Tlemcen", "Tiaret", 
    "Tizi Ouzou", "Alger", "Djelfa", "Jijel", "Sétif", "Saïda", "Skikda", 
    "Sidi Bel Abbès", "Annaba", "Guelma", "Constantine", "Médéa", "Mostaganem", 
    "M'Sila", "Mascara", "Ouargla", "Oran", "El Bayadh", "Illizi", "Bordj Bou Arréridj", 
    "Boumerdès", "El Tarf", "Tindouf", "Tissemsilt", "El Oued", "Khenchela", "Souk Ahras", 
    "Tipaza", "Mila", "Aïn Defla", "Naâma", "Aïn Témouchent", "Ghardaïa", "Relizane",
    "Timimoun", "Bordj Badji Mokhtar", "Ouled Djellal", "Béni Abbès", "In Salah", 
    "In Guezzam", "Touggourt", "Djanet", "El M'Ghair", "El Meniaa"
]

# Algerian popular job categories and specific jobs
ALGERIAN_JOB_CATEGORIES = {
    "Home Services": [
        "Plumbing repairs", "Electrical installations", "Housekeeping", "Painting services", 
        "Furniture assembly", "Air conditioner maintenance", "Home renovation"
    ],
    "Teaching & Education": [
        "Private math tutoring", "French language lessons", "Arabic language teaching", 
        "University exam preparation", "Science tutoring", "Computer skills training"
    ],
    "Transportation": [
        "Moving assistance", "Furniture transport", "Delivery services", 
        "Airport pickup", "City tours", "Intercity transport"
    ],
    "Digital Services": [
        "Website development", "Graphic design", "Social media management", 
        "Logo design", "E-commerce setup", "Mobile app development", "Video editing"
    ],
    "Handcrafts": [
        "Traditional carpet making", "Pottery services", "Leather crafting", 
        "Jewelry making", "Traditional dress sewing", "Wood crafting"
    ],
    "Maintenance & Repairs": [
        "Smartphone repair", "Computer maintenance", "Home appliance repair", 
        "Car mechanics", "Electronics repair", "Bicycle repair"
    ],
    "Events & Celebrations": [
        "Wedding planning", "Event photography", "Catering services", 
        "DJ services", "Traditional music performance", "Event decoration"
    ],
    "Beauty & Wellness": [
        "Home hairdressing", "Makeup services", "Henna art", 
        "Manicure & pedicure", "Traditional hamam services", "Massage therapy"
    ],
    "Food & Cuisine": [
        "Traditional pastry making", "Homemade couscous", "Catering for special events", 
        "Cooking lessons", "Home chef services", "Traditional bread baking"
    ],
    "Agriculture & Gardening": [
        "Garden maintenance", "Plant care", "Agricultural consulting", 
        "Irrigation system installation", "Fruit picking", "Olive harvesting"
    ]
}

# Algerian user profile templates
ALGERIAN_USER_PROFILES = [
    "Recent graduate from {university} with skills in {skills}. Available for {job_type} jobs in {wilaya}.",
    "Professional with {experience} years of experience in {field}. Specialized in {specialization} and seeking opportunities in {wilaya}.",
    "Artisan specializing in traditional {craft_type}. Creating custom work for clients across {wilaya}.",
    "Freelance {profession} based in {wilaya}, offering services in {services}.",
    "Student at {university} with part-time availability for {job_type} work. Familiar with {skills}.",
    "Self-taught {profession} with portfolio of completed projects in {wilaya} and surrounding areas.",
    "Retired professional offering expertise in {field} based on {experience} years of industry experience.",
    "Multilingual individual fluent in {languages}, providing {services} services in {wilaya}.",
    "Skilled tradesperson specializing in {trade} with availability for projects in {wilaya} region.",
    "Home-based entrepreneur offering {services} with delivery/service in {wilaya} area."
]

# Algerian universities
ALGERIAN_UNIVERSITIES = [
    "University of Algiers", "University of Constantine", "University of Oran", "USTHB",
    "University of Batna", "University of Tlemcen", "University of Annaba", "ENP Alger",
    "ESI Alger", "University of Blida", "University of Béjaïa", "University of Sétif",
    "University of Mostaganem", "University of Tizi Ouzou", "ESC Algier"
]

# Common professions and skills for Algerian gig economy
PROFESSIONS = [
    "programmer", "designer", "translator", "tutor", "photographer", "electrician", 
    "plumber", "driver", "carpenter", "cook", "tailor", "mechanic", "handyman",
    "web developer", "digital marketer", "language teacher", "accountant", "craftsman"
]

SKILLS = [
    "programming", "design", "writing", "teaching", "photography", "electrical repair",
    "plumbing", "driving", "woodworking", "cooking", "sewing", "car repair", "home repair",
    "web development", "digital marketing", "language instruction", "accounting", "crafting",
    "customer service", "project management", "social media", "Microsoft Office", "Arabic",
    "French", "English", "Tamazight", "negotiation", "event planning"
]

LANGUAGES = ["Arabic", "French", "English", "Tamazight", "Spanish", "German", "Italian", "Turkish"]

def generate_realistic_job_description(category, job_type, wilaya):
    """Generate a realistic job description for an Algerian context."""
    descriptions = [
        f"Looking for an experienced professional for {job_type} in {wilaya}. The work involves {random.choice(['immediate assistance', 'regular maintenance', 'one-time service', 'ongoing support'])} with competitive compensation.",
        f"Seeking skilled individual for {job_type} services in {wilaya}. Must have own tools and transportation. {random.choice(['Urgent need', 'Flexible hours', 'Weekend work', 'Weekday availability'])} required.",
        f"{job_type} expert needed in {wilaya}. Project involves {random.choice(['residential work', 'commercial space', 'public venue', 'private property'])}. Good communication skills and punctuality essential.",
        f"Hiring for {job_type} in the {wilaya} area. {random.choice(['Experience required', 'Training provided', 'Portfolio requested', 'References needed'])}. Fair payment and potential for regular work.",
        f"{category} professional wanted for {job_type} in {wilaya}. {random.choice(['Short-term', 'Medium-term', 'Long-term', 'Recurring'])} project with possibility for extension.",
        f"Need assistance with {job_type} in {wilaya}. Ideal for {random.choice(['students', 'part-time workers', 'professionals', 'retirees'])} with knowledge in {category.lower()}.",
        f"Looking for talented {category.lower()} specialist for {job_type} work in {wilaya}. Project starts {random.choice(['immediately', 'next week', 'this month', 'upon agreement'])}."
    ]
    return random.choice(descriptions)

def generate_sample_data(
    n_users: int = 1000,
    n_jobs: int = 500,
    n_interactions: int = 2000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate sample data for users, jobs, and their interactions with realistic Algerian context.
    
    Args:
        n_users: Number of users to generate
        n_jobs: Number of jobs to generate
        n_interactions: Number of user-job interactions to generate
    
    Returns:
        Tuple of (users_df, jobs_df, interactions_df)
    """
    # Generate users with realistic Algerian profiles
    user_ids = [f'user_{i}' for i in range(n_users)]
    user_names = []
    user_descriptions = []
    user_skills = []
    user_preferences = []
    
    for i in range(n_users):
        # Generate user name (more realistic than just "User X")
        common_first_names = ["Mohamed", "Ahmed", "Ali", "Karim", "Youcef", "Omar", "Amine", "Sofiane", 
                             "Fatima", "Amina", "Yasmine", "Meriem", "Amel", "Leila", "Samira", "Nadia"]
        user_names.append(f"{random.choice(common_first_names)} {chr(65 + random.randint(0, 25))}")
        
        # Generate realistic user profile
        profile_template = random.choice(ALGERIAN_USER_PROFILES)
        user_wilaya = random.choice(ALGERIAN_WILAYAS)
        user_university = random.choice(ALGERIAN_UNIVERSITIES)
        user_profession = random.choice(PROFESSIONS)
        user_experience = random.randint(1, 15)
        field = random.choice(list(ALGERIAN_JOB_CATEGORIES.keys()))
        
        # Generate 2-4 random skills
        user_skill_list = random.sample(SKILLS, k=random.randint(2, 4))
        user_skills.append(",".join(user_skill_list))
        
        # Generate 1-3 job category preferences 
        user_pref_list = random.sample(list(ALGERIAN_JOB_CATEGORIES.keys()), k=random.randint(1, 3))
        user_preferences.append(",".join(user_pref_list))
        
        # Fill template with random appropriate values
        user_description = profile_template.format(
            university=user_university,
            skills=", ".join(user_skill_list),
            job_type=random.choice(list(ALGERIAN_JOB_CATEGORIES.keys())).lower(),
            wilaya=user_wilaya,
            experience=user_experience,
            field=field,
            specialization=random.choice(ALGERIAN_JOB_CATEGORIES[field]),
            craft_type=random.choice(ALGERIAN_JOB_CATEGORIES["Handcrafts"]),
            profession=user_profession,
            services=", ".join(random.sample(user_skill_list, k=min(2, len(user_skill_list)))),
            languages=", ".join(random.sample(LANGUAGES, k=random.randint(2, 3))),
            trade=random.choice(PROFESSIONS)
        )
        
        user_descriptions.append(user_description)
    
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'name': user_names,
        'description_profil_utilisateur_anglais': user_descriptions,
        'skills': user_skills,
        'preferences': user_preferences
    })
    
    # Generate jobs with realistic Algerian context
    job_ids = [f'job_{i}' for i in range(n_jobs)]
    job_titles = []
    job_descriptions = []
    job_categories = []
    job_locations = []
    
    for i in range(n_jobs):
        # Select random category and job type
        category = random.choice(list(ALGERIAN_JOB_CATEGORIES.keys()))
        job_type = random.choice(ALGERIAN_JOB_CATEGORIES[category])
        wilaya = random.choice(ALGERIAN_WILAYAS)
        
        job_title = f"{job_type} in {wilaya}"
        job_titles.append(job_title)
        job_categories.append(category)
        job_locations.append(wilaya)
        
        # Generate realistic description
        description = generate_realistic_job_description(category, job_type, wilaya)
        job_descriptions.append(description)
    
    jobs_df = pd.DataFrame({
        'job_id': job_ids,
        'title': job_titles,
        'description_mission_anglais': job_descriptions,
        'categorie_mission': job_categories,
        'location': job_locations
    })
      # Generate interactions with realistic feedback
    def generate_realistic_comment(rating, job_category, job_type, job_location):
        """Generate a realistic comment based on rating and job details."""
        if pd.isna(rating):
            return np.nan
            
        positive_comments = [
            f"Excellent service for {job_type}. Very professional and punctual.",
            f"Great work in {job_location}! Would recommend to others looking for {job_category.lower()} services.",
            f"Very satisfied with the {job_type} service. Fast and efficient.",
            f"The quality of work was outstanding. I'll definitely hire again for {job_type}.",
            f"Professional and courteous. Completed the {job_type} work perfectly.",
            f"Exceptional skills in {job_category.lower()}. The job was completed ahead of schedule.",
            f"Very pleased with the results. Expert in {job_type} with fair pricing.",
            f"Excellent communication and service. Best {job_category.lower()} professional in {job_location}."
        ]
        
        neutral_comments = [
            f"The {job_type} service was adequate. Nothing exceptional but got the job done.",
            f"Average work for {job_category.lower()}. Could improve on timeliness.",
            f"Acceptable service in {job_location}. Some minor issues but overall okay.",
            f"Reasonable quality work. Some aspects of the {job_type} could be better.",
            f"Service was as expected. Neither impressive nor disappointing.",
            f"Completed the job as required. Standard {job_category.lower()} service.",
            f"Decent work but took longer than expected for {job_type} completion.",
            f"Fair price for the quality. Average {job_category.lower()} service provider in {job_location}."
        ]
        
        negative_comments = [
            f"Disappointed with the {job_type} service. Many issues left unresolved.",
            f"Poor quality work in {job_location}. Would not recommend for {job_category.lower()} jobs.",
            f"The service was below expectations. Delays and communication problems.",
            f"Unsatisfactory result for {job_type}. Several problems after completion.",
            f"Not worth the price. Better {job_category.lower()} services available in {job_location}.",
            f"Unprofessional approach to the {job_type} job. Wouldn't hire again.",
            f"Very late and poor quality work. Avoid for {job_category.lower()} needs.",
            f"Didn't complete the {job_type} job properly. Had to find someone else."
        ]
        
        if rating >= 4:
            return random.choice(positive_comments)
        elif 2 <= rating <= 3:
            return random.choice(neutral_comments)
        else:
            return random.choice(negative_comments)
    
    # Generate random interactions
    interactions = []
    for _ in range(n_interactions):
        # Select random user and job
        user_idx = random.randint(0, n_users - 1)
        job_idx = random.randint(0, n_jobs - 1)
        
        user_id = users_df.loc[user_idx, 'user_id']
        job_id = jobs_df.loc[job_idx, 'job_id']
        
        # Determine if this interaction has a rating (70% have ratings)
        if random.random() < 0.7:
            # Bias towards higher ratings (more realistic for services)
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30])
        else:
            rating = np.nan
            
        # Determine if this interaction has a comment (60% of rated interactions have comments)
        if not pd.isna(rating) and random.random() < 0.6:
            job_category = jobs_df.loc[job_idx, 'categorie_mission']
            job_title = jobs_df.loc[job_idx, 'title']
            job_location = jobs_df.loc[job_idx, 'location']
            job_type = job_title.split(" in ")[0] if " in " in job_title else job_title
            
            comment = generate_realistic_comment(rating, job_category, job_type, job_location)
        else:
            comment = np.nan
            
        interactions.append({
            'user_id': user_id,
            'job_id': job_id,
            'rating_explicite': rating,
            'commentaire_texte_anglais': comment
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    return users_df, jobs_df, interactions_df

def save_dataframes(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    output_dir: str = 'data'
) -> None:
    """
    Save the generated DataFrames to CSV files.
    
    Args:
        users_df: DataFrame containing user information
        jobs_df: DataFrame containing job information
        interactions_df: DataFrame containing user-job interactions
        output_dir: Directory to save the CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the original files
    users_df.to_csv(f'{output_dir}/users_df.csv', index=False)
    jobs_df.to_csv(f'{output_dir}/jobs_df.csv', index=False)
    interactions_df.to_csv(f'{output_dir}/interactions_df.csv', index=False)
    
    print(f"Generated {len(users_df)} users with realistic Algerian profiles")
    print(f"Generated {len(jobs_df)} jobs across the 58 Algerian wilayas")
    print(f"Generated {len(interactions_df)} user-job interactions with realistic feedback")
    print(f"Data files saved in {output_dir}/ directory")

if __name__ == '__main__':
    # Generate realistic Algerian data
    print("Generating realistic JibJob data for the Algerian market...")
    users_df, jobs_df, interactions_df = generate_sample_data(n_users=1000, n_jobs=500, n_interactions=3000)
    save_dataframes(users_df, jobs_df, interactions_df)
