"""
Sample data generator for testing the JibJob recommendation system.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jibjob_recommender_system.utils.logging_config import setup_logging

def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate sample data for JibJob recommendation system")
        
    parser.add_argument('--output-dir', type=str, default='../sample_data',
                      help='Directory to save generated data')
    parser.add_argument('--num-users', type=int, default=1000,
                      help='Number of users to generate')
    parser.add_argument('--num-jobs', type=int, default=500,
                      help='Number of job postings to generate')
    parser.add_argument('--num-categories', type=int, default=20,
                      help='Number of job categories to generate')
    parser.add_argument('--num-locations', type=int, default=30,
                      help='Number of locations to generate')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
                      
    return parser

def generate_categories(num_categories: int) -> pd.DataFrame:
    """
    Generate sample job categories.
    
    Args:
        num_categories: Number of categories to generate.
        
    Returns:
        pd.DataFrame: DataFrame with category data.
    """
    logging.info(f"Generating {num_categories} job categories")
    
    # List of sample job categories
    sample_categories = [
        ("Software Development", "Development of software applications, websites, and systems"),
        ("Data Science", "Analysis of data, machine learning, and statistical modeling"),
        ("Web Development", "Development of websites, web applications, and web services"),
        ("Mobile Development", "Development of mobile applications for iOS, Android, and other platforms"),
        ("DevOps", "Development operations, CI/CD, and infrastructure management"),
        ("UI/UX Design", "User interface and user experience design"),
        ("Cybersecurity", "Information security, network security, and vulnerability management"),
        ("Network Engineering", "Network design, implementation, and management"),
        ("Cloud Computing", "Cloud architecture, migration, and management"),
        ("IT Support", "Technical support, troubleshooting, and help desk"),
        ("Project Management", "Project planning, coordination, and execution"),
        ("Product Management", "Product strategy, roadmap, and feature definition"),
        ("Marketing", "Marketing strategy, campaigns, and analytics"),
        ("Sales", "Sales strategy, business development, and account management"),
        ("Customer Service", "Customer support, service, and satisfaction"),
        ("Human Resources", "Recruitment, employee relations, and talent management"),
        ("Finance", "Financial analysis, accounting, and reporting"),
        ("Legal", "Legal advice, compliance, and risk management"),
        ("Healthcare", "Medical, nursing, and healthcare services"),
        ("Education", "Teaching, training, and educational services"),
        ("Engineering", "Engineering design, analysis, and implementation"),
        ("Manufacturing", "Production, quality control, and supply chain management"),
        ("Logistics", "Transportation, warehousing, and distribution"),
        ("Retail", "Retail management, merchandising, and operations"),
        ("Hospitality", "Hotel, restaurant, and tourism services")
    ]
    
    # Ensure we have enough sample categories
    if num_categories > len(sample_categories):
        for i in range(len(sample_categories), num_categories):
            sample_categories.append((f"Category {i+1}", f"Description for Category {i+1}"))
    
    # Take a subset of categories
    categories = sample_categories[:num_categories]
    
    # Create DataFrame
    df = pd.DataFrame({
        'category_id': [f"cat_{i:03d}" for i in range(num_categories)],
        'name': [cat[0] for cat in categories],
        'description': [cat[1] for cat in categories]
    })
    
    return df

def generate_locations(num_locations: int) -> pd.DataFrame:
    """
    Generate sample locations.
    
    Args:
        num_locations: Number of locations to generate.
        
    Returns:
        pd.DataFrame: DataFrame with location data.
    """
    logging.info(f"Generating {num_locations} locations")
    
    # List of sample cities with approximate coordinates
    sample_locations = [
        ("New York, NY", 40.7128, -74.0060),
        ("Los Angeles, CA", 34.0522, -118.2437),
        ("Chicago, IL", 41.8781, -87.6298),
        ("Houston, TX", 29.7604, -95.3698),
        ("Phoenix, AZ", 33.4484, -112.0740),
        ("Philadelphia, PA", 39.9526, -75.1652),
        ("San Antonio, TX", 29.4241, -98.4936),
        ("San Diego, CA", 32.7157, -117.1611),
        ("Dallas, TX", 32.7767, -96.7970),
        ("San Jose, CA", 37.3382, -121.8863),
        ("Austin, TX", 30.2672, -97.7431),
        ("Jacksonville, FL", 30.3322, -81.6557),
        ("Fort Worth, TX", 32.7555, -97.3308),
        ("Columbus, OH", 39.9612, -82.9988),
        ("San Francisco, CA", 37.7749, -122.4194),
        ("Charlotte, NC", 35.2271, -80.8431),
        ("Indianapolis, IN", 39.7684, -86.1581),
        ("Seattle, WA", 47.6062, -122.3321),
        ("Denver, CO", 39.7392, -104.9903),
        ("Washington, DC", 38.9072, -77.0369),
        ("Boston, MA", 42.3601, -71.0589),
        ("Nashville, TN", 36.1627, -86.7816),
        ("Baltimore, MD", 39.2904, -76.6122),
        ("Louisville, KY", 38.2527, -85.7585),
        ("Portland, OR", 45.5051, -122.6750),
        ("Las Vegas, NV", 36.1699, -115.1398),
        ("Milwaukee, WI", 43.0389, -87.9065),
        ("Albuquerque, NM", 35.0844, -106.6504),
        ("Tucson, AZ", 32.2226, -110.9747),
        ("Fresno, CA", 36.7378, -119.7871)
    ]
    
    # Ensure we have enough sample locations
    if num_locations > len(sample_locations):
        # Generate random locations for any additional ones
        for i in range(len(sample_locations), num_locations):
            # Random coordinates in continental US
            lat = random.uniform(24.396308, 49.384358)
            lon = random.uniform(-125.000000, -66.934570)
            sample_locations.append((f"Location {i+1}", lat, lon))
    
    # Take a subset of locations
    locations = sample_locations[:num_locations]
    
    # Create DataFrame
    df = pd.DataFrame({
        'location_id': [f"loc_{i:03d}" for i in range(num_locations)],
        'name': [loc[0] for loc in locations],
        'latitude': [loc[1] for loc in locations],
        'longitude': [loc[2] for loc in locations]
    })
    
    return df

def generate_users(
    num_users: int,
    locations_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate sample users.
    
    Args:
        num_users: Number of users to generate.
        locations_df: DataFrame with location data.
        
    Returns:
        pd.DataFrame: DataFrame with user data.
    """
    logging.info(f"Generating {num_users} users")
    
    # List of sample first and last names
    first_names = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen"
    ]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
        "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson"
    ]
    
    # List of sample profile bios
    bio_templates = [
        "Experienced {role} with {years} years of expertise in {skill1}, {skill2}, and {skill3}.",
        "Passionate {role} seeking opportunities to use my skills in {skill1} and {skill2}.",
        "{years}-year professional with extensive knowledge of {skill1} and {skill2}.",
        "Certified {role} with a focus on {skill1}. {years} years of experience in {skill2} and {skill3}.",
        "Results-driven {role} with proven success in {skill1} and {skill2}.",
        "Detail-oriented {role} with {years}+ years of experience in {skill3}.",
        "Creative {role} with expertise in {skill1} and {skill2}.",
        "Innovative {role} passionate about {skill1} and {skill3}.",
        "Skilled {role} with a background in {skill1}, {skill2}, and {skill3}.",
        "Accomplished {role} with {years} years of experience in {skill2} and {skill3}."
    ]
    
    roles = [
        "Software Developer", "Web Developer", "Data Scientist", "Project Manager", "UI/UX Designer",
        "Network Engineer", "DevOps Engineer", "Business Analyst", "Product Manager", "Marketing Specialist",
        "Sales Representative", "Financial Analyst", "HR Specialist", "Customer Service Representative",
        "Content Writer", "Graphic Designer", "Teacher", "Nurse", "Accountant", "Lawyer"
    ]
    
    skills = [
        "Java", "Python", "JavaScript", "SQL", "React", "Angular", "Node.js", "AWS", "Docker", "Kubernetes",
        "Machine Learning", "Data Analysis", "Project Management", "Agile Methodologies", "UI Design",
        "UX Research", "Digital Marketing", "Content Strategy", "SEO", "Social Media Marketing",
        "Sales", "Customer Relationship Management", "Financial Analysis", "Budget Planning",
        "Recruitment", "Employee Relations", "Customer Service", "Problem Solving", "Communication",
        "Leadership", "Team Management", "Strategic Planning", "Research", "Writing", "Editing",
        "Graphic Design", "Adobe Creative Suite", "Teaching", "Patient Care", "Accounting"
    ]
    
    # Generate users
    data = []
    for i in range(num_users):
        user_type = "professional" if i < int(num_users * 0.8) else "employer"  # 80% professionals, 20% employers
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        if user_type == "professional":
            role = random.choice(roles)
            years = random.randint(1, 20)
            skill1 = random.choice(skills)
            skill2 = random.choice([s for s in skills if s != skill1])
            skill3 = random.choice([s for s in skills if s not in [skill1, skill2]])
            
            bio = random.choice(bio_templates).format(
                role=role, years=years, skill1=skill1, skill2=skill2, skill3=skill3
            )
        else:
            company_names = ["Tech Solutions", "Global Innovators", "Creative Designs", "Modern Software", 
                          "Data Systems", "Digital Agency", "Web Experts", "Smart Tech", "Future Solutions"]
            company = random.choice(company_names) + " " + random.choice(["Inc.", "LLC", "Ltd.", "Group"])
            bio = f"Hiring manager at {company}. Looking for talented professionals to join our team."
        
        # Random location
        location_id = random.choice(locations_df['location_id'])
        location_row = locations_df[locations_df['location_id'] == location_id].iloc[0]
        
        # Add small random variation to coordinates (within about 10km)
        lat_variation = random.uniform(-0.09, 0.09)
        lon_variation = random.uniform(-0.09, 0.09)
        
        user_data = {
            'user_id': f"user_{i:04d}",
            'first_name': first_name,
            'last_name': last_name,
            'user_type': user_type,
            'profile_bio': bio,
            'location_id': location_id,
            'latitude': location_row['latitude'] + lat_variation,
            'longitude': location_row['longitude'] + lon_variation
        }
        
        data.append(user_data)
    
    return pd.DataFrame(data)

def generate_jobs(
    num_jobs: int,
    categories_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    users_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate sample job postings.
    
    Args:
        num_jobs: Number of job postings to generate.
        categories_df: DataFrame with category data.
        locations_df: DataFrame with location data.
        users_df: DataFrame with user data.
        
    Returns:
        pd.DataFrame: DataFrame with job posting data.
    """
    logging.info(f"Generating {num_jobs} job postings")
    
    # Get employer user IDs
    employer_ids = users_df[users_df['user_type'] == 'employer']['user_id'].tolist()
    
    # Create job title templates based on categories
    job_title_templates = {
        "Software Development": ["Software Developer", "Software Engineer", "Full Stack Developer", "Backend Developer", "Frontend Developer"],
        "Data Science": ["Data Scientist", "Data Analyst", "Machine Learning Engineer", "AI Specialist", "Business Intelligence Analyst"],
        "Web Development": ["Web Developer", "Web Designer", "UI Developer", "JavaScript Developer", "WordPress Developer"],
        "Mobile Development": ["Mobile App Developer", "iOS Developer", "Android Developer", "React Native Developer", "Flutter Developer"],
        "DevOps": ["DevOps Engineer", "Site Reliability Engineer", "Cloud Engineer", "Infrastructure Specialist", "CI/CD Specialist"],
        "UI/UX Design": ["UI Designer", "UX Designer", "Product Designer", "Interaction Designer", "Visual Designer"],
        "Cybersecurity": ["Security Engineer", "Security Analyst", "Penetration Tester", "Security Consultant", "Compliance Specialist"],
        "Project Management": ["Project Manager", "Scrum Master", "Agile Coach", "Project Coordinator", "Program Manager"],
        "Marketing": ["Marketing Specialist", "Digital Marketer", "Content Marketer", "SEO Specialist", "Social Media Manager"],
        "Sales": ["Sales Representative", "Account Manager", "Business Development Manager", "Sales Consultant", "Sales Engineer"]
    }
    
    # For other categories not in the templates
    generic_titles = [
        "{} Specialist", "{} Expert", "{} Consultant", "{} Professional", "{} Analyst",
        "Senior {}", "Junior {}", "{} Manager", "{} Coordinator", "{} Associate"
    ]
    
    # Job description templates
    job_description_templates = [
        "We are looking for a {title} to join our team. The ideal candidate will have experience in {skill1}, {skill2}, and {skill3}. "
        "Responsibilities include {responsibility1} and {responsibility2}. "
        "Requirements: {requirement1}, {requirement2}, and {requirement3}. "
        "{experience_level} experience preferred.",
        
        "Exciting opportunity for a {title} with skills in {skill1} and {skill2}. "
        "You will be responsible for {responsibility1}, {responsibility2}, and {responsibility3}. "
        "We're looking for someone with {experience_level} experience in {skill3}. "
        "Requirements: {requirement1}, {requirement2}.",
        
        "Join our growing team as a {title}. "
        "In this role, you will {responsibility1} and {responsibility2}. "
        "The ideal candidate has {experience_level} experience with {skill1}, {skill2}, and {skill3}. "
        "Requirements include {requirement1} and {requirement2}.",
        
        "{title} needed for a {employment_type} position. "
        "Main duties: {responsibility1}, {responsibility2}. "
        "Required skills: {skill1}, {skill2}, and {skill3}. "
        "{experience_level} experience in {requirement1} is essential."
    ]
    
    # Skills, responsibilities, and requirements based on category
    skills_by_category = {
        "Software Development": ["Java", "Python", "C#", "JavaScript", "SQL", "React", "Angular", "Node.js", "Spring Boot", "ASP.NET"],
        "Data Science": ["Python", "R", "SQL", "Machine Learning", "Statistical Analysis", "Data Visualization", "TensorFlow", "PyTorch", "Tableau", "Power BI"],
        "Web Development": ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "PHP", "Ruby on Rails", "Node.js", "WordPress"],
        "Mobile Development": ["Swift", "Kotlin", "Java", "React Native", "Flutter", "Mobile UI Design", "iOS", "Android", "Firebase", "App Store Optimization"],
        "DevOps": ["Docker", "Kubernetes", "Jenkins", "AWS", "Azure", "GCP", "Terraform", "Ansible", "CI/CD", "Linux"],
        "UI/UX Design": ["Figma", "Sketch", "Adobe XD", "User Research", "Wireframing", "Prototyping", "User Testing", "Interaction Design", "Visual Design", "Information Architecture"],
        "Cybersecurity": ["Network Security", "Penetration Testing", "Vulnerability Assessment", "SIEM", "Firewall Configuration", "Security Auditing", "Encryption", "Authentication Systems", "Compliance", "Threat Analysis"],
        "Project Management": ["Agile", "Scrum", "Kanban", "Project Planning", "Risk Management", "Stakeholder Management", "Budgeting", "Resource Allocation", "JIRA", "Microsoft Project"],
        "Marketing": ["SEO", "SEM", "Content Marketing", "Social Media", "Email Marketing", "Analytics", "CRM", "Marketing Automation", "Adobe Analytics", "Google Analytics"],
        "Sales": ["CRM Software", "Sales Strategy", "Account Management", "Lead Generation", "Negotiation", "Customer Relationships", "Sales Presentations", "Market Research", "Salesforce", "HubSpot"]
    }
    
    generic_skills = ["Communication", "Team Collaboration", "Problem Solving", "Analytical Thinking", "Attention to Detail", 
                    "Time Management", "Organization", "Critical Thinking", "Creativity", "Adaptability"]
    
    responsibilities_by_category = {
        "Software Development": [
            "developing software solutions", "writing clean and maintainable code", 
            "collaborating with cross-functional teams", "debugging and troubleshooting issues",
            "participating in code reviews", "implementing best practices", 
            "maintaining existing codebases", "optimizing application performance"
        ],
        "Data Science": [
            "analyzing large datasets", "developing machine learning models", 
            "creating data visualizations", "interpreting trends and patterns",
            "developing predictive models", "performing statistical analysis", 
            "communicating insights to stakeholders", "optimizing data collection procedures"
        ]
    }
    
    generic_responsibilities = [
        "collaborating with team members", "contributing to project goals", 
        "ensuring quality deliverables", "meeting deadlines",
        "communicating with stakeholders", "documenting processes", 
        "participating in meetings and planning sessions", "staying updated with industry trends"
    ]
    
    requirements_by_category = {
        "Software Development": [
            "Bachelor's degree in Computer Science or related field", "proficiency in programming languages",
            "knowledge of software development methodologies", "experience with version control systems",
            "understanding of data structures and algorithms", "experience with database technologies"
        ],
        "Data Science": [
            "Bachelor's or Master's degree in Data Science, Statistics, or related field", "proficiency in data analysis tools",
            "experience with machine learning algorithms", "strong statistical background",
            "data visualization skills", "experience with big data technologies"
        ]
    }
    
    generic_requirements = [
        "strong communication skills", "ability to work in a team environment",
        "problem-solving abilities", "attention to detail",
        "self-motivation", "ability to work under pressure",
        "adaptability to changing requirements", "continuous learning mindset"
    ]
    
    experience_levels = ["Entry-level", "1-3 years", "3-5 years", "5+ years", "Senior-level"]
    employment_types = ["Full-time", "Part-time", "Contract", "Remote", "Hybrid"]
    
    # Generate jobs
    base_date = datetime.now() - timedelta(days=90)
    data = []
    
    for i in range(num_jobs):
        # Select a random category and employer
        category_id = random.choice(categories_df['category_id'])
        category_row = categories_df[categories_df['category_id'] == category_id].iloc[0]
        category_name = category_row['name']
        
        employer_id = random.choice(employer_ids)
        
        # Select a random location
        location_id = random.choice(locations_df['location_id'])
        location_row = locations_df[locations_df['location_id'] == location_id].iloc[0]
        
        # Add small random variation to coordinates (within about 10km)
        lat_variation = random.uniform(-0.09, 0.09)
        lon_variation = random.uniform(-0.09, 0.09)
        
        # Generate job title
        if category_name in job_title_templates:
            title = random.choice(job_title_templates[category_name])
        else:
            title_template = random.choice(generic_titles)
            title = title_template.format(category_name)
        
        # Get skills, responsibilities, and requirements for this category
        skills = skills_by_category.get(category_name, []) + generic_skills
        responsibilities = responsibilities_by_category.get(category_name, []) + generic_responsibilities
        requirements = requirements_by_category.get(category_name, []) + generic_requirements
        
        # Select random skills, responsibilities, and requirements
        skill1 = random.choice(skills)
        skill2 = random.choice([s for s in skills if s != skill1])
        skill3 = random.choice([s for s in skills if s not in [skill1, skill2]])
        
        responsibility1 = random.choice(responsibilities)
        responsibility2 = random.choice([r for r in responsibilities if r != responsibility1])
        responsibility3 = random.choice([r for r in responsibilities if r not in [responsibility1, responsibility2]])
        
        requirement1 = random.choice(requirements)
        requirement2 = random.choice([r for r in requirements if r != requirement1])
        requirement3 = random.choice([r for r in requirements if r not in [requirement1, requirement2]])
        
        # Other job details
        experience_level = random.choice(experience_levels)
        employment_type = random.choice(employment_types)
        
        # Generate job description
        description_template = random.choice(job_description_templates)
        description = description_template.format(
            title=title,
            skill1=skill1, skill2=skill2, skill3=skill3,
            responsibility1=responsibility1, responsibility2=responsibility2, responsibility3=responsibility3,
            requirement1=requirement1, requirement2=requirement2, requirement3=requirement3,
            experience_level=experience_level, employment_type=employment_type
        )
        
        # Random date in the last 90 days
        days_ago = random.randint(0, 90)
        created_at = (base_date + timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        job_data = {
            'job_id': f"job_{i:04d}",
            'title': title,
            'description': description,
            'required_category_id': category_id,
            'created_at': created_at,
            'employer_id': employer_id,
            'location_id': location_id,
            'latitude': location_row['latitude'] + lat_variation,
            'longitude': location_row['longitude'] + lon_variation,
            'experience_level': experience_level,
            'employment_type': employment_type
        }
        
        data.append(job_data)
    
    return pd.DataFrame(data)

def generate_professional_categories(
    users_df: pd.DataFrame,
    categories_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate professional category mappings.
    
    Args:
        users_df: DataFrame with user data.
        categories_df: DataFrame with category data.
        
    Returns:
        pd.DataFrame: DataFrame with professional category mappings.
    """
    # Get professional user IDs
    professional_ids = users_df[users_df['user_type'] == 'professional']['user_id'].tolist()
    category_ids = categories_df['category_id'].tolist()
    
    logging.info(f"Generating professional category mappings for {len(professional_ids)} professionals")
    
    # Generate mappings
    data = []
    
    for user_id in professional_ids:
        # Each professional has 1-3 categories
        num_categories = random.randint(1, 3)
        selected_categories = random.sample(category_ids, num_categories)
        
        for category_id in selected_categories:
            data.append({
                'user_id': user_id,
                'category_id': category_id
            })
    
    return pd.DataFrame(data)

def generate_job_applications(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    professional_categories_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate job applications.
    
    Args:
        users_df: DataFrame with user data.
        jobs_df: DataFrame with job data.
        professional_categories_df: DataFrame with professional category mappings.
        
    Returns:
        pd.DataFrame: DataFrame with job application data.
    """
    # Get professional user IDs
    professional_ids = users_df[users_df['user_type'] == 'professional']['user_id'].tolist()
    
    # Get user category preferences
    user_categories = {}
    for user_id in professional_ids:
        categories = professional_categories_df[professional_categories_df['user_id'] == user_id]['category_id'].tolist()
        user_categories[user_id] = categories
    
    logging.info(f"Generating job applications for {len(professional_ids)} professionals")
    
    # Generate applications
    data = []
    application_count = 0
    
    # Base date for applications
    base_date = datetime.now() - timedelta(days=60)
    
    # Each professional applies to 0-10 jobs
    for user_id in professional_ids:
        # 80% of users have at least one application
        if random.random() < 0.8:
            # Number of applications for this user (1-10)
            num_applications = random.randint(1, 10)
            
            # User's preferred categories
            preferred_categories = user_categories.get(user_id, [])
            
            # Filter jobs by preferred categories if available
            if preferred_categories:
                preferred_jobs = jobs_df[jobs_df['required_category_id'].isin(preferred_categories)]
                # If no jobs match the categories, fall back to all jobs
                job_pool = preferred_jobs if len(preferred_jobs) > 0 else jobs_df
            else:
                job_pool = jobs_df
            
            # Get list of job IDs
            job_ids = job_pool['job_id'].tolist()
            
            # Randomly select jobs (without replacement if possible)
            if len(job_ids) >= num_applications:
                selected_jobs = random.sample(job_ids, num_applications)
            else:
                selected_jobs = job_ids.copy()
                
            for job_id in selected_jobs:
                # Random date in the last 60 days
                days_ago = random.randint(0, 60)
                applied_at = (base_date + timedelta(days=days_ago)).strftime('%Y-%m-%d')
                
                # Random status with probabilities
                status_prob = random.random()
                if status_prob < 0.3:
                    status = "pending"
                elif status_prob < 0.6:
                    status = "viewed"
                elif status_prob < 0.8:
                    status = "interviewed"
                elif status_prob < 0.9:
                    status = "offered"
                else:
                    status = "rejected"
                
                data.append({
                    'application_id': f"app_{application_count:04d}",
                    'user_id': user_id,
                    'job_id': job_id,
                    'applied_at': applied_at,
                    'status': status
                })
                
                application_count += 1
    
    logging.info(f"Generated {application_count} job applications")
    return pd.DataFrame(data)

def save_dataframes(
    output_dir: str,
    categories_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    professional_categories_df: pd.DataFrame,
    job_applications_df: pd.DataFrame
) -> None:
    """
    Save DataFrames to CSV files.
    
    Args:
        output_dir: Directory to save files.
        categories_df: DataFrame with category data.
        locations_df: DataFrame with location data.
        users_df: DataFrame with user data.
        jobs_df: DataFrame with job data.
        professional_categories_df: DataFrame with professional category mappings.
        job_applications_df: DataFrame with job application data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrames to CSV
    categories_df.to_csv(os.path.join(output_dir, 'categories.csv'), index=False)
    locations_df.to_csv(os.path.join(output_dir, 'locations.csv'), index=False)
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    jobs_df.to_csv(os.path.join(output_dir, 'jobs.csv'), index=False)
    professional_categories_df.to_csv(os.path.join(output_dir, 'professional_categories.csv'), index=False)
    job_applications_df.to_csv(os.path.join(output_dir, 'job_applications.csv'), index=False)
    
    logging.info(f"All data saved to {output_dir}")

def main() -> None:
    """
    Main function to generate sample data.
    """
    # Set up argument parser
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    logging.info("Starting sample data generation")
    
    # Generate data
    categories_df = generate_categories(args.num_categories)
    locations_df = generate_locations(args.num_locations)
    users_df = generate_users(args.num_users, locations_df)
    jobs_df = generate_jobs(args.num_jobs, categories_df, locations_df, users_df)
    professional_categories_df = generate_professional_categories(users_df, categories_df)
    job_applications_df = generate_job_applications(users_df, jobs_df, professional_categories_df)
    
    # Save data
    save_dataframes(
        args.output_dir,
        categories_df,
        locations_df,
        users_df,
        jobs_df,
        professional_categories_df,
        job_applications_df
    )
    
    logging.info("Sample data generation complete")

if __name__ == "__main__":
    main()
