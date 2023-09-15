import random
from faker import Faker
import pandas as pd

# Initialize the Faker library
fake = Faker()

# List of Indian cities for location
indian_cities = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow',
    'Kanpur', 'Nagpur', 'Patna', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Vadodara', 'Firozabad', 'Ludhiana'
]

# List of Indian languages
indian_languages = [
    'English', 'Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi', 'Gujarati', 'Kannada', 'Oriya', 'Punjabi',
    'Assamese', 'Malayalam', 'Urdu', 'Sanskrit', 'Kashmiri', 'Konkani', 'Manipuri', 'Nepali', 'Sindhi', 'Maithili'
]

# List of experience levels
experience_levels = ['Novice', 'Intermediate', 'Expert']

# Generate synthetic LSP profiles
def generate_lsp_profile():
    years_of_experience = random.randint(1, 30)  # Generate a random number for years of experience
    normalized_experience = (years_of_experience - 1) / (30 - 1)  # Normalize years of experience to [0, 1]
    return {
        'name': fake.name(),
        'area_of_expertise': fake.random_element(elements=('Criminal Law', 'Family Law', 'Corporate Law', 'Intellectual Property')),
        'location': fake.random_element(elements=indian_cities),
        # 'qualifications': fake.sentence(),
        'availability': fake.random_element(elements=('Full-time', 'Part-time')),
        'years_of_experience': years_of_experience,  # Actual years of experience
        'languages_spoken': fake.random_element(elements=indian_languages),
        'fee_structure': random.randint(1000, 50000),
    }

# Generate synthetic user profiles
def generate_user_profile():
    experience_level = fake.random_element(elements=experience_levels)  # Randomly select an experience level
    return {
        'name': fake.name(),
        'legal_needs': fake.random_element(elements=('Divorce', 'Property Dispute', 'Contract Review', 'Criminal Defense')),
        'location': fake.random_element(elements=indian_cities),
        'availability': fake.random_element(elements=('Full-time', 'Part-time')),
        'experience_level': experience_level,  # Experience level for the user
        'preferred_language': fake.random_element(elements=indian_languages),
        'budget_constraints': random.randint(1000, 50000),
    }

# Map experience levels to numerical values
experience_level_mapping = {
    'Novice': 0,
    'Intermediate': 0.5,
    'Expert': 1,
}

# Generate synthetic interaction data
def generate_interaction_data(lsp_count, user_count):
    interactions = []
    for _ in range(random.randint(500, 1000)):
        interaction = {
            'user_id': random.randint(1, user_count),
            'lsp_id': random.randint(1, lsp_count),
            'booking_date': fake.date_this_year(before_today=True, after_today=False),
            'rating': random.randint(1, 5),
        }
        interactions.append(interaction)
    return interactions

# Generate LSP profiles
lsp_profiles = [generate_lsp_profile() for _ in range(100)]

# Generate user profiles
user_profiles = [generate_user_profile() for _ in range(200)]

# Generate interaction data
interactions = generate_interaction_data(len(lsp_profiles), len(user_profiles))

# Create dataframes
lsp_df = pd.DataFrame(lsp_profiles)
user_df = pd.DataFrame(user_profiles)
interactions_df = pd.DataFrame(interactions)

# Map experience levels to numerical values in the user dataframe
user_df['experience_level'] = user_df['experience_level'].map(experience_level_mapping)

# Save data to CSV files
lsp_df.to_csv('./data/lsp_profiles.csv', index=False)
user_df.to_csv('./data/user_profiles.csv', index=False)
interactions_df.to_csv('./data/interactions.csv', index=False)
