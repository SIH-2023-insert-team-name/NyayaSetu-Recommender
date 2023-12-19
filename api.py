import numpy as np
import math
import ast
import pandas as pd
import torch

import pickle
import uvicorn
import requests

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

torch.device('cpu')

def process_lawyer_list(df_lsp_transf):

    # Get the preprocessed latititude and longitude coordinates
    with open('./intermediate_files/lat_long_mapping.pkl', 'rb') as fp:
        lat_long_mapping = pickle.load(fp)

    # Load the embeddings model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    # Encode the area of expertise of the lawyer 
    # Also encode the legal needs of the user
    df_lsp_transf['category'] = df_lsp_transf['category'].apply(lambda phrase: 
                                                                 model.encode(phrase, convert_to_tensor=True))

    # Encode the years of experience of a lawyer into the vector
    min_years = df_lsp_transf['experience'].min()
    max_years = df_lsp_transf['experience'].max()
    df_lsp_transf['experience'] = (df_lsp_transf['experience'] - min_years) / (max_years - min_years)

    # Encode the availability of the lawyer into the vector
    df_lsp_transf['availability'] = df_lsp_transf['availability'].apply(lambda job_type: 
                                                                        1 if job_type == "Full-time" else 0)
   
    # Encode the languages spoken and preferred language into a vector
    df_lsp_transf['languages_spoken'] = df_lsp_transf['languages_spoken'].apply(lambda phrase: 
                                                                                model.encode(phrase, convert_to_tensor=True))
                                                                    

    # Encode the location into latitude and longitude
    df_lsp_transf['location'] = df_lsp_transf['location'].apply(lambda location: 
                                                                lat_long_mapping.get(location, get_lat_long(location)))
    
    # Save the loaded PyTorch object to a Pickle file using pickle.dump
    with open('./intermediate_files/df_lsp_transf.pkl', 'wb') as fp:
        pickle.dump(df_lsp_transf, fp)

def get_lat_long(location):
    # Define the URL for the Nominatim API request with your string address
    # address = "Bangalore"
    encoded_address = requests.utils.quote(location)  # URL encode the address
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_address}&format=json"

    # Make the GET request
    response = requests.get(url)

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        data = response.json()
        # Process the response data to extract latitude and longitude
        if len(data) > 0:
            lat = data[0]['lat']
            lon = data[0]['lon']
        else:
            print("Location not found.")
    else:
        print("Error: Unable to retrieve data from Nominatim API.")

    # Append the result to the pickle file
    with open('lat_long_mapping.pkl', 'ab') as file:
        pickle.dump({location: (float(lat), float(lon))}, file)

    return float(lat), float(lon)

# def save_lat_long(df_lsp):
#     lat_long_mapping = {}

#     for city in set(df_lsp['location']):
#         lat_long_mapping[city] = get_lat_long(city)
        
#     with open('./intermediate_files/lat_long_mapping.pkl', 'wb') as fp:
#         pickle.dump(lat_long_mapping, fp)
#         print('dictionary saved successfully to file')

    
# def check_double_quotes(my_string):
# # Check if the string has double quotes around it
#     if my_string.startswith('"') and my_string.endswith('"'):
#         True
#     else:
#         False

def recommend_lawyer(legal_needs, location, availability, experience_level, preferred_language, budget_constraints):
    # Read dictionary pkl file
    with open('./intermediate_files/lat_long_mapping.pkl', 'rb') as fp:
        lat_long_mapping = pickle.load(fp)

    with open('./intermediate_files/df_lsp_transf.pkl', 'rb') as fp:
        df_lsp_transf = pickle.load(fp)

    # Encode user data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
    
    legal_needs = model.encode(legal_needs, convert_to_tensor=True)
    location = lat_long_mapping[location]
    availability = (1 if availability == "Full-time" else 0)
    preferred_language = model.encode(preferred_language, convert_to_tensor=True)

    user_dict = {
        # 'name': user_name,
        'legal_needs': legal_needs,
        'location': location,
        'availability': availability,
        'experience_level': experience_level,
        'preferred_language': preferred_language,
        'budget_constraints': budget_constraints
    }

    # Ranking criteria for features of the modelkey: int
    # 1. area of expertise and location 
    # 2. fee structure and years of experience
    # 3. language spoken
    # 4. Availability
    similarity_dict = {}
    weights = {
        # "name": 0, 
        "category": 10, 
        "location": 4, 
        "availability": 1, 
        "experience": 3, 
        "languages_spoken": 2, 
        "cost": 3
    }

    for index in range(len(df_lsp_transf)):
        similarity = 0
        
        for (lsp_col, user_col) in zip(df_lsp_transf.columns, user_dict.keys()):
            lsp_feature = df_lsp_transf.iloc[index][lsp_col]
            user_feature = user_dict[user_col]
            
            if lsp_col not in weights.keys():
                continue

            # if check_double_quotes(lsp_feature):
            #     lsp_feature = ast.literal_eval(lsp_feature)

            if isinstance(lsp_feature, tuple):
                lsp_feature = lsp_feature
            
            if isinstance(lsp_feature, tuple):
                lsp_feature = torch.tensor(lsp_feature)
                
            if isinstance(user_feature, tuple):
                user_feature = torch.tensor(user_feature)
            
            if isinstance(lsp_feature, (np.int64, np.float64, int)):
                # Calculate distance
                sigma = 5000.0
                # Gassian curve. Recommend highly when difference is close to zero
                similarity_score = math.exp(-((user_feature - lsp_feature)**2) / (2 * sigma**2))
                similarity += weights[lsp_col] * similarity_score
            
            else:
                similarity += weights[lsp_col] * util.pytorch_cos_sim(lsp_feature, user_feature).item()

        unique_id = df_lsp_transf.loc[index, "_id"]
        similarity_dict[unique_id] = similarity

    sorted_similarity_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True))
    recommended_ids_sorted = [key for key in sorted_similarity_dict.keys()]

    return recommended_ids_sorted

class RecommendationRequest(BaseModel):
    user_name: str
    legal_needs: str
    location: str
    availability: str
    experience_level: int
    preferred_language: str
    budget_constraints: float

# Define a Pydantic model for the JSON response
class RecommendationResponse(BaseModel):
    recommendations: List

# Define a route that accepts a JSON request and returns a response
@app.post("/recommend_lawyer")
async def recommend_lawyer_endpoint(request: RecommendationRequest):
    # try:
    # Call your recommendation function with the provided parameters
    recommended_lawyer = recommend_lawyer(
        # request.user_name,
        request.legal_needs,
        request.location,
        request.availability,
        request.experience_level,
        request.preferred_language,
        request.budget_constraints
    )

    return RecommendationResponse(recommendations=recommended_lawyer)
    
    # except Exception as e:
    # raise HTTPException(status_code=500, detail=str(e))

# Assume your existing GET endpoint is hosted at this URL
get_lawyer_url = "https://nyayasetu-backend.onrender.com/get/lawyers"

@app.post("/process_and_store")
async def process_and_store_lawyers():
    # Make a GET request to the existing endpoint
    response = requests.get(get_lawyer_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Your logic to process the data and store it in an intermediate file
        lawyer_list = response.json()
        lsp_df = pd.DataFrame(lawyer_list)
        process_lawyer_list(lsp_df)

        return {"message": "Data processed and stored successfully"}

    # Handle other status codes if needed
    else:
        return {"error": f"Failed to retrieve data. Status code: {response.status_code}"}
    
@app.get("/")
def default():
    return 'Nyayasetu Recommendation Algorithm'

if __name__ == "__main__":
    # process_and_store_lawyers()
    uvicorn.run(app, port=8000, host="0.0.0.0")

    