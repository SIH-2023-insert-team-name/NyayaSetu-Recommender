from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import math
import torch
import pickle
import json
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Define a Pydantic model for the JSON request
class JsonRequest(BaseModel):
    key: str

# Define a route that accepts a JSON request and returns a response
@app.post("/process_json")
async def process_json(json_data: JsonRequest):
    # Access the JSON data from the request
    legal_needs = json_data.legal_needs
    location = json_data.location
    availability = json_data.availability
    experience_level = json_data.experience_level
    preferred_language 	= json_data.preferred_language 	
    budget_constraints = json_data.budget_constraints

    print("Hello")

    # Read dictionary pkl file
    with open('lat_long_mapping.pkl', 'rb') as fp:
        lat_long_mapping = pickle.load(fp)

    # Encode user data
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    legal_needs = model.encode(legal_needs, convert_to_tensor=True)
    location = lat_long_mapping[location]
    availability = (1 if availability == "Full-time" else 0)
    preferred_language = model.encode(preferred_language, convert_to_tensor=True)

    # Read encoded lawyer database
    df_lsp_transf = pd.read_csv('./data/transformed_lsp_profiles')

    user_dict = {
        'legal_needs': legal_needs,
        'location': location,
        'availability': availability,
        'experience_level': experience_level,
        'preferred_language': preferred_language,
        'budget_constraints': budget_constraints
    }


    # Ranking criteria for features of the model
    # 1. area of expertise and location 
    # 2. fee structure and years of experience
    # 3. language spoken
    # 4. Availability
    similarity_dict = {}
    weights = {
        "name": 0, 
        "area_of_expertise": 6, 
        "location": 4, 
        "availability": 1, 
        "years_of_experience": 3, 
        "languages_spoken": 2, 
        "fee_structure": 3
    }

    for index in range(len(df_lsp_transf)):
        similarity = 0
        
        for (lsp_col, user_col) in zip(df_lsp_transf.columns, user_dict.keys()):
            
            lsp_feature = df_lsp_transf.iloc[index][lsp_col]
            user_feature = user_dict[user_col]
            
            if lsp_col == 'name':
                continue
            
            if isinstance(lsp_feature, tuple):
                lsp_feature = torch.tensor(lsp_feature)
                
            if isinstance(user_feature, tuple):
                user_feature = torch.tensor(user_feature)
            
            if isinstance(lsp_feature, (np.int64, np.float64, int)):
                # Calculate distance
                distance = (lsp_feature - user_feature)
                
                # map it to range (-1, 1) with sigmoid
                mapped_distance = math.tanh(distance)
                similarity += weights[lsp_col] * mapped_distance
            
            else:
                similarity += weights[lsp_col] * util.pytorch_cos_sim(lsp_feature, user_feature).item()
        
        similarity_dict[index] = similarity

    sorted_similarity_dict = dict(sorted(similarity_dict.items(), key=lambda item: item[1], reverse=True))
    json_string = json.dumps(sorted_similarity_dict)

    return json_string
