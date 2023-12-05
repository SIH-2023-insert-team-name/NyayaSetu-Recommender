from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import math
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

def recommend_lawyer(user_name, legal_needs, location, availability, experience_level, preferred_language, budget_constraints):
    # Read dictionary pkl file
    with open('./intermediate_files/lat_long_mapping.pkl', 'rb') as fp:
        lat_long_mapping = pickle.load(fp)

    with open('./intermediate_files/df_lsp_transf.pkl', 'rb') as fp:
        df_lsp_transf = pickle.load(fp)

    # Encode user data
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    legal_needs = model.encode(legal_needs, convert_to_tensor=True)
    location = lat_long_mapping[location]
    availability = (1 if availability == "Full-time" else 0)
    preferred_language = model.encode(preferred_language, convert_to_tensor=True)

    # Read encoded lawyer database
    # df_lsp_transf = pd.read_csv('./data/transformed_lsp_profiles.csv')

    user_dict = {
        'name': user_name,
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
            
            if lsp_col == 'Lawyer name':
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
    # json_string = json.dumps(sorted_similarity_dict)
    recommended_ids_sorted = [key for key in sorted_similarity_dict.keys()]

    return recommended_ids_sorted

sorted_dict = recommend_lawyer("Rishab", "Criminal Defense","Hyderabad","Part-time",1.0,"Malayalam",44973)
print(sorted_dict)


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
    try:
        # Call your recommendation function with the provided parameters
        recommended_lawyer = recommend_lawyer(
            request.user_name,
            request.legal_needs,
            request.location,
            request.availability,
            request.experience_level,
            request.preferred_language,
            request.budget_constraints
        )
        return recommended_lawyer
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    