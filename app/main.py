from fastapi import FastAPI, HTTPException
from app.schemas import CandidateInput, JobInput
from app.feature_engineering import feature_engineer
from app.model_loader import two_tower_inference

app = FastAPI(title="Two-Tower Model Serving API")

@app.get("/")
def read_root():
    return {"message": "Two-Tower Model Serving API is running"}

@app.post("/embed/candidate")
def embed_candidate(candidate: CandidateInput):
    try:
        # 1. Feature Engineering
        feature_vector = feature_engineer.transform_candidate(candidate)
        
        # 2. Model Inference
        embedding = two_tower_inference.get_candidate_embedding(feature_vector)
        
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/job")
def embed_job(job: JobInput):
    try:
        # 1. Feature Engineering
        feature_vector = feature_engineer.transform_job(job)
        
        # 2. Model Inference
        embedding = two_tower_inference.get_job_embedding(feature_vector)
        
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
