import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "notebook", "models")
    MLFLOW_DB_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow_database', 'mlflow.db')}"

    # Model Names (as registered in MLflow)
    MODEL_NAME_DLEM = "dlem"
    MODEL_NAME_RL = "RL-graph-emb"
    MODEL_NAME_TWO_TOWER = "two_tower_candidate_job"

    # Pickle Files
    CANDIDATE_OHE_PATH = os.path.join(MODELS_DIR, "candidate_ohe.pkl")
    JOB_OHE_PATH = os.path.join(MODELS_DIR, "job_ohe.pkl")
    WORD2VEC_PATH = os.path.join(MODELS_DIR, "job_candidate_word2vec.kv")
    
    # Candidate Vectorizer Paths
    CANDIDATE_VEC_DEGREE_INSTITUTES = os.path.join(MODELS_DIR, "candidate_vectorizer_degree_institutes.pkl")
    CANDIDATE_VEC_DEGREE_NAMES = os.path.join(MODELS_DIR, "candidate_vectorizer_degree_names.pkl")
    CANDIDATE_VEC_DEGREE_MAJORS = os.path.join(MODELS_DIR, "candidate_vectorizer_degree_majors.pkl")
    CANDIDATE_VEC_DEPARTMENTS = os.path.join(MODELS_DIR, "candidate_vectorizer_preferredJobCategory_department_names.pkl")
    CANDIDATE_VEC_INDUSTRIES = os.path.join(MODELS_DIR, "candidate_vectorizer_preferredJobCategory_industry_names.pkl")

    # Ordinal Encoder Paths  
    QUALIFICATION_ENCODER_PATH = os.path.join(MODELS_DIR, "qualification_ordinal_encoder.pkl")
    LEVEL_ENCODER_PATH = os.path.join(MODELS_DIR, "level_ordinal_encoder.pkl")

settings = Config()
