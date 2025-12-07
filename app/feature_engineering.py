import pickle
import torch
import numpy as np
import pandas as pd
import bentoml
import mlflow
from gensim.models import KeyedVectors
from app.config import settings
from app.preprocessing import clean_text, log_normalize, tokenize_whitespace_remove_special
from app.schemas import CandidateInput, JobInput

class FeatureEngineer:
    def __init__(self):
        print("Loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load OHE Models
        with open(settings.CANDIDATE_OHE_PATH, 'rb') as f:
            self.candidate_ohe = pickle.load(f)
        with open(settings.JOB_OHE_PATH, 'rb') as f:
            self.job_ohe = pickle.load(f)
            
        # Load Vectorizer Models
        with open(settings.CANDIDATE_VEC_DEGREE_INSTITUTES, 'rb') as f:
            self.vec_degree_institutes = pickle.load(f)
        with open(settings.CANDIDATE_VEC_DEGREE_NAMES, 'rb') as f:
            self.vec_degree_names = pickle.load(f)
        with open(settings.CANDIDATE_VEC_DEGREE_MAJORS, 'rb') as f:
            self.vec_degree_majors = pickle.load(f)
        with open(settings.CANDIDATE_VEC_DEPARTMENTS, 'rb') as f:
            self.vec_departments = pickle.load(f)
        with open(settings.CANDIDATE_VEC_INDUSTRIES, 'rb') as f:
            self.vec_industries = pickle.load(f)
            
        # Load Ordinal Encoders
        with open(settings.QUALIFICATION_ENCODER_PATH, 'rb') as f:
            self.qualification_encoder = pickle.load(f)
        with open(settings.LEVEL_ENCODER_PATH, 'rb') as f:
            self.level_encoder = pickle.load(f)
            
        # Load Word2Vec
        self.w2v_model = KeyedVectors.load(settings.WORD2VEC_PATH)
        
        # Load DLEM and RL Models via BentoML/MLflow
        try:
            self.dlem_service = bentoml.mlflow.load_model(f"{settings.MODEL_NAME_DLEM}:latest")
            self.dlem_model = self.dlem_service._model_impl.get_raw_model()
            self.dlem_model.to(self.device)
            self.dlem_model.eval()
            
            self.rl_service = bentoml.mlflow.load_model(f"{settings.MODEL_NAME_RL}:latest")
            self.rl_model = self.rl_service._model_impl.get_raw_model()
            self.rl_model.to(self.device)
            self.rl_model.eval()
        except Exception as e:
            print(f"Error loading BentoML models: {e}")
            raise e
        print("Models loaded successfully.")

    def sentence_to_tensor(self, text, max_len=5000, embed_dim=128):
        tokens = str(text).lower().split()
        embeddings = []
        for token in tokens[:max_len]:
            if token in self.w2v_model:
                embeddings.append(self.w2v_model[token])
            else:
                embeddings.append(np.zeros(embed_dim))
        
        if len(embeddings) < max_len:
            padding = np.zeros((max_len - len(embeddings), embed_dim))
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = np.array(embeddings[:max_len])
            
        tensor = torch.tensor(embeddings, dtype=torch.float32)
        tensor = tensor.transpose(0, 1) # (embed_dim, max_len)
        return tensor

    def get_dlem_embedding(self, text):
        with torch.no_grad():
            tensor = self.sentence_to_tensor(text)
            tensor = tensor.unsqueeze(0).to(self.device)
            embedding = self.dlem_model.get_job_embedding(tensor)
            embedding = embedding.detach().cpu()
        return embedding.tolist()[0]

    def get_single_node_vector(self, node_name, node_type):
        if not isinstance(node_name, str) or len(node_name.strip()) == 0:
            node_name = "dummy"
        with torch.no_grad():
            emb = self.rl_model.encode(node_name, node_type).detach().cpu().numpy()
        return emb.tolist()

    def get_weighted_node_cluster_vector(self, node_names, node_weights, node_type, include_stats=False):
        # NOTE: include_stats=False to match model dimension (50 dim only)
        if not isinstance(node_names, list): node_names = []
        if not isinstance(node_weights, list): node_weights = []
        
        safe_weights = []
        for w in node_weights:
            try:
                safe_weights.append(float(w))
            except (ValueError, TypeError):
                safe_weights.append(0.0)
        node_weights = safe_weights
        
        min_len = min(len(node_names), len(node_weights))
        node_names = node_names[:min_len]
        node_weights = node_weights[:min_len]
        
        if len(node_names) == 0 or sum(node_weights) == 0:
            dummy_vec = self.rl_model.encode("dummy", node_type).detach().cpu().numpy()
            if include_stats:
                return list(dummy_vec) + [0.0, 0.0, 0.0, 0, 0.0]
            else:
                return list(dummy_vec)
            
        vectors = []
        for name in node_names:
            with torch.no_grad():
                emb = self.rl_model.encode(name, node_type).detach().cpu().numpy()
                vectors.append(emb)
        vectors = np.array(vectors, dtype=np.float32)
        weights = np.array(node_weights, dtype=np.float32).reshape(-1, 1)
        
        if weights.sum() == 0:
            weighted_avg = vectors.mean(axis=0)
        else:
            weighted_avg = (vectors * weights).sum(axis=0) / weights.sum()
        
        if include_stats:
            dists = np.linalg.norm(vectors - weighted_avg, axis=1) if len(vectors) > 0 else np.array([0.0])
            eps = 1e-8
            cluster_vec = np.concatenate([
                weighted_avg,
                [dists.std()],
                [dists.min()],
                [dists.max()],
                [len(node_names)],
                [1.0 / (dists.std() + eps)]
            ])
            
            norm = np.linalg.norm(cluster_vec)
            if norm > 0:
                cluster_vec_normalized = cluster_vec / norm
            else:
                cluster_vec_normalized = cluster_vec
            return cluster_vec_normalized.tolist()
        else:
            return weighted_avg.tolist()

    def transform_candidate(self, candidate: CandidateInput):
        # 1. One-Hot Encoding
        ohe_cols = ['gender', 'martial_status', 'searching_for_job_status', 'district_name', 
                    'salary_currency_name', 'salary_type_name', 'candidate_type']
        ohe_data = pd.DataFrame([candidate.dict()])[ohe_cols]
        onehot_vec = self.candidate_ohe.transform(ohe_data)[0].tolist()
        
        # 2. Vectorizer Encoding (Tfidf)
        def get_vec(vectorizer, data_list):
            text = " ".join(data_list) if isinstance(data_list, list) else str(data_list)
            vec = vectorizer.transform([text])
            if hasattr(vec, "toarray"):
                vec = vec.toarray()
            elif hasattr(vec, "todense"):
                vec = vec.todense()
            return np.array(vec)[0].tolist()

        tfidf_vec = []
        tfidf_vec.extend(get_vec(self.vec_degree_institutes, candidate.degree_institutes))
        tfidf_vec.extend(get_vec(self.vec_departments, candidate.preferredJobCategory_department_names))
        tfidf_vec.extend(get_vec(self.vec_industries, candidate.preferredJobCategory_industry_names))
        tfidf_vec.extend(get_vec(self.vec_degree_names, candidate.degree_names))
        tfidf_vec.extend(get_vec(self.vec_degree_majors, candidate.degree_majors))
        
        # 3. Log Normalization
        log_vec = [
            log_normalize(candidate.present_salary),
            log_normalize(candidate.expected_salary),
            log_normalize(candidate.total_experience),
            log_normalize(candidate.age)
        ]
        
        # 4. Ordinal Encoding
        level_input = pd.DataFrame([[candidate.level_name.lower()]])
        qualification_input = pd.DataFrame([[candidate.qualification_name.lower()]])
        ord_vec = [
            float(self.level_encoder.transform(level_input)[0][0]),
            float(self.qualification_encoder.transform(qualification_input)[0][0])
        ]
        
        # 5. DLEM Embedding
        clean_resume = clean_text(candidate.candidate_latest_resume_text)
        clean_resume = tokenize_whitespace_remove_special(clean_resume)
        dlem_vec = self.get_dlem_embedding(clean_resume)
        
        # 6. RL Embedding
        # Exclude stats and profession to match dimensions
        rl_vec = []
        rl_vec.extend(self.get_weighted_node_cluster_vector(candidate.skills_names, candidate.skills_year_of_experiences, 'skill', include_stats=False))
        rl_vec.extend(self.get_weighted_node_cluster_vector(candidate.candidate_experience_roles, candidate.candidate_experience_role_duration, 'job', include_stats=False))
        # Exclude profession
        # rl_vec.extend(self.get_single_node_vector(candidate.profession, 'job'))
        
        # Assemble
        feature_vector = onehot_vec + tfidf_vec + log_vec + ord_vec + dlem_vec + rl_vec
        return feature_vector

    def transform_job(self, job: JobInput):
        # 1. One-Hot Encoding
        ohe_cols = ['job_gender', 'position_name', 'job_district_name', 'job_type_name', 
                    'salary_currency', 'job_salary_type', 'industry_name', 'department_name']
        ohe_data = pd.DataFrame([job.dict()])[ohe_cols]
        onehot_vec = self.job_ohe.transform(ohe_data)[0].tolist()
        
        # 2. Log Normalization
        log_vec = [
            log_normalize(job.minimum_salary),
            log_normalize(job.maximum_salary),
            log_normalize(job.age_from),
            log_normalize(job.age_to),
            log_normalize(job.job_experience),
            log_normalize(job.minimum_experience),
            log_normalize(job.maximum_experience)
        ]
        
        # 3. Ordinal Encoding
        job_level_input = pd.DataFrame([[job.job_level_name.lower()]])
        job_qualification_input = pd.DataFrame([[job.job_qualification_name.lower()]])
        qualification_prefer_input = pd.DataFrame([[job.qualification_prefer_name.lower()]])
        ord_vec = [
            float(self.level_encoder.transform(job_level_input)[0][0]),
            float(self.qualification_encoder.transform(job_qualification_input)[0][0]),
            float(self.qualification_encoder.transform(qualification_prefer_input)[0][0])
        ]
        
        # 4. DLEM Embedding
        job_text = (job.job_requirement or "") + " " + (job.job_description or "")
        clean_job_text = clean_text(job_text)
        clean_job_text = tokenize_whitespace_remove_special(clean_job_text)
        dlem_vec = self.get_dlem_embedding(clean_job_text)
        
        # 5. RL Embedding
        # Include stats for job skills to match dimensions (580 = 401 + 7 + 3 + 64 + 55 + 50)
        rl_vec = []
        rl_vec.extend(self.get_weighted_node_cluster_vector(job.job_skill_name, job.job_skill_experience, 'skill', include_stats=True))
        rl_vec.extend(self.get_single_node_vector(job.job_title, 'job'))
        
        # Assemble
        feature_vector = onehot_vec + log_vec + ord_vec + dlem_vec + rl_vec
        return feature_vector

feature_engineer = FeatureEngineer()
