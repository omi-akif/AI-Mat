import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature Engineering Pipeline

    This notebook handles the feature engineering process for both Jobs and Candidates.
    It performs the following steps:
    1.  **Load Data**: Reads cleaned data and helper datasets.
    2.  **One-Hot Encoding**: Encodes categorical variables.
    3.  **Doc2Vec Encoding**: Encodes text-based lists (e.g., degrees, industries) using MultiLabelBinarizer (as a proxy or placeholder, actual Doc2Vec seems to be handled via `bentoml` models later?). *Correction*: The code uses `MultiLabelBinarizer` for some fields and `bentoml` loaded models for others.
    4.  **Log Normalization**: Normalizes continuous variables like salary and experience.
    5.  **Ordinal Encoding**: Encodes ordinal variables like education level.
    6.  **Deep Learning Embeddings (DLEM)**: Generates embeddings for resumes and job descriptions using a pre-trained DLEM model.
    7.  **Reinforcement Learning (RL) Embeddings**: Generates embeddings for skills, roles, and professions using a pre-trained RL graph model.
    8.  **Feature Vector Assembly**: Concatenates all features into a single vector for each job and candidate.
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import pandas as pd
    import bentoml
    import mlflow
    from gensim.models import KeyedVectors
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
    from tqdm import tqdm
    import ast
    import random
    from sklearn.preprocessing import MultiLabelBinarizer
    import pickle
    import marimo as mo
    return (
        MultiLabelBinarizer,
        OrdinalEncoder,
        ast,
        mlflow,
        mo,
        np,
        pd,
        pickle,
        torch,
        tqdm,
    )


@app.cell
def _(mlflow, tqdm):
    import os
    mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow_database/mlflow.db')}")
    mlflow_client = mlflow.tracking.MlflowClient()
    tqdm.pandas()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)
    return


@app.cell
def _(pd):
    candidate_df = pd.read_csv('../datasets/cleaned_datasets/candidate_data_df_clean.csv')
    job_df = pd.read_csv('../datasets/cleaned_datasets/job_data_df_clean.csv')

    # skills_df = pd.read_csv('../datasets/helping_datasets/skills_list.csv')
    # positions_df = pd.read_csv('../datasets/helping_datasets/positions_list.csv')
    # # location_df = pd.read_csv('../datasets/helping_datasets/locations_list.csv')

    # degrees_df = pd.read_csv('../datasets/helping_datasets/degrees_list.csv')
    # majors_df = pd.read_csv('../datasets/helping_datasets/majors_list.csv')
    # institutes_df = pd.read_csv('../datasets/helping_datasets/institutes_list.csv')

    # industry_df = pd.read_csv('../datasets/helping_datasets/industries_list.csv')
    # department_df = pd.read_csv('../datasets/helping_datasets/departments_list.csv')
    return candidate_df, job_df


@app.cell
def _(candidate_df, job_df):
    print("Job Data Shape:", job_df.shape)
    print("Candidate Data Shape:", candidate_df.shape)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Dataset Columns Overview

    **Job Data Columns:**
    *   `post_id`: Identifier
    *   `job_title`: RL Encoding
    *   `job_description`, `job_requirement`: Text (DLEM Encoding)
    *   `job_experience`, `minimum_experience`, `maximum_experience`: Continuous (Log Norm)
    *   `minimum_salary`, `maximum_salary`: Continuous (Log Norm)
    *   `negotiable`: Binary
    *   `age_from`, `age_to`: Continuous (Log Norm)
    *   `job_gender`: Categorical (One-Hot)
    *   `industry_name`, `department_name`: MultiLabelBinarizer Encoding
    *   `position_name`, `job_district_name`, `job_type_name`: Categorical (One-Hot)
    *   `job_level_name`, `job_qualification_name`, `qualification_prefer_name`: Ordinal
    *   `salary_currency`, `job_salary_type`: Categorical (One-Hot)
    *   `job_skill_name`: RL Encoding (List)
    *   `job_skill_experience`: Weights (List)

    **Candidate Data Columns:**
    *   `id`: Identifier
    *   `expected_salary`, `present_salary`: Continuous (Log Norm)
    *   `gender`, `martial_status`, `searching_for_job_status`: Categorical (One-Hot)
    *   `total_experience`: Continuous (Log Norm)
    *   `district_name`, `salary_currency_name`, `salary_type_name`: Categorical (One-Hot)
    *   `level_name`, `qualification_name`: Ordinal
    *   `degree_institutes`, `degree_names`, `degree_majors`: MultiLabelBinarizer Encoding
    *   `skills_names`: RL Encoding (List)
    *   `skills_year_of_experiences`: Weights (List)
    *   `candidate_experience_roles`: RL Encoding (List)
    *   `candidate_experience_role_duration`: Weights
    *   `age`: Continuous (Log Norm)
    *   `candidate_latest_resume_text`: DLEM Encoding
    """)
    return


@app.cell
def _(candidate_df, job_df):
    job_df['job_text_data'] = job_df['job_requirement'] + ' ' + job_df['job_description']

    # Mapping for better cleaning of martial status
    status_mapping = {
        "bachelor": "unmarried",
        "cª‡hvr¨ b‡n": "unmarried",
        "divorcee": "unmarried",
        "marred": "married",
        "marriage": "married",
        "married": "married",
        "singal": "unmarried",
        "single": "unmarried",
        "un-married": "unmarried",
        "unidentified": "unidentified",
        "unmaried": "unmarried",
        "unmarred": "unmarried",
        "unmarried": "unmarried",
        "unmerrid": "unmarried",
        "ﺔﻴﻨﻬﻤﻟا": "unmarried"
    }

    candidate_df['martial_status'] = candidate_df['martial_status'].str.lower().replace(status_mapping)


    candidate_df['skills_names'] = candidate_df['skills_names'].apply(lambda x: x if isinstance(x, list) else [])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Type Casting
    """)
    return


@app.cell
def _(ast, candidate_df, job_df):
    # ------------------------------
    #  IDENTIFIER COLUMNS
    # ------------------------------
    candidate_df['id'] = candidate_df['id'].astype('int64')

    # ------------------------------
    #  TEXT COLUMNS
    # ------------------------------
    candidate_df['candidate_latest_resume_text'] = candidate_df['candidate_latest_resume_text'].astype('string')

    # ------------------------------
    #  NUMERICAL (CONTINUOUS) COLUMNS
    # ------------------------------
    candidate_df['expected_salary'] = candidate_df['expected_salary'].astype('float64')
    candidate_df['present_salary'] = candidate_df['present_salary'].astype('float64')
    candidate_df['total_experience'] = candidate_df['total_experience'].astype('float64')
    candidate_df['age'] = candidate_df['age'].astype('int64')

    # ------------------------------
    #  CATEGORICAL COLUMNS
    # ------------------------------ 
    candidate_df['gender'] = candidate_df['gender'].astype('category')
    candidate_df['martial_status'] = candidate_df['martial_status'].astype('category')
    candidate_df['searching_for_job_status'] = candidate_df['searching_for_job_status'].astype('category')
    candidate_df['district_name'] = candidate_df['district_name'].astype('category')
    candidate_df['salary_currency_name'] = candidate_df['salary_currency_name'].astype('category')
    candidate_df['salary_type_name'] = candidate_df['salary_type_name'].astype('category')
    candidate_df['candidate_type'] = candidate_df['candidate_type'].astype('category')

    # ------------------------------
    #  ORDINAL COLUMNS
    # ------------------------------
    candidate_df['level_name'] = candidate_df['level_name'].astype('category')
    candidate_df['qualification_name'] = candidate_df['qualification_name'].astype('category')

    # ------------------------------
    #  LIST / COMPLEX TEXT COLUMNS
    # ------------------------------
    candidate_df['degree_institutes'] = candidate_df['degree_institutes'].astype('object')
    candidate_df['preferredJobCategory_department_names'] = candidate_df['preferredJobCategory_department_names'].astype('object')
    candidate_df['preferredJobCategory_industry_names'] = candidate_df['preferredJobCategory_industry_names'].astype('object')
    candidate_df['degree_names'] = candidate_df['degree_names'].astype('object')
    candidate_df['degree_majors'] = candidate_df['degree_majors'].astype('object')

    # ------------------------------
    #  RL ENCODING / LIST-BASED COLUMNS
    # ------------------------------
    candidate_df['profession'] = candidate_df['profession'].astype('object')
    candidate_df['skills_names'] = candidate_df['skills_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    candidate_df['skills_year_of_experiences'] = candidate_df['skills_year_of_experiences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    candidate_df['candidate_experience_roles'] = candidate_df['candidate_experience_roles'].astype('object')
    candidate_df['candidate_experience_role_duration'] = candidate_df['candidate_experience_role_duration'].astype('object')


    # Job DataFrame: type selection
    # ------------------------------
    #  BASIC / IDENTIFIER COLUMNS
    # ------------------------------
    job_df['post_id'] = job_df['post_id'].astype('object')

    # ------------------------------
    #  TEXT COLUMNS
    # ------------------------------
    job_df['job_title'] = job_df['job_title'].astype('string')
    job_df['job_text_data'] = job_df['job_text_data'].astype('string')


    # ------------------------------
    #  NUMERICAL BOOLEAN
    # ------------------------------
    job_df['negotiable'] = job_df['negotiable'].astype('int64')

    # ------------------------------
    #  NUMERICAL (CONTINUOUS) COLUMNS
    # ------------------------------
    job_df['job_experience'] = job_df['job_experience'].astype('float64')
    job_df['minimum_experience'] = job_df['minimum_experience'].astype('float64')
    job_df['maximum_experience'] = job_df['maximum_experience'].astype('float64')
    job_df['minimum_salary'] = job_df['minimum_salary'].astype('float64')
    job_df['maximum_salary'] = job_df['maximum_salary'].astype('float64')
    job_df['age_from'] = job_df['age_from'].astype('int64')
    job_df['age_to'] = job_df['age_to'].astype('int64')

    # ------------------------------
    #  CATEGORICAL COLUMNS
    # ------------------------------
    job_df['job_gender'] = job_df['job_gender'].astype('category')
    job_df['industry_name'] = job_df['industry_name'].astype('category')
    job_df['department_name'] = job_df['department_name'].astype('category')
    job_df['position_name'] = job_df['position_name'].astype('category')
    job_df['job_district_name'] = job_df['job_district_name'].astype('category')
    job_df['job_type_name'] = job_df['job_type_name'].astype('category')
    job_df['job_level_name'] = job_df['job_level_name'].astype('category')
    job_df['job_qualification_name'] = job_df['job_qualification_name'].astype('category')
    job_df['qualification_prefer_name'] = job_df['qualification_prefer_name'].astype('category')
    job_df['salary_currency'] = job_df['salary_currency'].astype('category')
    job_df['job_salary_type'] = job_df['job_salary_type'].astype('category')

    # ------------------------------
    #  LIST / COMPLEX TEXT COLUMNS
    # ------------------------------
    job_df['job_skill_name'] = job_df['job_skill_name'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    job_df['job_skill_experience'] = job_df['job_skill_experience'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Load Models (Word2Vec, DLEM, RL)
    """)
    return


app._unparsable_cell(
    r"""
    candidate_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    job_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    job_mlb = MultiLabelBinarizer(sparse_ouI am not seeing the codes thattput=False)
    candidate_mlb = MultiLabelBinarizer(sparse_output=False)

    # Load Word2Vec model
    w2v_model = KeyedVectors.load('models/job_candidate_word2vec.kv')

    # Load DLEM model
    model_name_dlem = 'dlem'
    model_name_rl = 'RL-graph-emb'


    mlflow_client_local = mlflow.tracking.MlflowClient()

    versions_dlem = mlflow_client_local.search_model_versions(f\"name='{model_name_dlem}'\")
    versions_rl = mlflow_client_local.search_model_versions(f\"name='{model_name_rl}'\")

    latest_dlem = sorted(versions_dlem, key=lambda v: int(v.version))[-1]
    latest_rl = sorted(versions_rl, key=lambda v: int(v.version))[-1]

    bentoml.mlflow.import_model(
        name=model_name_dlem,
        model_uri=latest_dlem.source,
    )
    bentoml.mlflow.import_model(
        name=model_name_rl,
        model_uri=latest_rl.source,
    )


    lit_dlem_model_load = bentoml.mlflow.load_model(\"dlem:latest\")
    rl_model_load = bentoml.mlflow.load_model(\"RL-graph-emb:latest\")


    dlem_model = lit_dlem_model_load._model_impl.get_raw_model()  # This is LitDLEM
    rl_model = rl_model_load._model_impl.get_raw_model()  # This is RL


    device_dlem = next(dlem_model.parameters()).device
    device_rl = next(rl_model.parameters()).device


    def get_model_info(model_name):
        versions = mlflow_client.search_model_versions(f\"name='{model_name}'\")
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        return {
            \"name\": latest.name,
            \"version\": latest.version,
            \"stage\": latest.current_stage,
            \"source\": latest.source,
            \"run_id\": latest.run_id,
        }

    dlem_info = get_model_info(model_name_dlem)
    rl_info = get_model_info(model_name_rl)

    print(\"DLEM Model Info:\", dlem_info)
    print(\"RL Model Info:\", rl_info)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Helper Functions
    """)
    return


@app.cell
def _(np, pd, torch):
    def sentence_to_tensor(text, word2vec_model, max_len=5000, embed_dim=128):
        """Convert text to word2vec embedding tensor."""
        tokens = str(text).lower().split()
        embeddings = []

        for token in tokens[:max_len]:
            if token in word2vec_model:
                embeddings.append(word2vec_model[token])
            else:
                embeddings.append(np.zeros(embed_dim))

        # Pad to max_len
        if len(embeddings) < max_len:
            padding = np.zeros((max_len - len(embeddings), embed_dim))
            embeddings = np.vstack([embeddings, padding])
        else:
            embeddings = np.array(embeddings[:max_len])

        tensor = torch.tensor(embeddings, dtype=torch.float32)
        tensor = tensor.transpose(0, 1)
        return tensor

    def log_normalize_series(series):
        """Apply logarithmic normalization to a pandas Series, handling edge cases."""
        def log_norm(x):
            # Handle NaN, None, and non-numeric values
            if pd.isna(x) or x is None:
                return 0.0
            try:
                x = float(x)
                # Handle negative and zero values
                if x < 0:
                    return 0.0
                elif x == 0:
                    return 0.0
                else:
                    return np.log1p(x)
            except (ValueError, TypeError):
                return 0.0

        return series.apply(log_norm)

    def get_dlem_embedding(text, w2v_model, dlem_model, device_dlem):
        """Get DLEM embedding for text."""
        with torch.no_grad():
            tensor = sentence_to_tensor(text, w2v_model)
            tensor = tensor.unsqueeze(0).to(device_dlem)
            embedding = dlem_model.get_job_embedding(tensor)
            embedding = embedding.detach().cpu()
        del tensor
        return embedding.tolist()


    def get_rl_embedding(rl_model, node_name:str, node_type:str):
        """Get RL vector"""
        with torch.no_grad():
            embedding = rl_model.encode(node_name, node_type)
            embedding = embedding.detach().cpu()

        return embedding.tolist()


    def get_weighted_node_cluster_vector(rl_model, node_names, node_weights, node_type):
        """
        Computes a normalized trait vector for a cluster of weighted node embeddings.
        """
        # Ensure lists
        if not isinstance(node_names, list):
            node_names = []
        if not isinstance(node_weights, list):
            node_weights = []

        # Convert weights to floats safely, non-numeric -> 0
        safe_weights = []
        for w in node_weights:
            try:
                safe_weights.append(float(w))
            except (ValueError, TypeError):
                safe_weights.append(0.0)
        node_weights = safe_weights

        # Trim or pad to match lengths
        min_len = min(len(node_names), len(node_weights))
        node_names = node_names[:min_len]
        node_weights = node_weights[:min_len]

        # Handle empty or all-zero weights
        if len(node_names) == 0 or sum(node_weights) == 0:
            dummy_vec = rl_model.encode("dummy", node_type).detach().cpu().numpy()
            return list(dummy_vec) + [0.0, 0.0, 0.0, 0, 0.0]

        # Step 1: compute weighted embeddings
        vectors = []
        for name in node_names:
            with torch.no_grad():
                emb = rl_model.encode(name, node_type).detach().cpu().numpy()
                vectors.append(emb)
        vectors = np.array(vectors, dtype=np.float32)  # shape (N, D)

        weights = np.array(node_weights, dtype=np.float32).reshape(-1, 1)

        # Weighted average safely
        if weights.sum() == 0:
            weighted_avg = vectors.mean(axis=0)
        else:
            weighted_avg = (vectors * weights).sum(axis=0) / weights.sum()

        # Step 2: distances to centroid
        dists = np.linalg.norm(vectors - weighted_avg, axis=1) if len(vectors) > 0 else np.array([0.0])

        # Step 3: create augmented vector
        eps = 1e-8  # avoid division by zero
        cluster_vec = np.concatenate([
            weighted_avg,                  # weighted average embedding
            [dists.std()],                 # std of distances
            [dists.min()],                 # min distance
            [dists.max()],                 # max distance
            [len(node_names)],             # number of nodes
            [1.0 / (dists.std() + eps)]    # inverse std
        ])

        # Normalize
        norm = np.linalg.norm(cluster_vec)
        if norm > 0:
            cluster_vec_normalized = cluster_vec / norm
        else:
            cluster_vec_normalized = cluster_vec

        return cluster_vec_normalized.tolist()


    def get_single_node_vector(rl_model, node_name, node_type):
        """
        Computes a normalized embedding vector for a single node (text value).
        """
        # Ensure node_name is a string
        if not isinstance(node_name, str) or len(node_name.strip()) == 0:
            node_name = "dummy"

        # Get embedding
        with torch.no_grad():
            emb = rl_model.encode(node_name, node_type).detach().cpu().numpy()

        return emb.tolist()
    return (
        get_dlem_embedding,
        get_single_node_vector,
        get_weighted_node_cluster_vector,
        log_normalize_series,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. One-Hot Encoding
    """)
    return


@app.cell
def _(candidate_df, candidate_ohe, job_df, job_ohe, pickle):
    candidate_df_cols_one_hot = [
        'gender',
        'martial_status',
        'searching_for_job_status',
        'district_name',
        'salary_currency_name',
        'salary_type_name',
        'candidate_type'
    ]


    job_df_cols_one_hot = [
        'job_gender',
        'position_name',
        'job_district_name',
        'job_type_name',
        'salary_currency',
        'job_salary_type',
        'industry_name',
        'department_name'
    ]

    # ------------------------------
    # Candidate DataFrame
    # ------------------------------
    candidate_encoded_ohe_array = candidate_ohe.fit_transform(candidate_df[candidate_df_cols_one_hot])
    job_encoded_ohe_array = job_ohe.fit_transform(job_df[job_df_cols_one_hot])

    # Save the encoder
    with open('models/candidate_ohe.pkl', 'wb') as f_job_ohe:
        pickle.dump(candidate_ohe, f_job_ohe)
    print("Candidate encoder saved as candidate_ohe.pkl")

    # Save the encoder
    with open('models/job_ohe.pkl', 'wb') as f_candidate_ohe:
        pickle.dump(job_ohe, f_candidate_ohe)
    print("Job encoder saved as job_ohe.pkl", )

    # Create a single column with the full one-hot vector as a list
    candidate_df['onehot_vec'] = candidate_encoded_ohe_array.tolist()
    job_df['onehot_vec'] = job_encoded_ohe_array.tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. MultiLabel Binarizer Encoding
    """)
    return


@app.cell
def _(MultiLabelBinarizer, candidate_df, np, pickle):
    candidate_df_cols_doc_2_vec = [
        'degree_institutes',
        'preferredJobCategory_department_names',
        'preferredJobCategory_industry_names',
        'degree_names',
        'degree_majors'
    ]

    candidate_df_degree_institutes_list = candidate_df['degree_institutes'].tolist()
    candidate_df_preferred_departments_list = candidate_df['preferredJobCategory_department_names'].tolist()
    candidate_df_preferred_industries_list = candidate_df['preferredJobCategory_industry_names'].tolist()
    candidate_df_degree_names_list = candidate_df['degree_names'].tolist()
    candidate_df_degree_majors_list = candidate_df['degree_majors'].tolist()


    candidate_df_degree_institutes_mlb = MultiLabelBinarizer(sparse_output=False)
    candidate_df_preferred_departments_mlb = MultiLabelBinarizer(sparse_output=False)
    candidate_df_preferred_industries_mlb = MultiLabelBinarizer(sparse_output=False)
    candidate_df_degree_names_mlb = MultiLabelBinarizer(sparse_output=False)
    candidate_df_degree_majors_mlb = MultiLabelBinarizer(sparse_output=False)

    candidate_df_degree_institutes_mlb.fit(candidate_df_degree_institutes_list)
    candidate_df_preferred_departments_mlb.fit(candidate_df_preferred_departments_list)
    candidate_df_preferred_industries_mlb.fit(candidate_df_preferred_industries_list)
    candidate_df_degree_names_mlb.fit(candidate_df_degree_names_list)
    candidate_df_degree_majors_mlb.fit(candidate_df_degree_majors_list)


    degree_institutes_vector = candidate_df_degree_institutes_mlb.transform(candidate_df_degree_institutes_list)
    preferred_departments_vector = candidate_df_preferred_departments_mlb.transform(candidate_df_preferred_departments_list)
    preferred_industries_vector = candidate_df_preferred_industries_mlb.transform(candidate_df_preferred_industries_list)
    degree_names_vector = candidate_df_degree_names_mlb.transform(candidate_df_degree_names_list)
    degree_majors_vector = candidate_df_degree_majors_mlb.transform(candidate_df_degree_majors_list)

    # Save MLB models
    with open('models/candidate_mlb_degree_institutes.pkl', 'wb') as f:
        pickle.dump(candidate_df_degree_institutes_mlb, f)
    with open('models/candidate_mlb_preferred_departments.pkl', 'wb') as f:
        pickle.dump(candidate_df_preferred_departments_mlb, f)
    with open('models/candidate_mlb_preferred_industries.pkl', 'wb') as f:
        pickle.dump(candidate_df_preferred_industries_mlb, f)
    with open('models/candidate_mlb_degree_names.pkl', 'wb') as f:
        pickle.dump(candidate_df_degree_names_mlb, f)
    with open('models/candidate_mlb_degree_majors.pkl', 'wb') as f:
        pickle.dump(candidate_df_degree_majors_mlb, f)
    print("MLB models saved to 'models' folder.")

    candidate_df['degree_institutes_vector'] = list(degree_majors_vector)
    candidate_df['preferred_departments_vector'] = list(preferred_departments_vector)
    candidate_df['preferred_industries_vector'] = list(preferred_industries_vector)
    candidate_df['degree_names_vector'] = list(degree_names_vector)
    candidate_df['degree_majors_vector'] = list(degree_majors_vector)


    candidate_df['mlb_vec'] = candidate_df.apply(
        lambda row: np.concatenate([
            row['degree_institutes_vector'],
            row['preferred_departments_vector'],
            row['preferred_industries_vector'],
            row['degree_names_vector'],
            row['degree_majors_vector']
        ]).tolist(),
        axis=1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Log Normalization
    """)
    return


@app.cell
def _(candidate_df, job_df, log_normalize_series, np):
    candidate_df_cols_log_norm = [
        'present_salary',
        'expected_salary',
        'total_experience',
        'age'
    ]
    job_df_cols_log_norm = [
        'minimum_salary',
        'maximum_salary',
        'age_from',
        'age_to',
        'job_experience',
        'minimum_experience',
        'maximum_experience'
    ]

    # Apply log normalization to candidate_df columns
    candidate_vec_cols = []
    for col_ in candidate_df_cols_log_norm:
        candidate_df[f'{col_}_vec'] = log_normalize_series(candidate_df[col_])
        candidate_vec_cols.append(f'{col_}_vec')

    # Combine into single log_vec column
    candidate_df['log_vec'] = candidate_df[candidate_vec_cols].apply(
        lambda row: np.array(row.values), axis=1
    )

    # Apply log normalization to job_df columns
    job_vec_cols = []
    for col_ in job_df_cols_log_norm:
        job_df[f'{col_}_vec'] = log_normalize_series(job_df[col_])
        job_vec_cols.append(f'{col_}_vec')

    # Combine into single log_vec column
    job_df['log_vec'] = job_df[job_vec_cols].apply(
        lambda row: np.array(row.values), axis=1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Ordinal Encoding
    """)
    return


@app.cell
def _(OrdinalEncoder, candidate_df, job_df, np, pickle):
    # Define ordinal categories in order from lowest to highest
    qualification_categories = [[
        'unidentified', 'psc', 'jsc', 'ssc', 'hsc', 'diploma', 'bachelor', 'mbbs', 'master', 'pdg', 'pdghrm', 'phd', 'phr', 'sphr'
    ]]

    level_categories = [[
        'unidentified', 'student', 'fresher', 'fresher/entry level', 'entry level', 'mid level', 'senior'
    ]]

    # Create encoders
    qualification_encoder = OrdinalEncoder(categories=qualification_categories, handle_unknown='use_encoded_value', unknown_value=-1)
    level_encoder = OrdinalEncoder(categories=level_categories, handle_unknown='use_encoded_value', unknown_value=-1)

    # Fit and transform candidate ordinal features
    candidate_df['level_name_lower'] = candidate_df['level_name'].str.lower()
    candidate_df['qualification_name_lower'] = candidate_df['qualification_name'].str.lower()

    level_encoder.fit(candidate_df[['level_name_lower']])
    qualification_encoder.fit(candidate_df[['qualification_name_lower']])

    candidate_df['level_name_vec'] = level_encoder.transform(candidate_df[['level_name_lower']]).flatten()
    candidate_df['qualification_name_vec'] = qualification_encoder.transform(candidate_df[['qualification_name_lower']]).flatten()

    # Transform job ordinal features using the same encoders
    job_df['job_level_name_lower'] = job_df['job_level_name'].str.lower()
    job_df['job_qualification_name_lower'] = job_df['job_qualification_name'].str.lower()
    job_df['qualification_prefer_name_lower'] = job_df['qualification_prefer_name'].str.lower()

    job_df['job_level_name_vec'] = level_encoder.transform(job_df[['job_level_name_lower']]).flatten()
    job_df['job_qualification_name_vec'] = qualification_encoder.transform(job_df[['job_qualification_name_lower']]).flatten()
    job_df['qualification_prefer_name_vec'] = qualification_encoder.transform(job_df[['qualification_prefer_name_lower']]).flatten()

    # Save the encoders
    with open('models/qualification_ordinal_encoder.pkl', 'wb') as f:
        pickle.dump(qualification_encoder, f)
    with open('models/level_ordinal_encoder.pkl', 'wb') as f:
        pickle.dump(level_encoder, f)
    print("Ordinal encoders saved to 'models' folder.")

    # Combine into single vectors
    candidate_df['ord_vec'] = candidate_df[['level_name_vec', 'qualification_name_vec']].apply(lambda x: np.array(x), axis=1)
    job_df['ord_vec'] = job_df[['job_level_name_vec', 'job_qualification_name_vec', 'qualification_prefer_name_vec']].apply(lambda x: np.array(x), axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. DLEM Embedding
    """)
    return


@app.cell
def _(
    candidate_df,
    device_dlem,
    dlem_model,
    get_dlem_embedding,
    job_df,
    w2v_model,
):
    job_df['dlem_vec'] = job_df['job_text_data'].progress_apply(
        lambda x: get_dlem_embedding(x, w2v_model, dlem_model, device_dlem)
    )

    candidate_df['dlem_vec'] = candidate_df['candidate_latest_resume_text'].progress_apply(
        lambda x: get_dlem_embedding(x, w2v_model, dlem_model, device_dlem)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. RL Embedding
    """)
    return


@app.cell
def _(
    candidate_df,
    get_single_node_vector,
    get_weighted_node_cluster_vector,
    job_df,
    np,
    rl_model,
):
    candidate_df['skills_names_vec'] = candidate_df.progress_apply(
        lambda row: get_weighted_node_cluster_vector(
            rl_model=rl_model,
            node_names=row['skills_names'],
            node_weights=row['skills_year_of_experiences'],
            node_type='skill'
        ),
        axis=1
    )

    candidate_df['candidate_experienc_vec'] = candidate_df.progress_apply(
        lambda row: get_weighted_node_cluster_vector(
            rl_model=rl_model,
            node_names=row['candidate_experience_roles'],
            node_weights=row['candidate_experience_role_duration'],
            node_type='job'
        ),
        axis=1
    )

    candidate_df['profession_vec'] = candidate_df['profession'].progress_apply(
        lambda x: get_single_node_vector(
            rl_model=rl_model,
            node_name=x,
            node_type='job'
        )
    )

    job_df['job_skill_name_vec'] = job_df.progress_apply(
        lambda row: get_weighted_node_cluster_vector(
            rl_model=rl_model,
            node_names=row['job_skill_name'],
            node_weights=row['job_skill_experience'],
            node_type='skill'
        ),
        axis=1
    )


    job_df['job_title_vec'] = job_df['job_title'].progress_apply(
        lambda x: get_single_node_vector(
            rl_model=rl_model,
            node_name=x,
            node_type='job'
        )
    )


    # Combine into single rl_vec column
    candidate_df['rl_vec'] = candidate_df.apply(
        lambda row: np.concatenate([
            row['skills_names_vec'],
            row['candidate_experienc_vec'],
            row['profession_vec']
        ]),
        axis=1
    )


    job_df['rl_vec'] = job_df.apply(
        lambda row: np.concatenate([
            row['job_skill_name_vec'],
            row['job_title_vec']
        ]),
        axis=1
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Feature Vector Assembly and Saving
    """)
    return


@app.cell
def _(candidate_df, job_df):
    candidate_df['candidate_feature_vector'] = candidate_df.apply(
        lambda row: row['onehot_vec'] + row['mlb_vec'] + row['log_vec'].tolist() + row['ord_vec'].tolist() + row['dlem_vec'] + row['rl_vec'].tolist(),
        axis=1
    )

    job_df['job_feature_vector'] = job_df.apply(
        lambda row: row['onehot_vec'] + row['log_vec'].tolist() + row['ord_vec'].tolist() + row['dlem_vec'] + row['rl_vec'].tolist(),
        axis=1
    )
    return


@app.cell
def _(candidate_df, job_df):
    candidate_df[['id', 'candidate_feature_vector']].to_csv('../datasets/processed_dataset/candidate_feature_vectors.csv', index=False)
    job_df[['post_id', 'job_feature_vector']].to_csv('../datasets/processed_dataset/job_feature_vectors.csv', index=False)
    print("Feature vectors saved to 'processed_dataset' folder.")
    return


if __name__ == "__main__":
    app.run()
