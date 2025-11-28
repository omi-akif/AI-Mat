import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Wrangling and Dataset Generation

    This notebook prepares various datasets for different models and tasks:
    1.  **Word2Vec**: Combines job text and resume text for training word embeddings.
    2.  **Two Tower Model**: Creates positive and negative samples for training the retrieval model.
    3.  **DLEM Model**: Prepares data for the Deep Learning Entity Matching model.
    4.  **Graph Models**: Generates transition data for Job-Job, Job-Skill, and Skill-Skill graphs.
    """)
    return


@app.cell
def _():
    import ast
    import csv
    import json
    import pandas as pd
    from datetime import datetime
    from itertools import combinations
    from difflib import SequenceMatcher
    import marimo as mo
    return SequenceMatcher, ast, combinations, json, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Cleaned Data
    """)
    return


@app.cell
def _(pd):
    job_df_cleaned = pd.read_csv('../datasets/cleaned_datasets/job_data_df_clean.csv')
    candidate_df_cleaned = pd.read_csv('../datasets/cleaned_datasets/candidate_data_df_clean.csv')
    return candidate_df_cleaned, job_df_cleaned


@app.cell
def _(candidate_df_cleaned, job_df_cleaned):
    print("Job Data Shape:", job_df_cleaned.shape)
    print("Candidate Data Shape:", candidate_df_cleaned.shape)
    return


@app.cell
def _(ast, candidate_df_cleaned):
    # Parse stringified lists
    candidate_df_cleaned['preferredJobCategory_industry_names'] = candidate_df_cleaned['preferredJobCategory_industry_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])

    candidate_df_cleaned['preferredJobCategory_department_names'] = candidate_df_cleaned['preferredJobCategory_department_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Word2Vec Data Preparation
    """)
    return


@app.cell
def _(candidate_df_cleaned, job_df_cleaned, pd):
    # Word2Vec data production

    job_df_cleaned['job_text_data'] = job_df_cleaned['job_requirement'] + ' ' + job_df_cleaned['job_description']

    combined_texts = pd.concat([
        job_df_cleaned['job_text_data'].dropna(),
        candidate_df_cleaned[candidate_df_cleaned['candidate_latest_resume_text'] != 'undefined']['candidate_latest_resume_text'].reset_index(drop=True).dropna()
    ], ignore_index=True).tolist()

    with open('../datasets/helping_datasets/job_candidate_text_data_word2vec.txt', 'w') as f:
        for text in combined_texts:
            f.write(text + '\n')
    print("Word2Vec text data saved.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Two Tower Model Data Preparation (Negative Sampling)
    """)
    return


@app.cell
def _(SequenceMatcher, candidate_df_cleaned, job_df_cleaned, pd):
    # Sample for Two Tower model

    random_state_job = 42
    random_state_candidate = 42
    job_sample_size = 1000
    candidate_sample_size = 2000

    candidates_with_text = candidate_df_cleaned[candidate_df_cleaned['candidate_latest_resume_text'] != 'undefined']
    candidates_undefined = candidate_df_cleaned[candidate_df_cleaned['candidate_latest_resume_text'] == 'undefined']

    sample_size = candidate_sample_size
    candidates_sample = pd.concat([
        candidates_with_text.sample(n=sample_size//2, random_state=random_state_candidate),
        candidates_undefined.sample(n=sample_size//2, random_state=random_state_candidate)
    ], ignore_index=True)

    # Sample jobs
    job_sample = job_df_cleaned.sample(n=job_sample_size, random_state=random_state_job)

    # Get unique job industries and departments
    job_industries = set(job_sample['industry_name'].unique())
    job_departments = set(job_sample['department_name'].unique())

    # Filter candidates: keep only those whose preferred industries/departments match job sample

    def has_matching_preference(row):
        candidate_industries = set(row['preferredJobCategory_industry_names']) if isinstance(row['preferredJobCategory_industry_names'], list) else set()
        candidate_departments = set(row['preferredJobCategory_department_names']) if isinstance(row['preferredJobCategory_department_names'], list) else set()

        candidate_industries.discard('unidentified')
        candidate_departments.discard('unidentified')

        if not candidate_industries and not candidate_departments:
            return True

        def fuzzy_match(candidate_set, job_value):
            return any(SequenceMatcher(None, cand.lower(), job_value.lower()).ratio() > 0.7 for cand in candidate_set)

        has_industry_match = fuzzy_match(candidate_industries, job_sample['industry_name'].iloc[0])
        has_department_match = fuzzy_match(candidate_departments, job_sample['department_name'].iloc[0])

        return has_industry_match or has_department_match


    # Create combinations
    combined_df = job_sample.merge(candidates_sample, how='cross')
    combined_df.to_csv('../datasets/helping_datasets/job_candidate_annotation_data_two_tower_v2.csv')
    print("Two Tower annotation data saved.")
    return (combined_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. DLEM Model Data Preparation
    """)
    return


@app.cell
def _(combined_df):
    # Sample for DLEM model
    combined_df[['job_text_data', 'candidate_latest_resume_text']].to_csv('../datasets/helping_datasets/job_candidate_annotation_data_dlem.csv', index=False)
    print("DLEM annotation data saved.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Graph Models Data Preparation
    """)
    return


@app.cell
def _(ast, candidate_df_cleaned, json, pd):
    # Sample for Job - Job Transition Graph
    def create_role_transitions(row):
        roles = row['candidate_experience_roles']
        start_dates = row['candidate_experience_start_dates']
        end_dates = row['candidate_experience_end_dates']

        # Try different parsing methods
        def parse_data(data):
            if isinstance(data, list):
                return data
            try:
                return json.loads(data)
            except:
                try:
                    return ast.literal_eval(data)
                except:
                    return None

        roles = parse_data(roles)
        start_dates = parse_data(start_dates)
        end_dates = parse_data(end_dates)

        # Check if parsing was successful
        if not roles or not start_dates or not end_dates or len(roles) == 0:
            return []

        # Create tuples of (role, start_date, end_date)
        role_data = list(zip(roles, start_dates, end_dates))

        # Sort by start_date chronologically
        role_data.sort(key=lambda x: x[1])

        transitions = []

        # Create transitions based on chronological order
        for i in range(len(role_data)):
            from_role = role_data[i-1][0] if i > 0 else 'undetermined'
            to_role = role_data[i][0]

            transitions.append({
                'from_role': from_role,
                'to_role': to_role
            })

        return transitions

    all_transitions = []
    for _, row in candidate_df_cleaned.iterrows():
        transitions = create_role_transitions(row)
        all_transitions.extend(transitions)

    job_to_job_transition_df = pd.DataFrame(all_transitions)

    job_to_job_transition_df.fillna('unidentified',inplace=True)

    job_to_job_transition_df = job_to_job_transition_df.astype(str)

    job_to_job_transition_df.to_csv('../datasets/helping_datasets/job_to_job_transition_inf_lear.csv', index=False)
    print("Job-Job transition data saved.")
    return


@app.cell
def _(ast, job_df_cleaned):
    # Sample for Job Title to Job Skill transition graph

    # Step 1: Convert string list → actual Python list
    job_df_cleaned['job_skill_name'] = job_df_cleaned['job_skill_name'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Step 2: Explode into individual skills
    job_to_skill_relation_df = (
        job_df_cleaned[['job_title', 'job_skill_name']]
        .explode('job_skill_name')
        .reset_index(drop=True)
    )

    # Step 3: Rename column
    job_to_skill_relation_df.columns = ['job_title', 'job_skill']


    job_to_skill_relation_df.fillna('unidentified', inplace=True)

    job_to_skill_relation_df = job_to_skill_relation_df.astype(str)

    # Step 4: Save to CSV
    job_to_skill_relation_df.to_csv(
        '../datasets/helping_datasets/job_to_skill_relation_inf_lear.csv',
        index=False
    )
    print("Job-Skill relation data saved.")
    return


@app.cell
def _(ast, candidate_df_cleaned, combinations, pd):
    # Sample for Job Title to Job Skill transition graph

    def generate_skill_pairs(df, col='skill_names'):
        all_pairs = []

        for skills in df[col]:
            # Convert string representation to list if needed
            if isinstance(skills, str):
                skills = ast.literal_eval(skills)

            # Remove duplicates and sort for consistent pairing
            skills = sorted(set(skills))

            # Create all unique pairs
            pairs = list(combinations(skills, 2))

            # Store results
            for s1, s2 in pairs:
                all_pairs.append({"skill_1": s1, "skill_2": s2})

        return pd.DataFrame(all_pairs)

    # Example usage:
    skill_to_skill_relation_df = generate_skill_pairs(candidate_df_cleaned, "skills_names")

    skill_to_skill_relation_df.fillna('unidentified', inplace=True)

    skill_to_skill_relation_df = skill_to_skill_relation_df.astype(str)


    # Save to CSV
    skill_to_skill_relation_df.to_csv("../datasets/helping_datasets/skill_to_skill_relation_inf_lear.csv", index=False)
    print("Skill-Skill relation data saved.")
    return


if __name__ == "__main__":
    app.run()
