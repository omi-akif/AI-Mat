import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Processing and Merging

    This notebook merges the generated embeddings with the original job and candidate data.
    It reads the processed embedding files and the uncleaned source datasets, then merges them based on IDs.
    The final merged datasets are saved as Parquet files for efficient storage and retrieval.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import ast
    import os
    return mo, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Datasets
    """)
    return


@app.cell
def _(pd):
    # Load processed embeddings
    candidate_embedding_df = pd.read_csv('../datasets/processed_dataset/candidate_embedding_data.csv')
    job_embedding_df = pd.read_csv('../datasets/processed_dataset/job_embedding_data.csv')

    # Load uncleaned source data
    # Fixed paths to point to uncleaned_datasets
    job_df_unclean = pd.read_csv('../datasets/uncleaned_datasets/ai_matching_job_with_candidate_data_latest.csv')
    candidate_df_unclean = pd.read_csv('../datasets/uncleaned_datasets/candidate_opensearch_export.csv')

    print("Datasets loaded successfully.")
    return (
        candidate_df_unclean,
        candidate_embedding_df,
        job_df_unclean,
        job_embedding_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Preprocess and Fix Columns
    """)
    return


@app.cell
def _():
    # def fix_col(col):
    #     """Parse string representation of lists/dicts into actual objects."""
    #     return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return


@app.cell
def _(job_df_unclean):
    # Select relevant columns from job data
    job_df_unclean_ = job_df_unclean[['post_id', 'job_title', 'job_description', 'job_experience',
           'minimum_experience', 'maximum_experience', 'minimum_salary',
           'maximum_salary', 'negotiable', 'age_from', 'age_to', 'job_requirement',
           'job_gender', 'industry_name', 'department_name', 'position_name',
           'job_district_name', 'job_type_name', 'job_level_name',
           'job_qualification_name', 'qualification_prefer_name',
           'salary_currency', 'job_salary_type', 'job_skill_name',
           'job_skill_experience']]

    # job_df_unclean_
    return (job_df_unclean_,)


@app.cell
def _():
    # # Fix embedding columns
    # candidate_embedding_df['candidate_embedding'] = fix_col(candidate_embedding_df['candidate_embedding'])
    # job_embedding_df['job_embedding'] = fix_col(job_embedding_df['job_embedding'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Merge and Save Data
    """)
    return


@app.cell
def _(
    candidate_df_unclean,
    candidate_embedding_df,
    job_df_unclean_,
    job_embedding_df,
):
    # Merge embeddings with original data
    job_df_unclean_emb = job_df_unclean_.merge(job_embedding_df[['post_id', 'job_embedding']], on='post_id', how='left')
    candidate_df_unclean_emb = candidate_df_unclean.merge(candidate_embedding_df[['id', 'candidate_embedding']], on='id', how='left')

    print(f"Merged Job Data Shape: {job_df_unclean_emb.shape}")
    print(f"Merged Candidate Data Shape: {candidate_df_unclean_emb.shape}")
    return candidate_df_unclean_emb, job_df_unclean_emb


@app.cell
def _(candidate_df_unclean_emb, job_df_unclean_emb, os):
    output_dir = '../datasets/final_dataset'
    os.makedirs(output_dir, exist_ok=True)

    # Save as Parquet
    job_output_path = os.path.join(output_dir, 'job_embedding_data.parquet')
    candidate_output_path = os.path.join(output_dir, 'candidate_embedding_data.parquet')

    job_df_unclean_emb.to_parquet(job_output_path, index=False)
    candidate_df_unclean_emb.to_parquet(candidate_output_path, index=False)

    print(f"Saved job data to {job_output_path}")
    print(f"Saved candidate data to {candidate_output_path}")
    return


if __name__ == "__main__":
    app.run()
