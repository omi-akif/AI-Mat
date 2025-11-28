import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import ast
    return ast, pd


@app.cell
def _(pd):
    candidate_embedding_df = pd.read_csv('processed_dataset/candidate_embedding_data.csv')
    job_embedding_df = pd.read_csv('processed_dataset/job_embedding_data.csv')

    job_df_unclean = pd.read_csv('uncleaned_datasets/ai_matching_job_with_candidate_data_latest.csv')
    candidate_df_unclean = pd.read_csv('uncleaned_datasets/candidate_opensearch_export.csv')

    return (
        candidate_df_unclean,
        candidate_embedding_df,
        job_df_unclean,
        job_embedding_df,
    )


@app.cell
def _(ast):
    def fix_col(col):
        return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return (fix_col,)


@app.cell
def _(job_df_unclean):
    job_df_unclean_ = job_df_unclean[['post_id', 'job_title', 'job_description', 'job_experience',
           'minimum_experience', 'maximum_experience', 'minimum_salary',
           'maximum_salary', 'negotiable', 'age_from', 'age_to', 'job_requirement',
           'job_gender', 'industry_name', 'department_name', 'position_name',
           'job_district_name', 'job_type_name', 'job_level_name',
           'job_qualification_name', 'qualification_prefer_name',
           'salary_currency', 'job_salary_type', 'job_skill_name',
           'job_skill_experience']]

    job_df_unclean_
    return (job_df_unclean_,)


@app.cell
def _(candidate_df_unclean):
    candidate_df_unclean
    return


@app.cell
def _(candidate_embedding_df, fix_col, job_embedding_df):
    candidate_embedding_df['candidate_embedding'] = fix_col(candidate_embedding_df['candidate_embedding'])
    job_embedding_df['job_embedding'] = fix_col(job_embedding_df['job_embedding'])
    return


@app.cell
def _(
    candidate_df_unclean,
    candidate_embedding_df,
    job_df_unclean_,
    job_embedding_df,
):
    job_df_unclean_emb = job_df_unclean_.merge(job_embedding_df[['post_id', 'job_embedding']], on='post_id', how='left')
    candidate_df_unclean_emb = candidate_df_unclean.merge(candidate_embedding_df[['id', 'candidate_embedding']], on='id', how='left')
    return candidate_df_unclean_emb, job_df_unclean_emb


@app.cell
def _(candidate_df_unclean_emb, job_df_unclean_emb):
    # job_df_unclean_emb.to_csv('final_dataset/job_embedding_data.csv')
    # candidate_df_unclean_emb.to_csv('final_dataset/candidate_embedding_data.csv')

    job_df_unclean_emb.to_parquet('final_dataset/job_embedding_data.parquet', index=False)
    candidate_df_unclean_emb.to_parquet('final_dataset/candidate_embedding_data.parquet', index=False)

    return


if __name__ == "__main__":
    app.run()
