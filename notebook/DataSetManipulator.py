import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    candidate_df = pd.read_csv('../datasets/uncleaned_datasets/ai_matching_job_with_candidate_data_latest.csv')
    job_df = pd.read_csv('../datasets/uncleaned_datasets/candidate_opensearch_export.csv')
    return (job_df,)


@app.cell
def _(job_df):
    job_df
    return


if __name__ == "__main__":
    app.run()
