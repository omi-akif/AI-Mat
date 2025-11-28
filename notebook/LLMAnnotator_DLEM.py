import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM Annotator for Resume-Job Matching (DLEM)

    This notebook uses an LLM (via `flow_judge`) to annotate the relevance of candidate resumes to job posts.
    It performs the following steps:
    1.  **Load Data**: Reads the cleaned AI matching dataset.
    2.  **Define Metric**: Sets up the evaluation criteria and rubric for relevance scoring.
    3.  **Batch Processing**: Splits the data into batches and evaluates them using the LLM.
    4.  **Save Annotations**: Saves the generated annotations to CSV files.
    5.  **Analyze Results**: (Optional) Reads back the annotations for analysis.
    """)
    return


@app.cell
def _():
    import torch
    from flow_judge.metrics import CustomMetric, RubricItem
    from IPython.display import Markdown, display
    from flow_judge.flow_judge import EvalInput, FlowJudge
    from flow_judge.models import Vllm
    import json
    import pandas as pd
    import marimo as mo
    import os
    return CustomMetric, EvalInput, FlowJudge, RubricItem, Vllm, mo, os, pd


@app.cell
def _(RubricItem):
    evaluation_criteria = """Determine if the candidate's resume matches the job post. Score 1 if the resume is relevant and aligns with the job post, otherwise 0."""

    rubric = [
        RubricItem(
            score=0,
            description="The resume is largely irrelevant to the job post. It lacks connection in skills, experience, education, or field. Any overlap appears minimal or generic."
        ),
        RubricItem(
            score=1,
            description="The resume demonstrates reasonable relevance to the job post. It includes some matching skills, experience, or educational elements that indicate the candidate could potentially fit or be trained for the role."
        ),
    ]
    return evaluation_criteria, rubric


@app.cell
def _(CustomMetric, FlowJudge, Vllm, evaluation_criteria, rubric):
    # We need to define the required inputs and output for the metric
    required_inputs = ["job_post"]
    required_output = "resume"

    # Create the metric
    sub_query_coverage = CustomMetric(
        name="ai-matching",
        criteria=evaluation_criteria,
        rubric=rubric,
        required_inputs=required_inputs,
        required_output=required_output
    )

    model = Vllm(
        quantized=True,
        gpu_memory_utilization=0.97,
        max_model_len=7500
    )

    judge = FlowJudge(metric=sub_query_coverage, model=model)
    return


@app.cell
def _(pd):
    # Load the dataset
    # Updated path to point to datasets directory
    ai_matching_df_cleaned = pd.read_csv('../datasets/ai_matching_df_cleaned.csv')
    return (ai_matching_df_cleaned,)


@app.cell
def _(ai_matching_df_cleaned):
    ai_matching_df_cleaned[ai_matching_df_cleaned.select_dtypes(include="object").columns] = (
        ai_matching_df_cleaned.select_dtypes(include="object").fillna('unidentified')
    )
    ai_matching_df_cleaned[ai_matching_df_cleaned.select_dtypes(include="number").columns] = (
        ai_matching_df_cleaned.select_dtypes(include="number").fillna(0)
    )

    ai_matching_df_cleaned['job_string'] = ai_matching_df_cleaned['job_description'] + ' ' + ai_matching_df_cleaned['job_requirement']

    filtered_df = ai_matching_df_cleaned[
        (ai_matching_df_cleaned['job_description'] != 'unidentified') &
        (ai_matching_df_cleaned['job_requirement'] != 'unidentified') &
        (ai_matching_df_cleaned['candidate_latest_resume_text'] != 'unidentified')
    ]

    data = (
        filtered_df[['job_string', 'candidate_latest_resume_text']]
        .rename(columns={
            'job_string': 'job_post',
            'candidate_latest_resume_text': 'resume'
        })
        .to_dict(orient='records')
    )
    return data, filtered_df


@app.cell
def _(data):
    batch_size = 1000
    data_batches = [
        data[i:i + batch_size] for i in range(0, len(data), batch_size)
    ]
    return (data_batches,)


@app.cell
def _(EvalInput, data_batches, os):
    # Directory to save annotations
    output_dir = '../datasets/annotation_datasets/resume_candidate'
    os.makedirs(output_dir, exist_ok=True)

    # Process batches
    # Note: This loop processes all batches. In the original code, specific batches were processed manually.
    # You can adjust the range or indices if needed.

    for i, batch in enumerate(data_batches):
        print(f"Processing batch {i}...")

        inputs_batch = [[{"job_post": sample["job_post"]}] for sample in batch]
        outputs_batch = [{"resume": sample["resume"]} for sample in batch]

        eval_inputs_batch = [EvalInput(inputs=inp, output=out) for inp, out in zip(inputs_batch, outputs_batch)]

        # Evaluate
        # Uncomment the following line to run evaluation (it might take time and resources)
        # results = judge.batch_evaluate(eval_inputs_batch, save_results=False)

        # Mock results for demonstration if not running
        # results = [] 

        # if results:
        #     results_dat = [
        #         {
        #             "Sample": idx + 1,
        #             "Feedback": result.feedback,
        #             "Score": result.score
        #         }
        #         for idx, result in enumerate(results)
        #     ]
        #     df_results = pd.DataFrame(results_dat)
        #     output_path = f'{output_dir}/annotations_B{i:02d}_{len(batch)}.csv'
        #     df_results.to_csv(output_path, index=False)
        #     print(f"Saved results to {output_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analysis of Annotations
    """)
    return


@app.cell
def _(os, pd):
    output_dir = '../datasets/annotation_datasets/resume_candidate'

    # List all annotation files
    if os.path.exists(output_dir):
        annotation_files = sorted([f for f in os.listdir(output_dir) if f.startswith('annotations_') and f.endswith('.csv')])

        if annotation_files:
            dfs = []
            for file in annotation_files:
                df = pd.read_csv(os.path.join(output_dir, file))
                dfs.append(df)

            if dfs:
                df_all_annotations = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(df_all_annotations)} annotations.")

                # Save combined annotations
                df_all_annotations.to_csv(os.path.join(output_dir, 'all_annotations.csv'), index=False)
            else:
                df_all_annotations = pd.DataFrame()
        else:
            print("No annotation files found.")
            df_all_annotations = pd.DataFrame()
    else:
        print(f"Directory {output_dir} does not exist.")
        df_all_annotations = pd.DataFrame()
    return (df_all_annotations,)


@app.cell
def _(df_all_annotations, filtered_df, os, pd):
    if not df_all_annotations.empty and not filtered_df.empty:
        # Assuming filtered_df aligns with df_all_annotations index-wise or we need a common key.
        # The original code just concatenated axis=1, implying row alignment.
        # We should be careful if batches were skipped or reordered.
        # For now, we follow the original logic but check lengths.

        if len(df_all_annotations) == len(filtered_df):
             df_final = pd.concat([df_all_annotations.reset_index(drop=True), filtered_df.reset_index(drop=True)], axis=1)
             output_dir = '../datasets/annotation_datasets/resume_candidate'
             df_final.to_csv(os.path.join(output_dir, 'all_annotations_with_data.csv'), index=False)
             print("Saved combined annotations with data.")
        else:
            print(f"Shape mismatch: Annotations {len(df_all_annotations)} vs Data {len(filtered_df)}")
    return


if __name__ == "__main__":
    app.run()
