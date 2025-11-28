import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM Annotator for Two-Tower Model

    This notebook uses an LLM (via `flow_judge`) to annotate the relevance of candidate information to job post information for the Two-Tower model.
    It performs the following steps:
    1.  **Load Data**: Reads the Two-Tower annotation dataset.
    2.  **Define Metric**: Sets up the evaluation criteria and rubric for relevance scoring.
    3.  **Data Preparation**: Constructs string representations of job and candidate information.
    4.  **Batch Processing**: Splits the data into batches and evaluates them using the LLM.
    5.  **Save Annotations**: Saves the generated annotations to CSV files.
    """)
    return


@app.cell
def _():
    import torch
    from flow_judge.metrics import CustomMetric, RubricItem
    from flow_judge.flow_judge import EvalInput, FlowJudge
    from flow_judge.models import Vllm
    import json
    import pandas as pd
    import marimo as mo
    import os
    return CustomMetric, EvalInput, FlowJudge, RubricItem, Vllm, mo, os, pd


@app.cell
def _(RubricItem):
    evaluation_criteria = """Determine if the candidate's information matches the job post information. Score 1 if the resume is relevant and aligns with the job post, otherwise 0."""

    rubric = [
        RubricItem(
            score=0,
            description="The candidate information is largely irrelevant to the job post information. It lacks connection in skills, experience, education, or field. Any overlap appears minimal or generic."
        ),
        RubricItem(
            score=1,
            description="The candidate information demonstrates reasonable relevance to the job information. It includes some matching skills, experience, or educational elements that indicate the candidate could potentially fit or be trained for the role."
        ),
    ]
    return evaluation_criteria, rubric


@app.cell
def _(CustomMetric, FlowJudge, Vllm, evaluation_criteria, rubric):
    # We need to define the required inputs and output for the metric
    required_inputs = ["job_post_information"]
    required_output = "candidate_information"

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
        gpu_memory_utilization=0.98,
        max_model_len=6000
    )

    judge = FlowJudge(metric=sub_query_coverage, model=model)
    return


@app.cell
def _(pd):
    # Load the dataset
    ai_matching_df_cleaned = pd.read_csv('../datasets/helping_datasets/job_candidate_annotation_data_two_tower_v2.csv')
    return (ai_matching_df_cleaned,)


@app.cell
def _(ai_matching_df_cleaned):
    ai_matching_df_cleaned_  = ai_matching_df_cleaned.astype(str)

    ai_matching_df_cleaned_['job_post_information'] = (
        'job_title: ' + ai_matching_df_cleaned_['job_title'] + ', ' +
        'job_description: ' + ai_matching_df_cleaned_['job_description'] + ', ' +
        'job_experience: ' + ai_matching_df_cleaned_['job_experience'] + ', ' +
        'minimum_experience: ' + ai_matching_df_cleaned_['minimum_experience'].astype(str) + ', ' +
        'maximum_experience: ' + ai_matching_df_cleaned_['maximum_experience'].astype(str) + ', ' +
        'minimum_salary: ' + ai_matching_df_cleaned_['minimum_salary'].astype(str) + ', ' +
        'maximum_salary: ' + ai_matching_df_cleaned_['maximum_salary'].astype(str) + ', ' +
        'negotiable: ' + ai_matching_df_cleaned_['negotiable'].astype(str) + ', ' +
        'age from: ' + ai_matching_df_cleaned_['age_from'].astype(str) + ', ' +
        'age to: ' + ai_matching_df_cleaned_['age_to'].astype(str) + ', ' +
        'job_requirement: ' + ai_matching_df_cleaned_['job_requirement'] + ', ' +
        'job gender: ' + ai_matching_df_cleaned_['job_gender'] + ', ' +
        'industry name: ' + ai_matching_df_cleaned_['industry_name'] + ', ' +
        'department name: ' + ai_matching_df_cleaned_['department_name'] + ', ' +
        'position name: ' + ai_matching_df_cleaned_['position_name'] + ', ' +
        'job_district name: ' + ai_matching_df_cleaned_['job_district_name'] + ', ' +
        'job type name: ' + ai_matching_df_cleaned_['job_type_name'] + ', ' +
        'job level name: ' + ai_matching_df_cleaned_['job_level_name'] + ', ' +
        'job qualification name: ' + ai_matching_df_cleaned_['job_qualification_name'] + ', ' +
        'qualification prefer name: ' + ai_matching_df_cleaned_['qualification_prefer_name'] + ', ' +
        'salary currency: ' + ai_matching_df_cleaned_['salary_currency'] + ', ' +
        'job salary type: ' + ai_matching_df_cleaned_['job_salary_type'] + ', ' +
        'job skill name: ' + ai_matching_df_cleaned_['job_skill_name'].astype(str) + ', ' +
        'job skill experience: ' + ai_matching_df_cleaned_['job_skill_experience'].astype(str) + ', ' +
        'job text data: ' + ai_matching_df_cleaned_['job_text_data']
    )


    ai_matching_df_cleaned_['candidate_information'] = (
        'expected salary: ' + ai_matching_df_cleaned_['expected_salary'].astype(str) + ', ' +
        'gender: ' + ai_matching_df_cleaned_['gender'] + ', ' +
        'id: ' + ai_matching_df_cleaned_['id'].astype(str) + ', ' +
        'martial status: ' + ai_matching_df_cleaned_['martial_status'] + ', ' +
        'present salary: ' + ai_matching_df_cleaned_['present_salary'].astype(str) + ', ' +
        'profession: ' + ai_matching_df_cleaned_['profession'] + ', ' +
        'searching for job status: ' + ai_matching_df_cleaned_['searching_for_job_status'] + ', ' +
        'total experience: ' + ai_matching_df_cleaned_['total_experience'].astype(str) + ', ' +
        'district name: ' + ai_matching_df_cleaned_['district_name'] + ', ' +
        'salary currency name: ' + ai_matching_df_cleaned_['salary_currency_name'] + ', ' +
        'salary type name: ' + ai_matching_df_cleaned_['salary_type_name'] + ', ' +
        'level name: ' + ai_matching_df_cleaned_['level_name'] + ', ' +
        'qualification name: ' + ai_matching_df_cleaned_['qualification_name'] + ', ' +
        'degree institutes: ' + ai_matching_df_cleaned_['degree_institutes'].astype(str) + ', ' +
        'skills names: ' + ai_matching_df_cleaned_['skills_names'].astype(str) + ', ' +
        'skills year of experiences: ' + ai_matching_df_cleaned_['skills_year_of_experiences'].astype(str) + ', ' +
        'candidate experience roles: ' + ai_matching_df_cleaned_['candidate_experience_roles'].astype(str) + ', ' +
        'candidate experience start dates: ' + ai_matching_df_cleaned_['candidate_experience_start_dates'].astype(str) + ', ' +
        'candidate experience end dates: ' + ai_matching_df_cleaned_['candidate_experience_end_dates'].astype(str) + ', ' +
        'candidate type: ' + ai_matching_df_cleaned_['candidate_type'] + ', ' +
        'preferredJobCategory department names: ' + ai_matching_df_cleaned_['preferredJobCategory_department_names'].astype(str) + ', ' +
        'preferredJobCategory industry names: ' + ai_matching_df_cleaned_['preferredJobCategory_industry_names'].astype(str) + ', ' +
        'degree names: ' + ai_matching_df_cleaned_['degree_names'].astype(str) + ', ' +
        'degree majors: ' + ai_matching_df_cleaned_['degree_majors'].astype(str) + ', ' +
        'candidate experience role duration: ' + ai_matching_df_cleaned_['candidate_experience_role_duration'].astype(str) + ', ' +
        'age: ' + ai_matching_df_cleaned_['age'].astype(str) + ', ' +
        'candidate latest resume text: ' + ai_matching_df_cleaned_['candidate_latest_resume_text']
    )
    return (ai_matching_df_cleaned_,)


@app.cell
def _(ai_matching_df_cleaned_):
    ai_matching_df_cleaned_[['post_id','job_post_information', 'id', 'candidate_information']].to_csv('../datasets/processed_dataset/two_tower_annotations.csv', index=False)
    return


@app.cell
def _(pd):
    ai_matching_df_cleaned_read = pd.read_csv('../datasets/processed_dataset/two_tower_annotations.csv')
    return (ai_matching_df_cleaned_read,)


@app.cell
def _(ai_matching_df_cleaned_read):
    ai_matching_df_cleaned_sample = ai_matching_df_cleaned_read.sample(n=10000, random_state=42)
    return (ai_matching_df_cleaned_sample,)


@app.cell
def _(ai_matching_df_cleaned_sample):
    # === 2. Keep post_id and id in data ===
    data = (
        ai_matching_df_cleaned_sample[
            ['post_id', 'id', 'job_post_information', 'candidate_information']
        ].to_dict(orient='records')
    )

    # === 3. Create batches ===
    batch_size = 1000
    data_batches = [
        data[i:i + batch_size] for i in range(0, len(data), batch_size)
    ]
    return (data_batches,)


@app.cell
def _(EvalInput, data_batches, os):
    # Directory to save annotations
    output_dir = '../datasets/annotation_datasets/two_tower'
    os.makedirs(output_dir, exist_ok=True)

    # Process batches
    for i, batch in enumerate(data_batches):
        print(f"Processing batch {i}...")

        inputs_batch = [[{"job_post_information": sample["job_post_information"]}] for sample in batch]
        outputs_batch = [{"candidate_information": sample["candidate_information"]} for sample in batch]

        eval_inputs_batch = [EvalInput(inputs=inp, output=out) for inp, out in zip(inputs_batch, outputs_batch)]

        # Evaluate
        # Uncomment to run
        # results = judge.batch_evaluate(eval_inputs_batch, save_results=False)

        # Mock results for demonstration
        # results = []

        # if results:
        #     results_dat = [
        #         {
        #             "post_id": sample["post_id"],
        #             "id": sample["id"],
        #             "job_post_information": sample["job_post_information"],
        #             "candidate_information": sample["candidate_information"],
        #             "Feedback": result.feedback,
        #             "Score": result.score,
        #         }
        #         for sample, result in zip(batch, results)
        #     ]
        #     df_results = pd.DataFrame(results_dat)
        #     output_path = f'{output_dir}/annotations_B{i:02d}_{len(batch)}.csv'
        #     df_results.to_csv(output_path, index=False)
        #     print(f"Saved results to {output_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Analysis of Annotations
    """)
    return


@app.cell
def _(os, pd):
    output_dir = '../datasets/annotation_datasets/two_tower'

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
    return


if __name__ == "__main__":
    app.run()
