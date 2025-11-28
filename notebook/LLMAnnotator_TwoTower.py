import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import torch
    from flow_judge.metrics import CustomMetric, RubricItem
    # from IPython.display import Markdown, display
    from flow_judge.flow_judge import EvalInput, FlowJudge
    from flow_judge.models import Vllm
    import json
    import pandas as pd
    return CustomMetric, EvalInput, FlowJudge, RubricItem, Vllm, pd


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
def _():
    # ## Run this cell, for scoring purpose. for now, it is commented out

    # evaluation_criteria = """How well does the input job post string data matches with the output resume string?"""

    # # Define the rubric using RubricItem's
    # rubric = [
    #     RubricItem(
    #         score=0,
    #         description="The resume is completely irrelevant to the job post. No skills, experience, or qualifications align."
    #     ),
    #     RubricItem(
    #         score=1,
    #         description="The resume shows almost no relevance to the job post. Only isolated or generic terms overlap."
    #     ),
    #     RubricItem(
    #         score=2,
    #         description="The resume has very limited relevance. It mentions one or two related aspects but lacks meaningful alignment."
    #     ),
    #     RubricItem(
    #         score=3,
    #         description="The resume shows slight relevance. A few keywords or skills match, but the overall context is different."
    #     ),
    #     RubricItem(
    #         score=4,
    #         description="The resume is somewhat related. It includes some relevant skills or experience, though most of it does not align with the job post."
    #     ),
    #     RubricItem(
    #         score=5,
    #         description="The resume is moderately related. About half of the skills, qualifications, or experiences align with the job post."
    #     ),
    #     RubricItem(
    #         score=6,
    #         description="The resume is fairly relevant. Most key skills or experiences match, but some important elements are missing."
    #     ),
    #     RubricItem(
    #         score=7,
    #         description="The resume is strongly relevant. It aligns with most of the job requirements, with only minor gaps in skills or experience."
    #     ),
    #     RubricItem(
    #         score=8,
    #         description="The resume is very well matched. It demonstrates strong alignment across skills, experience, and qualifications, with small deviations."
    #     ),
    #     RubricItem(
    #         score=9,
    #         description="The resume is an excellent match. It clearly fits the job post in almost every aspect, with only minor or contextual differences."
    #     ),
    #     RubricItem(
    #         score=10,
    #         description="The resume is a perfect match. It fully aligns with the job post in skills, experience, and qualifications with clear, direct relevance."
    #     ),
    # ]
    return


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
        quantized=True,               # ✅ correct argument name
        gpu_memory_utilization=0.98,
        max_model_len=6000
    )

    judge = FlowJudge(metric=sub_query_coverage, model=model)
    return (judge,)


@app.cell
def _(pd):
    ai_matching_df_cleaned = pd.read_csv('helping_datasets/job_candidate_annotation_data_two_tower_v2.csv')
    return (ai_matching_df_cleaned,)


@app.cell
def _(ai_matching_df_cleaned):
    ai_matching_df_cleaned
    return


@app.cell
def _(ai_matching_df_cleaned):
    ai_matching_df_cleaned.columns
    return


@app.cell
def _():
    return


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




    # ai_matching_df_cleaned[ai_matching_df_cleaned.select_dtypes(include="object").columns] = (
    #     ai_matching_df_cleaned.select_dtypes(include="object").fillna('unidentified')
    # )
    # ai_matching_df_cleaned[ai_matching_df_cleaned.select_dtypes(include="number").columns] = (
    #     ai_matching_df_cleaned.select_dtypes(include="number").fillna(0)
    # )

    # ai_matching_df_cleaned['job_string'] = ai_matching_df_cleaned['job_description'] + ' ' + ai_matching_df_cleaned['job_requirement']

    # filtered_df = ai_matching_df_cleaned[
    #     (ai_matching_df_cleaned['job_description'] != 'unidentified') &
    #     (ai_matching_df_cleaned['job_requirement'] != 'unidentified') &
    #     (ai_matching_df_cleaned['candidate_latest_resume_text'] != 'unidentified')
    # ]

    # data = (
    #     filtered_df[['job_string', 'candidate_latest_resume_text']]
    #     .rename(columns={
    #         'job_string': 'job_post',
    #         'candidate_latest_resume_text': 'resume'
    #     })
    #     .to_dict(orient='records')
    # )
    return (ai_matching_df_cleaned_,)


@app.cell
def _(ai_matching_df_cleaned_):
    ai_matching_df_cleaned_[['post_id','job_post_information', 'id', 'candidate_information']].to_csv('processed_dataset/two_tower_annotations.csv', index=False)
    return


@app.cell
def _(pd):
    ai_matching_df_cleaned_read = pd.read_csv('processed_dataset/two_tower_annotations.csv')
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
def _(data_batches):
    inputs_batch_0 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[0]]
    inputs_batch_1 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[1]]
    inputs_batch_2 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[2]]
    inputs_batch_3 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[3]]
    inputs_batch_4 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[4]]
    inputs_batch_5 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[5]]
    inputs_batch_6 = [[{"job_post_information": sample["job_post_information"]}] for sample in data_batches[6]]

    # # Extra Batches
    # inputs_batch_7 = [[{"job_post": sample["job_post"]}] for sample in data_batches[7]]
    # inputs_batch_8 = [[{"job_post": sample["job_post"]}] for sample in data_batches[8]]
    # inputs_batch_9 = [[{"job_post": sample["job_post"]}] for sample in data_batches[9]]
    # inputs_batch_10 = [[{"job_post": sample["job_post"]}] for sample in data_batches[10]]
    # inputs_batch_11 = [[{"job_post": sample["job_post"]}] for sample in data_batches[11]]
    # inputs_batch_12 = [[{"job_post": sample["job_post"]}] for sample in data_batches[12]]
    # inputs_batch_13 = [[{"job_post": sample["job_post"]}] for sample in data_batches[13]]
    # inputs_batch_14 = [[{"job_post": sample["job_post"]}] for sample in data_batches[14]]
    # inputs_batch_15 = [[{"job_post": sample["job_post"]}] for sample in data_batches[15]]
    # inputs_batch_16 = [[{"job_post": sample["job_post"]}] for sample in data_batches[16]]
    # inputs_batch_17 = [[{"job_post": sample["job_post"]}] for sample in data_batches[17]]
    # inputs_batch_18 = [[{"job_post": sample["job_post"]}] for sample in data_batches[18]]
    # inputs_batch_19 = [[{"job_post": sample["job_post"]}] for sample in data_batches[19]]



    outputs_batch_0 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[0]]
    outputs_batch_1 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[1]]
    outputs_batch_2 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[2]]
    outputs_batch_3 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[3]]
    outputs_batch_4 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[4]]
    outputs_batch_5 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[5]]
    outputs_batch_6 = [{"candidate_information": sample["candidate_information"]} for sample in data_batches[6]]


    # # Extra Batches
    # outputs_batch_7 = [{"resume": sample["resume"]} for sample in data_batches[7]]
    # outputs_batch_8 = [{"resume": sample["resume"]} for sample in data_batches[8]]
    # outputs_batch_9 = [{"resume": sample["resume"]} for sample in data_batches[9]]
    # outputs_batch_10 = [{"resume": sample["resume"]} for sample in data_batches[10]]
    # outputs_batch_11 = [{"resume": sample["resume"]} for sample in data_batches[11]]
    # outputs_batch_12 = [{"resume": sample["resume"]} for sample in data_batches[12]]
    # outputs_batch_13 = [{"resume": sample["resume"]} for sample in data_batches[13]]
    # outputs_batch_14 = [{"resume": sample["resume"]} for sample in data_batches[14]]
    # outputs_batch_15 = [{"resume": sample["resume"]} for sample in data_batches[15]]
    # outputs_batch_16 = [{"resume": sample["resume"]} for sample in data_batches[16]]
    # outputs_batch_17 = [{"resume": sample["resume"]} for sample in data_batches[17]]
    # outputs_batch_18 = [{"resume": sample["resume"]} for sample in data_batches[18]]
    # outputs_batch_19 = [{"resume": sample["resume"]} for sample in data_batches[19]]
    return (
        inputs_batch_0,
        inputs_batch_1,
        inputs_batch_2,
        inputs_batch_3,
        inputs_batch_4,
        inputs_batch_5,
        inputs_batch_6,
        outputs_batch_0,
        outputs_batch_1,
        outputs_batch_2,
        outputs_batch_3,
        outputs_batch_4,
        outputs_batch_5,
        outputs_batch_6,
    )


@app.cell
def _(
    EvalInput,
    inputs_batch_0,
    inputs_batch_1,
    inputs_batch_2,
    inputs_batch_3,
    inputs_batch_4,
    inputs_batch_5,
    inputs_batch_6,
    outputs_batch_0,
    outputs_batch_1,
    outputs_batch_2,
    outputs_batch_3,
    outputs_batch_4,
    outputs_batch_5,
    outputs_batch_6,
):
    eval_inputs_batch_0 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_0, outputs_batch_0)] #//
    eval_inputs_batch_1 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_1, outputs_batch_1)] #//
    eval_inputs_batch_2 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_2, outputs_batch_2)] #//
    eval_inputs_batch_3 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_3, outputs_batch_3)] #//
    eval_inputs_batch_4 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_4, outputs_batch_4)] #//
    eval_inputs_batch_5 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_5, outputs_batch_5)] #//
    eval_inputs_batch_6 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_6, outputs_batch_6)] #//

    # # Extra Batches
    # eval_inputs_batch_7 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_7, outputs_batch_7)] #//
    # eval_inputs_batch_8 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_8, outputs_batch_8)] # //
    # eval_inputs_batch_9 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_9, outputs_batch_9)] # //
    # eval_inputs_batch_10 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_10, outputs_batch_10)] # //
    # eval_inputs_batch_11 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_11, outputs_batch_11)] # //
    # eval_inputs_batch_12 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_12, outputs_batch_12)] # //
    # eval_inputs_batch_13 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_13, outputs_batch_13)] # //
    # eval_inputs_batch_14 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_14, outputs_batch_14)] # //
    # eval_inputs_batch_15 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_15, outputs_batch_15)] #
    # eval_inputs_batch_16 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_16, outputs_batch_16)]
    # eval_inputs_batch_17 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_17, outputs_batch_17)]
    # eval_inputs_batch_18 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_18, outputs_batch_18)]
    # eval_inputs_batch_19 = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch_19, outputs_batch_19)]
    return (eval_inputs_batch_0,)


@app.cell
def _():
    return


@app.cell
def _(data_batches, eval_inputs_batch_0, judge, pd):
    results_01 = judge.batch_evaluate(eval_inputs_batch_0, save_results=False)

    # === 6. Create results dataframe WITH ids ===
    results_dat_01 = [
        {
            "post_id": sample["post_id"],
            "id": sample["id"],
            "job_post_information": sample["job_post_information"],
            "candidate_information": sample["candidate_information"],
            "Feedback": result.feedback,
            "Score": result.score,
        }
        for sample, result in zip(data_batches[0], results_01)
    ]

    df_01 = pd.DataFrame(results_dat_01)

    df_01.to_csv('annotation_datasets/two_tower_annotations_B00_1000.csv')
    return


@app.cell
def _(eval_inputs_batch_15, judge, pd):
    results_15 = judge.batch_evaluate(eval_inputs_batch_15, save_results=False)

    results_dat_15 = [
        {
            "Sample": i + 1,
            "Feedback": result.feedback,
            "Score": result.score
        }
        for i, result in enumerate(results_15)
    ]

    df_15 = pd.DataFrame(results_dat_15)
    df_15.to_csv('data/ai_matching_annotation/resume_candidate/annotations_B13_1000.csv')
    return


@app.cell
def _(eval_inputs_batch_16, judge, pd):
    results_16 = judge.batch_evaluate(eval_inputs_batch_16, save_results=False)

    results_dat_16 = [
        {
            "Sample": i + 1,
            "Feedback": result.feedback,
            "Score": result.score
        }
        for i, result in enumerate(results_16)
    ]

    df_16 = pd.DataFrame(results_dat_16)
    df_16.to_csv('data/ai_matching_annotation/resume_candidate/annotations_B14_1000.csv')
    return


@app.cell
def _(eval_inputs_batch_17, judge, pd):
    results_17 = judge.batch_evaluate(eval_inputs_batch_17, save_results=False)

    results_dat_17 = [
        {
            "Sample": i + 1,
            "Feedback": result.feedback,
            "Score": result.score
        }
        for i, result in enumerate(results_17)
    ]

    df_17 = pd.DataFrame(results_dat_17)
    df_17.to_csv('data/ai_matching_annotation/resume_candidate/annotations_B15_1000.csv')
    return


@app.cell
def _(eval_inputs_batch_18, judge, pd):
    results_18 = judge.batch_evaluate(eval_inputs_batch_18, save_results=False)

    results_dat_18 = [
        {
            "Sample": i + 1,
            "Feedback": result.feedback,
            "Score": result.score
        }
        for i, result in enumerate(results_18)
    ]

    df_18 = pd.DataFrame(results_dat_18)
    df_18.to_csv('data/ai_matching_annotation/resume_candidate/annotations_B16_1000.csv')
    return


@app.cell
def _(eval_inputs_batch_19, judge, pd):
    results_19 = judge.batch_evaluate(eval_inputs_batch_19, save_results=False)

    results_dat_19= [
        {
            "Sample": i + 1,
            "Feedback": result.feedback,
            "Score": result.score
        }
        for i, result in enumerate(results_19)
    ]

    df_19 = pd.DataFrame(results_dat_19)
    df_19.to_csv('data/ai_matching_annotation/resume_candidate/annotations_B17_1000.csv')
    return


@app.cell
def _(eval_inputs_batch_13, judge, pd):
    results_13 = judge.batch_evaluate(eval_inputs_batch_13, save_results=False)

    results_dat_13 = [
        {
            "Sample": i + 1,
            "Feedback": result.feedback,
            "Score": result.score
        }
        for i, result in enumerate(results_13)
    ]

    df_13 = pd.DataFrame(results_dat_13)
    df_13.to_csv('data/ai_matching_annotation/resume_candidate/annotations_B11_1000.csv')
    return


@app.cell
def _(pd):
    df_an_b00 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B00_3000.csv')
    df_an_b01 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B01_1000.csv')
    df_an_b02 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B02_1000.csv')
    df_an_b03 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B03_1000.csv')
    df_an_b04 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B04_1000.csv')
    df_an_b05 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B05_1000.csv')
    df_an_b06 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B06_1000.csv')
    df_an_b07 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B07_1000.csv')
    df_an_b08 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B08_1000.csv')
    df_an_b09 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B09_1000.csv')
    df_an_b10 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B10_1000.csv')
    df_an_b11 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B11_1000.csv')
    df_an_b12 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B12_1000.csv')
    df_an_b13 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B13_1000.csv')
    df_an_b14 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B14_1000.csv')
    df_an_b15 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B15_1000.csv')
    df_an_b16 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B16_1000.csv')
    df_an_b17 = pd.read_csv('data/ai_matching_annotation/resume_candidate/annotations_B17_1000.csv')
    return (
        df_an_b00,
        df_an_b01,
        df_an_b02,
        df_an_b03,
        df_an_b04,
        df_an_b05,
        df_an_b06,
        df_an_b07,
        df_an_b08,
        df_an_b09,
        df_an_b10,
        df_an_b11,
        df_an_b12,
        df_an_b13,
        df_an_b14,
        df_an_b15,
        df_an_b16,
        df_an_b17,
    )


@app.cell
def _(
    df_an_b00,
    df_an_b01,
    df_an_b02,
    df_an_b03,
    df_an_b04,
    df_an_b05,
    df_an_b06,
    df_an_b07,
    df_an_b08,
    df_an_b09,
    df_an_b10,
    df_an_b11,
    df_an_b12,
    df_an_b13,
    df_an_b14,
    df_an_b15,
    df_an_b16,
    df_an_b17,
    pd,
):
    df_all_annotations = pd.concat(
        [df_an_b00, df_an_b01, df_an_b02, df_an_b03, df_an_b04, df_an_b05, df_an_b06, df_an_b07,
         df_an_b08, df_an_b09, df_an_b10, df_an_b11, df_an_b12, df_an_b13, df_an_b14, df_an_b15,
         df_an_b16, df_an_b17],
        ignore_index=True
    )
    return (df_all_annotations,)


@app.cell
def _(df_all_annotations, df_resume_job_string_df, pd):
    df_all_annotations
    df_final = pd.concat([df_all_annotations, df_resume_job_string_df], axis=1)
    df_final.to_csv('data/ai_matching_annotation/resume_candidate/all_annotations.csv')
    return


@app.cell
def _(pd):
    df_actual_data = pd.read_csv('data/ai_matching_annotation/resume_candidate/resm_cand_df.csv')
    df_resume_job_string_df = df_actual_data[['candidate_latest_resume_text', 'job_string']]
    # filtered_df.to_csv('data/ai_matching_annotation/resume_candidate/resm_cand_df.csv')
    return (df_resume_job_string_df,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
