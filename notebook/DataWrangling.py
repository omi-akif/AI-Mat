import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import ast
    import csv
    import json
    import pandas as pd
    from datetime import datetime
    from itertools import combinations
    from difflib import SequenceMatcher
    return SequenceMatcher, ast, combinations, json, pd


@app.cell
def _(pd):
    job_df_cleaned = pd.read_csv('cleaned_datasets/job_data_df_clean.csv')
    candidate_df_cleaned = pd.read_csv('cleaned_datasets/candidate_data_df_clean.csv')
    return candidate_df_cleaned, job_df_cleaned


@app.cell
def _(job_df_cleaned):
    job_df_cleaned
    return


@app.cell
def _(candidate_df_cleaned):
    candidate_df_cleaned
    return


@app.cell
def _(ast, candidate_df_cleaned):
    candidate_df_cleaned['preferredJobCategory_industry_names'] = candidate_df_cleaned['preferredJobCategory_industry_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])

    candidate_df_cleaned['preferredJobCategory_department_names'] = candidate_df_cleaned['preferredJobCategory_department_names'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [])
    return


@app.cell
def _(candidate_df_cleaned, job_df_cleaned, pd):
    # Word2Vec data production

    job_df_cleaned['job_text_data'] = job_df_cleaned['job_requirement'] + ' ' + job_df_cleaned['job_description']

    combined_texts = pd.concat([
        job_df_cleaned['job_text_data'].dropna(),
        candidate_df_cleaned[candidate_df_cleaned['candidate_latest_resume_text'] != 'undefined']['candidate_latest_resume_text'].reset_index(drop=True).dropna()
    ], ignore_index=True).tolist()

    with open('helping_datasets/job_candidate_text_data_word2vec.txt', 'w') as f:
        for text in combined_texts:
            f.write(text + '\n')
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
    combined_df.to_csv('helping_datasets/job_candidate_annotation_data_two_tower_v2.csv')
    return (combined_df,)


@app.cell
def _(combined_df):
    # Sample for DLEM model
    combined_df[['job_text_data', 'candidate_latest_resume_text']].to_csv('helping_datasets/job_candidate_annotation_data_dlem.csv', index=False)
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

    job_to_job_transition_df.to_csv('helping_datasets/job_to_job_transition_inf_lear.csv', index=False)
    return


@app.cell
def _(ast, job_df_cleaned):
    # Sample for Job Title to Job Skill transition graph

    # job_to_skill_transition_df = job_df_cleaned[['job_title', 'job_skill_name']].explode('job_skill_name').reset_index(drop=True)
    # job_to_skill_transition_df.columns = ['job_title', 'job_skill']
    # job_to_skill_transition_df.to_csv('helping_datasets/job_to_skill_transition_inf_lear.csv')


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
        'helping_datasets/job_to_skill_relation_inf_lear.csv',
        index=False
    )
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
    skill_to_skill_relation_df.to_csv("helping_datasets/skill_to_skill_relation_inf_lear.csv", index=False)
    return


@app.cell(hide_code=True)
def _():
    # candidate_df = pd.read_csv('uncleaned_datasets/candidate_opensearch_export.csv')
    return


@app.cell(hide_code=True)
def _():
    # candidate_df.columns
    return


@app.cell(hide_code=True)
def _():
    # experience_filtered_df = candidate_df[candidate_df['candidate_experience'] != '[]']
    return


@app.cell(hide_code=True)
def _():
    # experience_filtered_df['candidate_experience']
    return


@app.cell(hide_code=True)
def _():
    # experience_filtered_df['candidate_experience'] = experience_filtered_df['candidate_experience'].apply(
    #     lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
    # )
    # experience_filtered_df['candidate_experience_len'] = experience_filtered_df['candidate_experience'].apply(len)
    return


@app.cell(hide_code=True)
def _():
    # def sort_experiences(lst):
    #     if not isinstance(lst, list):
    #         return lst
    #     # Convert strings to datetime safely; handle None
    #     def to_date(d):
    #         try:
    #             return datetime.strptime(d, "%Y-%m-%d")
    #         except Exception:
    #             return datetime.min  # treat missing/invalid as earliest

    #     return sorted(
    #         lst,
    #         key=lambda x: (
    #             to_date(x.get('start_date')),
    #             to_date(x.get('end_date'))
    #         ),
    #         reverse=True
    #     )


    # def extract_transitions_text_chronological(experience_list):
    #     """
    #     Convert sorted experiences into a list of directed job transitions as strings
    #     showing progression from oldest -> newest
    #     """
    #     if not isinstance(experience_list, list) or len(experience_list) < 2:
    #         return []

    #     # Reverse the list to go from oldest to newest
    #     lst = experience_list[::-1]

    #     transitions = []
    #     for i in range(len(lst) - 1):
    #         from_role = lst[i]['role'] or 'unknown'
    #         to_role = lst[i+1]['role'] or 'unknown'
    #         transitions.append(f"{from_role} -> {to_role}")
    #     return transitions


    # def lowercase_roles(experience_list):
    #     if not isinstance(experience_list, list):
    #         return experience_list

    #     for exp in experience_list:
    #         if 'role' in exp and exp['role']:
    #             exp['role'] = exp['role'].lower()
    #     return experience_list
    return


@app.cell(hide_code=True)
def _():
    # experience_filtered_df['candidate_experience_sorted'] = experience_filtered_df['candidate_experience'].apply(sort_experiences)
    # experience_filtered_df['candidate_experience_sorted'] = experience_filtered_df['candidate_experience_sorted'].apply(lowercase_roles)
    # experience_filtered_df['candidate_experience_sorted_graph'] = experience_filtered_df['candidate_experience_sorted'].apply(extract_transitions_text_chronological)
    return


@app.cell(hide_code=True)
def _():
    # column_name = 'candidate_experience_sorted_graph'
    # output_csv = "uncleaned_datasets/job_to_job_transition_ORIGINAL.csv"

    # with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
    #     writer = csv.writer(f_out)
    #     writer.writerow(["from_role", "to_role"])  # CSV header

    #     for transitions in experience_filtered_df[column_name]:
    #         if not transitions:  # skip empty lists
    #             continue
    #         for line in transitions:
    #             # Clean the line: remove quotes and extra spaces
    #             line_clean = line.strip().strip("'")
    #             if '->' in line_clean:
    #                 from_role, to_role = [x.strip() for x in line_clean.split('->', 1)]
    #                 writer.writerow([from_role, to_role])
    return


@app.cell(hide_code=True)
def _():
    # d_jj_df = pd.read_csv('uncleaned_datasets/job_to_job_transition.csv')
    # filtered_df_d_jj = d_jj_df[(d_jj_df['from_role'] != 'unknown') | (d_jj_df['to_role'] != 'unknown')]

    # # Clean up spaces and make everything lowercase
    # d_jj_df['from_role'] = d_jj_df['from_role'].str.strip().str.lower()
    # d_jj_df['to_role'] = d_jj_df['to_role'].str.strip().str.lower()

    # # Now filter out rows where either column is 'unknown'
    # clean_df = d_jj_df[(d_jj_df['from_role'] != 'unknown') & (d_jj_df['to_role'] != 'unknown')]
    # filtered_df_d_jj.to_csv('uncleaned_datasets/job_to_job_transition.csv')
    return


if __name__ == "__main__":
    app.run()
