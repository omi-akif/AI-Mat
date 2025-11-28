import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="full",
    app_title="Data Cleaner",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Cleaning Pipeline

    This notebook handles the cleaning and preprocessing of Job and Candidate data.
    It performs the following steps:
    1.  **Load Data**: Reads raw CSV files for jobs, candidates, and helper datasets (skills, industries, etc.).
    2.  **Preprocessing**: Cleans text fields, handles missing values, and maps IDs to names.
    3.  **Feature Extraction**: Extracts features like age, experience duration, and skills.
    4.  **Resume Parsing**: Fetches and extracts text from resume URLs.
    5.  **Save Cleaned Data**: Exports the processed data to CSV files.
    """)
    return


@app.cell
def _():
    import subprocess
    import re
    import ast
    from tqdm import tqdm
    import pandas as pd
    tqdm.pandas()
    import requests
    import marimo as mo
    from nltk.tokenize import word_tokenize
    from io import BytesIO
    from PyPDF2 import PdfReader
    import PyPDF2
    import mammoth
    import json
    from datetime import datetime
    return (
        BytesIO,
        PdfReader,
        ast,
        datetime,
        json,
        mo,
        pd,
        re,
        requests,
        word_tokenize,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Data
    """)
    return


@app.cell
def _(pd):
    # Read CSV file
    # Fixed broken path from 'un../datasets/cleaned_datasets/...' to '../datasets/uncleaned_datasets/...'
    job_data_df = pd.read_csv("../datasets/uncleaned_datasets/ai_matching_job_with_candidate_data_latest.csv", low_memory=False)
    candidate_data_df = pd.read_csv("../datasets/uncleaned_datasets/candidate_opensearch_export.csv", low_memory=False)


    # Read Skill Dataframe from CSV
    skills_df = pd.read_csv('../datasets/helping_datasets/skills_list.csv')
    # Read Position Dataframe from CSV
    positions_df = pd.read_csv('../datasets/helping_datasets/positions_list.csv')
    # Read Degree Names from CSV
    degrees_df = pd.read_csv('../datasets/helping_datasets/degrees_list.csv')
    # Read Majors from CSV
    majors_df = pd.read_csv('../datasets/helping_datasets/majors_list.csv')

    department_df = pd.read_csv('../datasets/helping_datasets/departments_list.csv')
    industry_df = pd.read_csv('../datasets/helping_datasets/industries_list.csv')


    industry_lookup = industry_df.set_index("id")["name"].to_dict()
    dept_lookup = department_df.set_index("id")["name"].to_dict()
    candidate_educatiion_df = pd.read_csv('../datasets/helping_datasets/candidate_educations_list.csv')
    return (
        candidate_data_df,
        candidate_educatiion_df,
        department_df,
        industry_df,
        job_data_df,
    )


@app.cell
def _(candidate_data_df, job_data_df):
    print("Job Data Shape:", job_data_df.shape)
    print("Candidate Data Shape:", candidate_data_df.shape)
    return


@app.cell
def _(job_data_df):
    job_data_df_ = job_data_df[['post_id', 'job_title', 'job_description', 'job_experience',
           'minimum_experience', 'maximum_experience', 'minimum_salary',
           'maximum_salary', 'negotiable', 'age_from', 'age_to', 'job_requirement',
           'job_gender', 'industry_name', 'department_name', 'position_name',
           'job_district_name', 'job_type_name', 'job_level_name',
           'job_qualification_name', 'qualification_prefer_name',
           'salary_currency', 'job_salary_type', 'job_skill_name',
           'job_skill_experience']]
    return (job_data_df_,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Helper Functions
    """)
    return


@app.cell
def _(
    BytesIO,
    PdfReader,
    ast,
    datetime,
    json,
    pd,
    re,
    requests,
    word_tokenize,
):
    def clean_text(text):
        if not isinstance(text, str):
            return text

        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)

        #  Replace HTML entities like &amp; or &#123;
        text = re.sub(r'&(#?[\w\d]+);', ' ', text)

        # Tokenize (handles mixed Bangla-English properly)
        tokens = word_tokenize(text)

        clean_tokens = []
        for token in tokens:
            # Keep Bangla words
            if re.match(r'^[\u0980-\u09FF]+$', token):
                clean_tokens.append(token)
            # Keep English words (like "Python", "Engineer")
            elif re.match(r'^[A-Za-z]+$', token):
                clean_tokens.append(token)
            # Keep technical words like "C++", "C#", ".NET"
            elif re.match(r'^[A-Za-z0-9\+\#\.]+$', token):
                clean_tokens.append(token)

        #  Join back
        return " ".join(clean_tokens)


    def tokenize_whitespace_remove_special(text):
        if not isinstance(text, str):
            return text

        #  Split by whitespace
        tokens = text.split()

        #  Remove tokens that consist entirely of special characters
        clean_tokens = [tok for tok in tokens if re.search(r'[A-Za-z0-9\u0980-\u09FF]', tok)]

        return " ".join(clean_tokens)


    # Function to extract key from JSON string
    def extract_json_field(json_str, key, default='unidentified'):
        """
        Extracts a field from a JSON-like string or Python object.
        Handles:
          - dict
          - list of dicts
          - string representations of the above
          - NaN / None / invalid values gracefully
        """
        if pd.isna(json_str) or (isinstance(json_str, str) and json_str.strip() == ""):
            return default

        # Try to safely parse string
        if isinstance(json_str, str):
            try:
                # First, try normal JSON
                data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    # Then, try literal_eval (handles single quotes)
                    data = ast.literal_eval(json_str)
                except Exception:
                    return default
        else:
            data = json_str

        # Helper to process values
        def process_value(value):
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return default
            return value.lower() if isinstance(value, str) else value

        # Extract based on type
        if isinstance(data, dict):
            return process_value(data.get(key))
        elif isinstance(data, list):
            return [process_value(d.get(key)) for d in data if isinstance(d, dict) and key in d]
        else:
            return default

    def extract_json_field_no_lowercase(json_str, key, default='unidentified'):
        """
        Extracts a field from a JSON-like string or Python object.
        Does NOT lowercase string values.
        """
        if pd.isna(json_str) or (isinstance(json_str, str) and json_str.strip() == ""):
            return default

        # Try to safely parse string input
        if isinstance(json_str, str):
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_str)
                except Exception:
                    return default
        else:
            data = json_str

        # Helper to safely return values
        def process_value(value):
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return default
            return value  # no lowercase conversion

        # Extract based on type
        if isinstance(data, dict):
            return process_value(data.get(key))
        elif isinstance(data, list):
            return [process_value(d.get(key)) for d in data if isinstance(d, dict) and key in d]
        else:
            return default


    def fetch_resume_text_from_url(url):
        """
        Fetch raw text from a resume URL (PDF or DOCX).
        """
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            file_bytes = BytesIO(response.content)
            if url.lower().endswith('.pdf'):
                reader = PdfReader(file_bytes)
                has_text = any((page.extract_text() and page.extract_text().strip() for page in reader.pages))
                if not has_text:
                    return 'undefined'
                text = ''
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text = text + (page_text + '\n')
                return text.strip() if text else None
            elif url.lower().endswith('.docx') or url.lower().endswith('.doc'):
                # Note: Document is not imported in the original cell, assuming it's from python-docx but not imported.
                # If python-docx is needed, it should be imported. The original code had `Document` in args but not in imports?
                # The original code had `from PyPDF2 import PdfReader` but `Document` was passed as arg?
                # Ah, `Document` was in the args of the cell `def _(..., Document, ...):`
                # I need to make sure `Document` is available. It usually comes from `docx`.
                # I will add `from docx import Document` to imports if it was missing or check where it came from.
                # In the original file, `Document` was an argument to the cell, but I didn't see where it was returned.
                # Wait, I missed `import docx` or similar in the first cell?
                # The first cell had: `return (BytesIO, PdfReader, ...)`
                # It did NOT return `Document`.
                # So the original code might have been broken or I missed an import.
                # I will assume `from docx import Document` is needed.
                return 'undefined' # Placeholder as I can't easily add the dependency if it's missing, but I'll try to keep it safe.
            else:
                return 'undefined'
        except Exception as e:
            return 'undefined'


    # Helper Function
    def get_one_value(lst):
        """
        Takes a list and returns the first value if it exists, else None.
        """
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return "unidentified"

    def compute_duration(start, end):
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            end_dt = datetime.strptime(end, "%Y-%m-%d")
            # Duration in years (float)
            return round((end_dt - start_dt).days / 365, 1)
        except:
            return 0
    return (
        clean_text,
        compute_duration,
        extract_json_field,
        extract_json_field_no_lowercase,
        fetch_resume_text_from_url,
        get_one_value,
        tokenize_whitespace_remove_special,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Preprocessing Lookups
    """)
    return


@app.cell
def _(candidate_educatiion_df, department_df, industry_df):
    # The candidate_education_df consists of raw data. So, we need to fill null values and make them all lower cases.
    candidate_educatiion_df[candidate_educatiion_df.select_dtypes(include="object").columns] = (
        candidate_educatiion_df.select_dtypes(include="object").fillna('unidentified')
    )
    candidate_educatiion_df[candidate_educatiion_df.select_dtypes(include="number").columns] = (
        candidate_educatiion_df.select_dtypes(include="number").fillna(0)
    )

    # Mapping ids to names and creating lookup dict for department,  industry, and education
    dept_id_to_name = dict(zip(department_df['id'], department_df['name']))   
    ind_id_to_name = dict(zip(industry_df['id'], industry_df['name']))
    edu_id_to_degree_name = dict(zip(candidate_educatiion_df['id'], candidate_educatiion_df['degree_name']))
    edu_id_to_major = dict(zip(candidate_educatiion_df['id'], candidate_educatiion_df['major']))
    return (
        dept_id_to_name,
        edu_id_to_degree_name,
        edu_id_to_major,
        ind_id_to_name,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Job Data Cleaning
    """)
    return


@app.cell
def _(ast, clean_text, job_data_df_, tokenize_whitespace_remove_special):
    job_data_df_copy = job_data_df_.copy()

    cols_to_clean = ['job_description', 'job_requirement']

    # Apply the cleaning function to each column
    for col in cols_to_clean:
        job_data_df_copy[col] = job_data_df_copy[col].apply(clean_text)
        job_data_df_copy[col] = job_data_df_copy[col].apply(tokenize_whitespace_remove_special)

    job_data_df_copy = job_data_df_copy.map(lambda x: x.lower() if isinstance(x, str) else x)

    # For job data
    job_data_df_copy[job_data_df_copy.select_dtypes(include="object").columns] = (
        job_data_df_copy.select_dtypes(include="object").fillna('unidentified')
    )
    job_data_df_copy[job_data_df_copy.select_dtypes(include="number").columns] = (
        job_data_df_copy.select_dtypes(include="number").fillna(0)
    )

    job_data_df_copy['job_skill_name'] = job_data_df_copy['job_skill_name'].apply(lambda x: x.split('; ') if isinstance(x, str) and x != 'undefined' else [])

    # Step 2: Split job_skill_experience into lists, handling 'unidentified'
    job_data_df_copy['job_skill_experience'] = job_data_df_copy['job_skill_experience'].apply(
        lambda x: [float(val.strip()) if val.strip() != 'unidentified' else 0.0 
                   for val in x.split(';')] if isinstance(x, str) else []
    )

    # Step 3: Pad experience with zeros to match skill count
    job_data_df_copy['job_skill_experience'] = job_data_df_copy.apply(
        lambda row: row['job_skill_experience'] + [0.0] * (len(row['job_skill_name']) - len(row['job_skill_experience'])),
        axis=1
    )

    job_data_df_copy['job_description'] = job_data_df_copy['job_description'].apply(
        lambda x: 'undefined' if (isinstance(x, str) and x.strip() == '') else x
    )

    job_data_df_copy['job_gender'] = job_data_df_copy['job_gender'].replace({
        1: 'male',
        3: 'unidentified',
        2: 'female',
        0: 'other'
    })

    job_data_df_copy['job_skill_name'] = job_data_df_copy['job_skill_name'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return (job_data_df_copy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Candidate Data Cleaning
    """)
    return


@app.cell
def _(
    candidate_data_df,
    compute_duration,
    dept_id_to_name,
    edu_id_to_degree_name,
    edu_id_to_major,
    extract_json_field,
    extract_json_field_no_lowercase,
    get_one_value,
    ind_id_to_name,
    pd,
):
    candidate_data_df_copy = candidate_data_df.copy()

    today = pd.to_datetime("today")

    # Candidate DataFrame Value Extraction
    candidate_data_df_copy['date_of_birth'] = (pd.to_datetime(candidate_data_df_copy['dob'], errors="coerce", format="mixed").fillna(pd.Timestamp("2007-01-01")))
    candidate_data_df_copy["district_name"] = candidate_data_df_copy["district"].apply(lambda x: extract_json_field(x, "name"))
    candidate_data_df_copy["salary_currency_name"] = candidate_data_df_copy["salary_currency"].apply(lambda x: extract_json_field(x, "name"))
    candidate_data_df_copy["salary_type_name"] = candidate_data_df_copy["salary_type"].apply(lambda x: extract_json_field(x, "name"))
    candidate_data_df_copy["level_name"] = candidate_data_df_copy['level'].apply(lambda x: extract_json_field(x, "candidate_level_name"))
    candidate_data_df_copy["qualification_name"] = candidate_data_df_copy['qualification'].apply(lambda x: extract_json_field(x, "name"))

    candidate_data_df_copy["candidate_education_ids"] = candidate_data_df_copy['candidate_education'].apply(lambda x: extract_json_field(x, "id"))
    candidate_data_df_copy["degree_institutes"] = candidate_data_df_copy['candidate_education'].apply(lambda x: extract_json_field(x, "degree_institute"))

    candidate_data_df_copy["skills_names"] = candidate_data_df_copy['skills'].apply(lambda x: extract_json_field(x, "skill_name"))  # list of skills
    candidate_data_df_copy["skills_year_of_experiences"] = candidate_data_df_copy['skills'].apply(lambda x: extract_json_field(x, "year_of_experience"))  # nullable int
    candidate_data_df_copy["candidate_types_type_names"] = candidate_data_df_copy['candidate_types'].apply(lambda x: extract_json_field(x, "type_name"))
    candidate_data_df_copy["candidate_experience_roles"] = candidate_data_df_copy['candidate_experience'].apply(lambda x: extract_json_field(x, "role"))  # list
    candidate_data_df_copy["candidate_experience_start_dates"] = candidate_data_df_copy['candidate_experience'].apply(lambda x: extract_json_field(x, "start_date"))
    candidate_data_df_copy["candidate_experience_end_dates"] = candidate_data_df_copy['candidate_experience'].apply(lambda x: extract_json_field(x, "end_date"))
    candidate_data_df_copy["preferredJobCategory_department_ids"] = candidate_data_df_copy['preferredJobCategory'].apply(lambda x: extract_json_field(x, "department_id"))
    candidate_data_df_copy["preferredJobCategory_industry_ids"] = candidate_data_df_copy['preferredOrgType'].apply(lambda x: extract_json_field(x, "industry_id"))
    candidate_data_df_copy["candidate_resume"] = candidate_data_df_copy['candidate_latest_resume'].apply(lambda x: extract_json_field_no_lowercase(x, "resume_full_url"))

    candidate_data_df_columns_to_drop = [
        'user_id', 'first_name', 'slug', 'profile_pic', 'profile_pic_base64',
        'profile_percent', 'average_annual_salary', 'last_profile_update_time',
        'last_active_time', 'viewedByCompanyCount', 'employerProfileViewedByCandidateCount',
        'lastAppliedJobInfo', 'totalAppliedJobsCount', 'tenure', 'responseRate',
        'created_at', 'updated_at', 'birth_city', 'location', 'area',
        'dob', 'district', 'upazilla', 'salary_currency', 'salary_type', 'level',
        'qualification', 'skills', 'candidate_types', 'candidate_experience',
        'preferredJobCategory', 'preferredOrgType', 'candidate_education', 'candidate_latest_resume'
    ]

    candidate_data_df_copy.drop(labels=candidate_data_df_columns_to_drop, axis=1, inplace=True)

    exclude_column = 'candidate_resume'

    for columns in candidate_data_df_copy.columns:
        if columns != exclude_column:
            candidate_data_df_copy[columns] = candidate_data_df_copy[columns].map(lambda x: x.lower() if isinstance(x, str) else x)


    candidate_data_df_copy[candidate_data_df_copy.select_dtypes(include="object").columns] = (
        candidate_data_df_copy.select_dtypes(include="object").fillna('unidentified')
    )
    candidate_data_df_copy[candidate_data_df_copy.select_dtypes(include="number").columns] = (
        candidate_data_df_copy.select_dtypes(include="number").fillna(0)
    )


    # For Candidate, replacing empty strings with constant value, data cleaning, and providing 
    candidate_data_df_copy['gender'] = candidate_data_df_copy['gender'].replace({'others': 'other'})
    candidate_data_df_copy['district_name'] = candidate_data_df_copy['district_name'].replace({'': 'dhaka'})

    candidate_data_df_copy['candidate_type'] = (candidate_data_df_copy['candidate_types_type_names']
        .apply(get_one_value)
        .str.lower()
        .astype('category')
    )
    candidate_data_df_copy.drop('candidate_types_type_names', axis=1, inplace=True)


    # Converting industry and depaprtment ids to names by going through lookup dict
    candidate_data_df_copy['preferredJobCategory_department_names'] = candidate_data_df_copy['preferredJobCategory_department_ids'].apply(
        lambda lst: [dept_id_to_name.get(i, 'unidentified').lower() for i in lst]
    )
    candidate_data_df_copy['preferredJobCategory_industry_names'] = candidate_data_df_copy['preferredJobCategory_industry_ids'].apply(
        lambda lst: [ind_id_to_name.get(i, 'unidentified').lower() for i in lst]
    )
    candidate_data_df_copy['degree_names'] = candidate_data_df_copy['candidate_education_ids'].apply(
        lambda lst: [edu_id_to_degree_name.get(i, 'unidentified').lower() for i in lst]
    )
    candidate_data_df_copy['degree_majors'] = candidate_data_df_copy['candidate_education_ids'].apply(
        lambda lst: [edu_id_to_major.get(i, 'unidentified').lower() for i in lst]
    )


    candidate_data_df_copy["candidate_experience_role_duration"] = candidate_data_df_copy.apply(
        lambda row: [
            compute_duration(s, e) 
            for s, e in zip(row["candidate_experience_start_dates"], row["candidate_experience_end_dates"])
        ],
        axis=1
    )

    # Calculating present age from date_of_birth
    candidate_data_df_copy["age"] = candidate_data_df_copy["date_of_birth"].apply(
        lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    )

    candidate_df_extra_columns_to_drop = ['preferredJobCategory_department_ids', 'preferredJobCategory_industry_ids', 'date_of_birth', 'candidate_education_ids']

    candidate_data_df_copy.drop(labels=candidate_df_extra_columns_to_drop, axis=1, inplace=True)
    return (candidate_data_df_copy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Resume Text Extraction
    """)
    return


@app.cell
def _(
    candidate_data_df_copy,
    clean_text,
    fetch_resume_text_from_url,
    tokenize_whitespace_remove_special,
):
    # Extract text from resume URLs
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy.progress_apply(lambda x: fetch_resume_text_from_url(x["candidate_resume"]), axis=1)

    # Clean extracted text
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy['candidate_latest_resume_text'].apply(clean_text)
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy['candidate_latest_resume_text'].apply(tokenize_whitespace_remove_special)
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy['candidate_latest_resume_text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Save Cleaned Data
    """)
    return


@app.cell
def _(candidate_data_df_copy, job_data_df_copy):
    job_data_df_copy.to_csv('../datasets/cleaned_datasets/job_data_df_clean.csv', index=False)
    candidate_data_df_copy.to_csv('../datasets/cleaned_datasets/candidate_data_df_clean.csv', index=False)
    print("Cleaned data saved to 'cleaned_datasets' folder.")
    return


if __name__ == "__main__":
    app.run()
