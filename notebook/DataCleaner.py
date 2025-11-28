import marimo

__generated_with = "0.17.7"
app = marimo.App(
    width="full",
    app_title="Data Cleaner",
    auto_download=["html"],
)


@app.cell
def _():
    import subprocess
    import re
    import ast
    from tqdm import tqdm
    import pandas as pd
    tqdm.pandas()
    import requests
    import re
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


@app.cell
def _(pd):
    # Read CSV file
    job_data_df = pd.read_csv("uncleaned_datasets/ai_matching_job_with_candidate_data_latest.csv", low_memory=False)
    candidate_data_df = pd.read_csv("uncleaned_datasets/candidate_opensearch_export.csv", low_memory=False)


    # Read Skill Dataframe from CSV
    skills_df = pd.read_csv('helping_datasets/skills_list.csv')
    # Read Position Dataframe from CSV
    positions_df = pd.read_csv('helping_datasets/positions_list.csv')
    # Read Location for Bounding Box from CSV
    # location_df = pd.read_csv('helping_datasets/locations_list.csv') # This contains all the geo-encodes, which is the bounding boxes
    # Read Degree Names from CSV
    degrees_df = pd.read_csv('helping_datasets/degrees_list.csv')
    # Read Majors from CSV
    majors_df = pd.read_csv('helping_datasets/majors_list.csv')
    # Read Institute Names from CSV
    # institutes_df = pd.read_csv('helping_datasets/institutes_list.csv')

    department_df = pd.read_csv('helping_datasets/departments_list.csv')
    industry_df = pd.read_csv('helping_datasets/industries_list.csv')


    industry_lookup = industry_df.set_index("id")["name"].to_dict()
    dept_lookup = department_df.set_index("id")["name"].to_dict()
    candidate_educatiion_df = pd.read_csv('helping_datasets/candidate_educations_list.csv')




    # job_to_job_data_df_copy = pd.read_csv("uncleaned_datasets/job_to_job_transition.csv", low_memory=False)
    return (
        candidate_data_df,
        candidate_educatiion_df,
        department_df,
        industry_df,
        job_data_df,
    )


@app.cell
def _(job_data_df):
    job_data_df
    return


@app.cell
def _(candidate_data_df):
    candidate_data_df
    return


@app.cell
def _(job_data_df):
    job_data_df.columns
    return


@app.cell
def _(candidate_data_df):
    candidate_data_df.columns
    return


@app.cell
def _(job_data_df):
    job_data_df.shape
    return


@app.cell
def _(candidate_data_df):
    candidate_data_df.shape
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


@app.cell
def _(
    BytesIO,
    Document,
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
        Handles:
          - dict
          - list of dicts
          - string representations of the above
          - NaN / None / invalid values gracefully
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

        Args:
            url (str): URL to the resume file

        Returns:
            str: extracted raw text or None if failed
        """
        try:
            response = requests.get(url, timeout=15)  # Check if URL is NaN or empty
            response.raise_for_status()  # if pd.isna(url) or not isinstance(url, str) or url.strip() == "":
            file_bytes = BytesIO(response.content)  #     return None
            if url.lower().endswith('.pdf'):
                reader = PdfReader(file_bytes)  # Download the file content
                has_text = any((page.extract_text() and page.extract_text().strip() for page in reader.pages))
                if not has_text:
                    return 'undefined'
                text = ''
                for page in reader.pages:  # Check file type by URL extension
                    page_text = page.extract_text()
                    if page_text:
                        text = text + (page_text + '\n')
                return text.strip() if text else None  # Check if PDF has extractable text
            elif url.lower().endswith('.docx') or url.lower().endswith('.doc'):
                doc = Document(file_bytes)
                text = '\n'.join([p.text for p in doc.paragraphs])
                return text.strip() if text else None
            else:  # Extract text normally
                return 'undefined'
        except Exception as e:
            return 'undefined'  # print(f"Unsupported file type for URL: {url}")  # print(f"Error fetching/parsing URL {url}: {e}")

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

    def tokenize_whitespace_remove_special(text):
        if not isinstance(text, str):
            return text

        #  Split by whitespace
        tokens = text.split()

        #  Remove tokens that consist entirely of special characters
        clean_tokens = [tok for tok in tokens if re.search(r'[A-Za-z0-9\u0980-\u09FF]', tok)]

        return " ".join(clean_tokens)


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
    # candidate_data_df_copy["upazilla_name"] = candidate_data_df_copy["upazilla"].apply(lambda x: extract_json_field(x, "name"))
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
    # job_data_df_copy['upazilla_name'] = job_data_df_copy['upazilla_name'].replace({'': 'dhaka'})
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

    # candidate_data_df_copy_extra_columns_to_drop = ['preferredJobCategory_department_ids', 'preferredJobCategory_industry_ids', 'skills_names', 'skills_year_of_experiences', 'candidate_experience_roles', 'candidate_experience_start_dates', 'candidate_experience_end_dates', 'upazilla_name', 'district_name', 'date_of_birth']
    candidate_df_extra_columns_to_drop = ['preferredJobCategory_department_ids', 'preferredJobCategory_industry_ids', 'date_of_birth', 'candidate_education_ids']
    # job_df_extra_columns_to_drop = ['job_shift_name', 'skills_years', 'upazilla_name', 'district_name']

    candidate_data_df_copy.drop(labels=candidate_df_extra_columns_to_drop, axis=1, inplace=True)
    # job_df.drop(labels=job_df_extra_columns_to_drop, axis=1, inplace=True)
    return (candidate_data_df_copy,)


@app.cell
def _(
    candidate_data_df_copy,
    clean_text,
    fetch_resume_text_from_url,
    tokenize_whitespace_remove_special,
):
    # candidate_data_df_copy_ = candidate_data_df_copy_.iloc[1:100]

    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy.progress_apply(lambda x: fetch_resume_text_from_url(x["candidate_resume"]), axis=1)
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy['candidate_latest_resume_text'].apply(clean_text)
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy['candidate_latest_resume_text'].apply(tokenize_whitespace_remove_special)
    candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy['candidate_latest_resume_text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return


@app.cell
def _():
    return


@app.cell
def _(job_data_df_copy):
    job_data_df_copy
    return


@app.cell
def _(candidate_data_df_copy):
    candidate_data_df_copy
    return


@app.cell
def _():
    # candidate_data_df_copy['candidate_latest_resume_text'] = candidate_data_df_copy_['candidate_latest_resume_text']
    return


@app.cell
def _(candidate_data_df_copy, job_data_df_copy):
    job_data_df_copy.to_csv('cleaned_datasets/job_data_df_clean.csv', index=False)
    candidate_data_df_copy.to_csv('cleaned_datasets/candidate_data_df_clean.csv', index=False)


    # job_data_df_copy_ = pd.read_csv('cleaned_datasets/job_data_df_clean.csv')
    # candidate_data_df_copy_ = pd.read_csv('cleaned_datasets/candidate_data_df_clean.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM based data cleaning for string type data
    """)
    return


@app.cell
def _():
    # def clean_role(role: str):
    #     """
    #     Clean and normalize a job title using a local Ollama model.
    #     Returns (thinking, cleaned_title)
    #     """
    #     if not isinstance(role, str) or not role.strip():
    #         return ("", role)

    #     prompt = f"Clean this job title below. If you think this is a wrong job title, provide the correct one. Also all job titles should be in lower case.\nJob Title: '{role}'. Only output the cleaned title."

    #     try:

    #         resp = requests.post(
    #             "http://localhost:11434/api/generate",
    #             json={"model": "qwen3:8b", "prompt": prompt, "stream": False},
    #             timeout=60
    #         )
    #         output = resp.json().get("response", "")

    #         response = output.split('</think>')[-1]

    #         return response
    #     except Exception:
    #         return role



    # def clean_roles_batch(roles: list):
    #     prompt = "Clean these job titles below and normalize them. If you think this is a wrong job title, provide the correct one. Also all job titles should be in lower case. Only give the job title. Maintain the same order the order that was used when the job titles are given. Remove Company names and tools from the jobs:\n Job Titles: \n"
    #     prompt += "\n".join(f"{r}" for r in roles)

    #     resp = requests.post(
    #         "http://localhost:11434/api/generate",
    #         json={"model": "qwen3:8b", "prompt": prompt, "stream": False},
    #         timeout=60
    #     )
    #     output = resp.json().get("response", "")

    #     response = output.split('</think>')[-1]

    #     job_list = [title.strip() for title in response.split("\n") if title.strip()]

    #     return job_list
    return


@app.cell(hide_code=True)
def _():
    # output = clean_roles_batch(roles=['customer service representative (c.s.r)','bookkeeper ( quickbooks online)', 'sr. manager (outlet operations)'])
    return


@app.cell(hide_code=True)
def _():
    # clean_role('customer service representative (c.s.r)')
    return


@app.cell
def _():
    # df_1 = df.iloc[1:10000]
    # df_2 = df.iloc[10000:20000]
    # df_3 = df.iloc[20000:30000]
    # df_4 = df.iloc[30000:35413]
    return


@app.cell(hide_code=True)
def _():
    # print("🧹 Cleaning from_role...")
    # # df_1["from_role_cleaned"] = df_1["from_role"].progress_apply(clean_role)

    # df_1[["from_role_thinking", "from_role_cleaned"]] = df_1["from_role"].progress_apply(clean_role).apply(pd.Series)
    return


@app.cell
def _():

    # #df_2['from_role_clean'] = df_2['from_role'].progress_apply(clean_role
    # # df_2['from_role_clean'] = df_2['from_role'].progress_apply(clean_role)
    # df_3['from_role_clean'] = df_3['from_role'].progress_apply(clean_role)
    # #df_4['from_role_clean'] = df_4['from_role'].progress_apply(clean_role)
    return


@app.cell(hide_code=True)
def _():
    # # cleaned_thinking, cleaned_title = [], []

    # cleaned_title = []
    # batch_size = 20


    # for i in tqdm(range(0, len(df_2_test), batch_size), desc="Cleaning batches"):
    #     batch = df_2_test["from_role"].iloc[i:i+batch_size].tolist()

    #     cleaned_title += clean_roles_batch(batch)


    # df_2_test["from_role_cleaned"] = cleaned_title
    return


@app.cell
def _():
    # df_2_test
    return


@app.cell
def _():
    # # df_1.to_csv('job_to_job_transition_1_10000.csv', index=False)
    # # df_2.to_csv('cleaned_datasets/job_to_job_transition_10000_20000.csv', index=False)
    # df_3.to_csv('cleaned_datasets/job_to_job_transition_20000_30000.csv', index=False)
    return


@app.cell
def _():
    # print("🧹 Cleaning to_role...")
    # df["to_role_cleaned"] = df["to_role"].progress_apply(clean_role)

    # # Save result
    # df.to_csv("roles_cleaned.csv", index=False)
    # print("\n✅ Cleaning complete! Saved as roles_cleaned.csv")
    return


if __name__ == "__main__":
    app.run()
