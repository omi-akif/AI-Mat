from pydantic import BaseModel
from typing import List, Optional, Union

class CandidateInput(BaseModel):
    id: int
    gender: Optional[str] = "unidentified"
    martial_status: Optional[str] = "unidentified"
    searching_for_job_status: Optional[str] = "unidentified"
    district_name: Optional[str] = "unidentified"
    salary_currency_name: Optional[str] = "unidentified"
    salary_type_name: Optional[str] = "unidentified"
    candidate_type: Optional[str] = "unidentified"
    
    degree_institutes: List[str] = []
    degree_names: List[str] = []
    degree_majors: List[str] = []
    preferredJobCategory_department_names: List[str] = []
    preferredJobCategory_industry_names: List[str] = []
    
    present_salary: Optional[float] = 0.0
    expected_salary: Optional[float] = 0.0
    total_experience: Optional[float] = 0.0
    age: Optional[int] = 0
    
    level_name: Optional[str] = "unidentified"
    qualification_name: Optional[str] = "unidentified"
    
    candidate_latest_resume_text: Optional[str] = ""
    
    skills_names: List[str] = []
    skills_year_of_experiences: List[float] = []
    candidate_experience_roles: List[str] = []
    candidate_experience_role_duration: List[float] = []
    profession: Optional[str] = ""

class JobInput(BaseModel):
    post_id: int
    job_gender: Optional[str] = "unidentified"
    position_name: Optional[str] = "unidentified"
    job_district_name: Optional[str] = "unidentified"
    job_type_name: Optional[str] = "unidentified"
    salary_currency: Optional[str] = "unidentified"
    job_salary_type: Optional[str] = "unidentified"
    industry_name: Optional[str] = "unidentified"
    department_name: Optional[str] = "unidentified"
    
    minimum_salary: Optional[float] = 0.0
    maximum_salary: Optional[float] = 0.0
    age_from: Optional[int] = 0
    age_to: Optional[int] = 0
    job_experience: Optional[float] = 0.0
    minimum_experience: Optional[float] = 0.0
    maximum_experience: Optional[float] = 0.0
    
    job_level_name: Optional[str] = "unidentified"
    job_qualification_name: Optional[str] = "unidentified"
    qualification_prefer_name: Optional[str] = "unidentified"
    
    job_description: Optional[str] = ""
    job_requirement: Optional[str] = ""
    
    job_skill_name: List[str] = []
    job_skill_experience: List[float] = []
    job_title: Optional[str] = ""
    negotiable: Optional[int] = 0
