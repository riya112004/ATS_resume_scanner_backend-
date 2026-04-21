from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ContactInfo(BaseModel):
    name: Optional[str] = "Unknown"
    email: Optional[str] = None
    phone: Optional[str] = None
    links: List[str] = Field(default_factory=list)

class WorkExperience(BaseModel):
    company: Optional[str] = "Unknown"
    role: Optional[str] = "Unknown"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = ""

class Project(BaseModel):
    title: Optional[str] = "Unknown Project"
    description: Optional[str] = ""

class ParsedResume(BaseModel):
    contact: ContactInfo
    skills: List[str] = Field(default_factory=list)
    experience_years: float = 0.0
    work_history: List[WorkExperience] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: str = ""
    parsing_warnings: List[str] = Field(default_factory=list)

class ParsedJD(BaseModel):
    role: Optional[str] = "Unknown"
    must_have_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    min_experience: float = 0.0
    education_requirements: List[str] = Field(default_factory=list)
    domain_keywords: List[str] = Field(default_factory=list)
    raw_text: str = ""

class ScoreBreakdown(BaseModel):
    skills_match: float = 0.0
    experience_relevance: float = 0.0
    role_alignment: float = 0.0
    project_relevance: Optional[float] = None
    education_certifications: float = 0.0
    keyword_coverage: float = 0.0
    formatting_readability: float = 0.0

class AnalysisResult(BaseModel):
    overall_ats_score: float = 0.0
    confidence_score: float = 0.0
    breakdown: ScoreBreakdown
    matched_skills: List[str] = Field(default_factory=list)
    missing_critical_skills: List[str] = Field(default_factory=list)
    missing_preferred_skills: List[str] = Field(default_factory=list)
    weak_areas: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    verdict: str = ""
    parsed_resume: Optional[Dict[str, Any]] = None
    parsed_jd: Optional[Dict[str, Any]] = None
