import json
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List
from recruiter.core.config import settings

class SeekerFeedback(BaseModel):
    missing_keywords: List[str]
    optimization_tips: List[str]
    is_compatible: bool
    final_verdict: str

class SeekerService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def get_resume_feedback(self, resume_text: str, job_description: str) -> SeekerFeedback:
        """Generates ATS feedback and optimization tips for the seeker."""
        prompt = f"""
        Compare the following resume text against the job description provided.
        Identify missing keywords, provide specific optimization tips to improve the ATS score, 
        and give a final verdict on compatibility.
        
        Return ONLY a JSON object with these EXACT keys (all lowercase):
        - missing_keywords (list of strings)
        - optimization_tips (list of strings)
        - is_compatible (boolean)
        - final_verdict (string, max 200 characters)

        Resume:
        {resume_text[:3000]}

        Job Description:
        {job_description[:3000]}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            feedback_data = json.loads(content)
            
            return SeekerFeedback(**feedback_data)
        except Exception as e:
            print(f"Error getting seeker feedback: {e}")
            return SeekerFeedback(
                missing_keywords=[],
                optimization_tips=["Could not generate tips at this moment."],
                is_compatible=False,
                final_verdict="Error during analysis."
            )

seeker_service = SeekerService()
