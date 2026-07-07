import json
import hashlib
import uuid
import random
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from openai import AsyncOpenAI
from recruiter.core.config import settings
from recruiter.core.database import db
from recruiter.services.embeddings import embedding_service

logger = logging.getLogger("interview_engine")

CACHE_SIMILARITY_THRESHOLD = 0.90

class InterviewSession:
    def __init__(self, session_id: str, candidate_name: str, resume_id: str,
                 level: str, batches: dict, jd_text: str = "",
                 question_type: str = ""):
        self.session_id = session_id
        self.candidate_name = candidate_name
        self.resume_id = resume_id
        self.level = level
        self.jd_text = jd_text
        self.resume_text = ""
        self.batches = batches
        self.max_batch_generated = len(batches)
        self.question_type = question_type
        self.created_at = datetime.now().isoformat()

class InterviewEngine:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def _compute_jd_hash(self, jd_text: str) -> str:
        return hashlib.sha256(jd_text.encode("utf-8")).hexdigest()

    async def _check_cache(self, jd_text: str, level: str) -> Optional[list]:
        jd_hash = await self._compute_jd_hash(jd_text)
        exact = await db.db["interview_cache"].find_one({
            "jd_hash": jd_hash,
            "level": level
        })
        if exact:
            logger.info(f"Cache HIT (exact hash) for level={level}")
            return exact["questions"]
        jd_embedding = await embedding_service.generate_embedding(jd_text)
        all_cached = await db.db["interview_cache"].find({
            "level": level
        }).to_list(length=500)
        for cached in all_cached:
            cached_emb = cached.get("jd_embedding")
            if not cached_emb:
                continue
            sim = self._cosine_similarity(jd_embedding, cached_emb)
            if sim >= CACHE_SIMILARITY_THRESHOLD:
                logger.info(f"Cache HIT (semantic, sim={sim:.3f}) for level={level}")
                return cached["questions"]
        return None

    async def _save_cache(self, jd_text: str, level: str, questions: list):
        jd_hash = await self._compute_jd_hash(jd_text)
        jd_embedding = await embedding_service.generate_embedding(jd_text)
        existing = await db.db["interview_cache"].find_one({
            "jd_hash": jd_hash,
            "level": level
        })
        if existing:
            await db.db["interview_cache"].update_one(
                {"_id": existing["_id"]},
                {"$set": {"questions": questions, "updated_at": datetime.now().isoformat()}}
            )
            return
        await db.db["interview_cache"].insert_one({
            "jd_hash": jd_hash,
            "jd_embedding": jd_embedding,
            "level": level,
            "questions": questions,
            "created_at": datetime.now().isoformat()
        })

    def _cosine_similarity(self, a: list, b: list) -> float:
        import numpy as np
        a = np.array(a)
        b = np.array(b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    async def _call_openai_questions(self, jd_text: str, resume_text: str,
                                      level: str, count: int = 5,
                                      avoid_skills: list = None,
                                      question_type: str = "") -> list:
        level_descriptions = {
            "fresher": "Fresher (0-1 years) — focus on fundamentals, academic projects, internships, learning ability, basic problem-solving. Do NOT expect industry experience.",
            "intermediate": "Intermediate (2-4 years) — focus on real projects, debugging, APIs, databases, optimization, practical experience, production scenarios.",
            "expert": "Experienced (5+ years) — focus on architecture, scalability, production issues, system design, leadership, trade-offs, decision-making."
        }
        level_desc = level_descriptions.get(level, level_descriptions["intermediate"])
        avoid_text = ""
        if avoid_skills:
            avoid_text = f"\n\nIMPORTANT: Do NOT generate questions about these already-covered skills/topics: {', '.join(avoid_skills)}. Cover fresh topics from the JD and Resume."
        type_instruction = ""
        if question_type:
            type_instruction = f"\n- ONLY generate questions of type: \"{question_type}\". Do NOT generate any other question types."
        prompt = f"""You are an expert interviewer. Generate interview questions based on the Job Description and Candidate Resume below.

TODAY'S DATE: {datetime.now().strftime("%B %Y")}

CANDIDATE LEVEL: {level_desc}

ROLE (from Job Title): Extract the core role from the job title (e.g., "Junior Software Engineer" → Software Engineer, "Registered Nurse" → Nurse) and test foundational knowledge expected for that role.

Guidelines:
- Questions must match the experience level requested
- Extract the key skills, tools, technologies, and requirements from the Job Description — test those specifically
- Ask role-specific foundational questions (e.g., Software Engineer → coding, databases, system design; Nurse → patient care, procedures, protocols; Data Analyst → SQL, statistics, visualization)
- Make questions realistic, practical, and directly tied to the JD — do NOT ask generic or surface-level questions
- Ensure every question maps to a specific skill or requirement mentioned in the JD or essential for the role
- Each question should test a different skill or topic{avoid_text}{type_instruction}

JOB DESCRIPTION (extract skills, tools, domain knowledge from this):
{jd_text[:4000]}

CANDIDATE RESUME (for reference):
{resume_text[:4000]}

Return ONLY a JSON array of question objects with these keys:
- id: unique string like "q1", "q2"
- type: question type ("technical", "behavioral", "project", "resume", "coding", "system_design", "database")
- skill: the primary skill/topic this question targets (must map to JD requirement or role foundation)
- question: the actual interview question text
- difficulty: "easy", "medium", or "hard" relative to the candidate level

Generate exactly {count} relevant questions."""
        try:
            response = await self.client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            questions = data if isinstance(data, list) else data.get("questions", [])
            for q in questions:
                if "id" not in q:
                    q["id"] = f"q{uuid.uuid4().hex[:4]}"
                q.setdefault("answer", None)
                q.setdefault("follow_ups", [])
            return questions[:count]
        except Exception as e:
            logger.error(f"OpenAI question generation failed: {e}")
            raise

    async def create_session(self, jd_text: str, resume_text: str,
                              level: str, candidate_name: str = "Candidate",
                              question_type: str = "") -> InterviewSession:
        logger.info(f"Creating session for {candidate_name}, level={level}")
        session_id = str(uuid.uuid4())
        return InterviewSession(
            session_id=session_id,
            candidate_name=candidate_name,
            resume_id="",
            level=level,
            batches={},
            jd_text=jd_text,
            question_type=question_type
        )

    async def get_batch(self, session: InterviewSession, resume_text: str, batch_num: int, question_type: str = "") -> list:
        if batch_num in session.batches:
            return session.batches[batch_num]
        already_used_skills = []
        for b in session.batches.values():
            for q in b:
                if q.get("skill"):
                    already_used_skills.append(q["skill"])
        new_questions = await self._call_openai_questions(
            session.jd_text, resume_text, session.level,
            count=5, avoid_skills=already_used_skills,
            question_type=question_type
        )
        for q in new_questions:
            q["batch"] = batch_num
        session.batches[batch_num] = new_questions
        session.max_batch_generated = max(session.max_batch_generated, batch_num)
        return new_questions

    async def generate_answer(self, question: dict, resume_text: str, jd_text: str) -> str:
        prompt = f"""You are a senior interview coach. Provide a detailed, practical answer to this interview question.

QUESTION: {question.get("question")}
SKILL/TOPIC: {question.get("skill")}
QUESTION TYPE: {question.get("type")}
DIFFICULTY: {question.get("difficulty")}

CANDIDATE RESUME (for context on their actual experience):
{resume_text[:3000]}

JOB DESCRIPTION (for role context):
{jd_text[:2000]}

Provide a model answer that:
1. Is practical and realistic (not textbook)
2. Includes code examples if technical
3. References real-world scenarios
4. Matches the question difficulty level
5. Is structured but conversational

Keep the answer concise but comprehensive (2-4 paragraphs). If the question expects code, include a short code example.

Return ONLY a JSON object with the key "answer" containing the model answer."""
        try:
            response = await self.client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("answer", data.get("content", data.get("text", "")))
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer at this time."

    async def generate_follow_up(self, question: dict, answer: str,
                                  resume_text: str, jd_text: str) -> str:
        prompt = f"""You are an interviewer conducting a live interview. The candidate just answered your question, now ask a natural follow-up question.

ORIGINAL QUESTION: {question.get("question")}
CANDIDATE'S ANSWER: {answer[:2000]}
CANDIDATE RESUME: {resume_text[:2000]}
JOB DESCRIPTION: {jd_text[:2000]}

Generate ONE follow-up question that:
1. Digs deeper into the same topic
2. Challenges the candidate's understanding
3. Is conversational and natural
4. Expects a practical example

Return a JSON object with: {{"question": "the follow-up question text", "skill": "the skill/topic"}}"""
        try:
            response = await self.client.chat.completions.create(
                model=settings.AI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("question", "")
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return ""

interview_engine = InterviewEngine()
