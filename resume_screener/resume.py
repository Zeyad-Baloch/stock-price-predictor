import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import io


class ResumeScreener:
    def __init__(self):

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.ml_screener = None

        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'react', 'angular', 'node.js', 'sql', 'html', 'css'],
            'tools': ['git', 'docker', 'kubernetes', 'aws', 'azure', 'jenkins', 'jira'],
            'soft_skills': ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical'],
            'frameworks': ['django', 'flask', 'spring', 'express', 'tensorflow', 'pytorch']
        }


        self.all_skills = []
        for category in self.skill_categories.values():
            self.all_skills.extend(category)

    def attach_ml_screener(self, ml_screener):
        """Attach trained ML model to use for scoring"""
        self.ml_screener = ml_screener

    def extract_text_from_pdf(self, pdf_content):
        """Extract text from PDF """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except:
            return ""


    def clean_text(self, text):
        """Clean and normalize text"""

        text = text.lower()

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,@-]', '', text)

        noise_words = ['resume', 'cv', 'curriculum vitae', 'page', 'confidential']
        for word in noise_words:
            text = text.replace(word, '')

        return text.strip()

    def extract_skills(self, text):
        """Extract skills from text"""
        text = self.clean_text(text)
        found_skills = []

        for skill in self.all_skills:

            skill_pattern = re.sub(r'[^a-z0-9]', r'[^a-z0-9]*', skill.lower())
            if re.search(r'\b' + skill_pattern + r'\b', text):
                found_skills.append(skill)

        return list(set(found_skills))

    def extract_experience_years(self, text):
        """Extract years of experience from text"""
        text = self.clean_text(text)


        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\s*(?:-\s*\d+)?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?'
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))

        return 0

    def extract_education_level(self, text):
        """Extract education level from text"""
        text = self.clean_text(text)

        education_levels = {
            'phd': ['phd', 'ph.d', 'doctorate', 'doctoral'],
            'masters': ['masters', 'master', 'msc', 'm.sc', 'mba', 'm.b.a'],
            'bachelors': ['bachelors', 'bachelor', 'bsc', 'b.sc', 'ba', 'b.a', 'be', 'b.e'],
            'diploma': ['diploma', 'certificate']
        }

        for level, keywords in education_levels.items():
            for keyword in keywords:
                if keyword in text:
                    return level

        return 'unknown'

    def parse_resume(self, resume_text):
        """Parse resume and extract structured information"""
        cleaned_text = self.clean_text(resume_text)

        return {
            'raw_text': resume_text,
            'cleaned_text': cleaned_text,
            'skills': self.extract_skills(cleaned_text),
            'experience_years': self.extract_experience_years(cleaned_text),
            'education_level': self.extract_education_level(cleaned_text)
        }

    def parse_job_posting(self, job_text):
        """Parse job posting and extract requirements"""
        cleaned_text = self.clean_text(job_text)

        return {
            'raw_text': job_text,
            'cleaned_text': cleaned_text,
            'required_skills': self.extract_skills(cleaned_text),
            'required_experience': self.extract_experience_years(cleaned_text),
            'required_education': self.extract_education_level(cleaned_text)
        }

    def calculate_skill_match_score(self, resume_skills, job_skills):
        """Calculate skill match score between resume and job"""
        if not job_skills:
            return 0.0
        matched_skills = set(resume_skills).intersection(set(job_skills))
        return len(matched_skills) / len(job_skills)

    def calculate_experience_match_score(self, resume_exp, job_exp):
        """Calculate experience match score"""
        if job_exp == 0:
            return 1.0

        if resume_exp >= job_exp:
            return 1.0
        elif resume_exp == 0:
            return 0.0
        else:
            return resume_exp / job_exp

    def calculate_education_match_score(self, resume_edu, job_edu):
        """Calculate education match score"""
        education_hierarchy = {
            'phd': 4,
            'masters': 3,
            'bachelors': 2,
            'diploma': 1,
            'unknown': 0
        }

        resume_level = education_hierarchy.get(resume_edu, 0)
        job_level = education_hierarchy.get(job_edu, 0)

        if job_level == 0:
            return 1.0

        return min(resume_level / job_level, 1.0)

    def calculate_text_similarity(self, resume_text, job_text):
        """Calculate semantic similarity using sentence transformers"""

        resume_embedding = self.model.encode([resume_text])
        job_embedding = self.model.encode([job_text])

        similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
        return similarity

    def calculate_match_score(self, resume, job):
        """Calculate overall match score with weighted factors"""

        skill_score = self.calculate_skill_match_score(resume['skills'], job['required_skills'])
        experience_score = self.calculate_experience_match_score(resume['experience_years'], job['required_experience'])
        education_score = self.calculate_education_match_score(resume['education_level'], job['required_education'])
        text_similarity = self.calculate_text_similarity(resume['cleaned_text'], job['cleaned_text'])

        weights = {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.2,
            'text_similarity': 0.1
        }

        # will use ML model if available and trained
        if hasattr(self, 'ml_screener') and self.ml_screener and self.ml_screener.is_trained:
            overall_score = self.ml_screener.predict_match_score(resume, job)
        else:
            # will use basic scoring if model not trained
            overall_score = (
                    skill_score * weights['skills'] +
                    experience_score * weights['experience'] +
                    education_score * weights['education'] +
                    text_similarity * weights['text_similarity']
            )

        return {
            'overall_score': overall_score,
            'skill_score': skill_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'text_similarity': text_similarity,
            'matched_skills': list(set(resume['skills']).intersection(set(job['required_skills'])))
        }

    def rank_resumes(self, resumes, job_posting):
        """Rank multiple resumes against a job posting"""
        job_parsed = self.parse_job_posting(job_posting)
        results = []

        for i, resume in enumerate(resumes):
            if isinstance(resume, str):
                resume_parsed = self.parse_resume(resume)
            else:
                resume_parsed = resume

            match_results = self.calculate_match_score(resume_parsed, job_parsed)

            results.append({
                'resume_id': i,
                'resume_data': resume_parsed,
                'match_results': match_results
            })


        results.sort(key=lambda x: x['match_results']['overall_score'], reverse=True)

        return results

    def display_results(self, ranked_results):
        """Display ranking results in a readable format"""
        print("\n" + "=" * 80)
        print("RESUME RANKING RESULTS")
        print("=" * 80)

        for i, result in enumerate(ranked_results, 1):
            match = result['match_results']
            resume = result['resume_data']

            print(f"\nRANK {i} - Resume ID: {result['resume_id']}")
            print(f"Overall Score: {match['overall_score']:.2f}")
            print(f"Skills Match: {match['skill_score']:.2f} | Experience Match: {match['experience_score']:.2f}")
            print(f"Education Match: {match['education_score']:.2f} | Text Similarity: {match['text_similarity']:.2f}")
            print(f"Matched Skills: {', '.join(match['matched_skills']) if match['matched_skills'] else 'None'}")
            print(f"Candidate Skills: {', '.join(resume['skills'][:5])}{'...' if len(resume['skills']) > 5 else ''}")
            print(f"Experience: {resume['experience_years']} years | Education: {resume['education_level']}")
            print("-" * 80)


# testing
def test_resume_screener():

    screener = ResumeScreener()

    job_posting = """
    Senior Python Developer

    We are looking for a Senior Python Developer with 5+ years of experience.

    Required Skills:
    - Python programming
    - Django or Flask framework
    - SQL databases
    - Git version control
    - AWS cloud services
    - React.js for frontend

    Requirements:
    - Bachelor's degree in Computer Science or related field
    - 5+ years of professional software development experience
    - Strong problem-solving skills
    - Experience with agile development
    """

    resumes = [
        """
        John Doe
        Software Engineer

        Experience: 6 years of software development

        Skills: Python, Django, PostgreSQL, JavaScript, React, AWS, Git, Docker

        Education: Bachelor's degree in Computer Science

        Previous role: Senior Python Developer at TechCorp
        - Developed web applications using Django
        - Worked with PostgreSQL databases
        - Deployed applications on AWS
        """,

        """
        Jane Smith
        Junior Developer

        Experience: 2 years programming experience

        Skills: Java, Spring Boot, MySQL, HTML, CSS

        Education: Bachelor's degree in Information Technology

        Previous role: Junior Java Developer
        - Built REST APIs using Spring Boot
        - Worked with MySQL databases
        - Basic frontend development
        """,

        """
        Alex Johnson
        Full Stack Developer

        Experience: 4 years of web development

        Skills: Python, Flask, MongoDB, React.js, Node.js, AWS, Git

        Education: Master's degree in Computer Science

        Previous role: Full Stack Developer at StartupXYZ
        - Developed REST APIs using Flask
        - Built responsive frontends with React
        - Managed AWS infrastructure
        """
    ]


    print("Testing Resume Screener...")
    print("Job Posting:", job_posting)
    print(f"Number of resumes to rank: {len(resumes)}")

    results = screener.rank_resumes(resumes, job_posting)
    screener.display_results(results)

    return screener, results

if __name__ == "__main__":
    screener, results = test_resume_screener()