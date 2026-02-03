import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_keywords(text, vectorizer, top_n=10):
    """Extract top keywords from text using TF-IDF."""
    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    top_indices = scores.argsort()[-top_n:][::-1]
    return [(feature_names[i], round(scores[i], 3)) for i in top_indices if scores[i] > 0]


def calculate_alignment_score(resume_text, job_description):
    """Calculate how well a resume matches a job description using TF-IDF and cosine similarity."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(score * 100, 2), vectorizer


def find_missing_keywords(resume_text, job_description, vectorizer):
    """Find keywords in the job description that are missing from the resume."""
    job_keywords = extract_keywords(job_description, vectorizer)
    resume_lower = resume_text.lower()
    missing = [kw for kw, score in job_keywords if kw.lower() not in resume_lower]
    return missing


def generate_feedback(resume_text, job_description):
    """Generate full feedback report for a resume against a job description."""
    score, vectorizer = calculate_alignment_score(resume_text, job_description)
    missing = find_missing_keywords(resume_text, job_description, vectorizer)
    resume_keywords = extract_keywords(resume_text, vectorizer)

    feedback = {
        "alignment_score": score,
        "resume_keywords": resume_keywords,
        "missing_keywords": missing,
        "recommendation": (
            "Strong match — focus on adding the missing keywords above."
            if score > 60
            else "Needs improvement — consider rewriting to include more relevant keywords."
        ),
    }
    return feedback


def print_feedback(feedback):
    """Print the feedback report in a readable format."""
    print("\n" + "=" * 50)
    print("  RESUME FEEDBACK REPORT")
    print("=" * 50)
    print(f"\n  Alignment Score: {feedback['alignment_score']}%")
    print(f"\n  Recommendation: {feedback['recommendation']}")

    print("\n  --- Your Top Keywords ---")
    for kw, score in feedback["resume_keywords"]:
        print(f"    • {kw} ({score})")

    print("\n  --- Missing Keywords (from Job Description) ---")
    if feedback["missing_keywords"]:
        for kw in feedback["missing_keywords"]:
            print(f"    • {kw}")
    else:
        print("    None — great match!")

    print("\n" + "=" * 50 + "\n")


# ─── Sample Data ───
sample_resume = """
Bhavya Devireddy
AI & Data Science student with hands-on experience in machine learning and Python.
Skills: Python, Scikit-learn, Pandas, NumPy, TF-IDF, Feature Engineering, Classification, Git.
Developed a crop recommendation system using Random Forest achieving 90% accuracy.
Built a resume shortlisting tool using TF-IDF vectorization.
Solved 100+ LeetCode problems. 96th percentile in JEE Mains.
"""

sample_job_description = """
We are hiring a Junior Data Scientist. The ideal candidate should have experience with
Python, machine learning, and data analysis. Must be comfortable with Pandas, Scikit-learn,
and SQL. Experience with NLP and deep learning is a plus. Strong problem-solving skills
and a passion for building scalable data pipelines are required. Familiarity with cloud
platforms like AWS is preferred.
"""


# ─── Run ───
if __name__ == "__main__":
    print("\nAnalyzing resume against job description...")
    feedback = generate_feedback(sample_resume, sample_job_description)
    print_feedback(feedback)
