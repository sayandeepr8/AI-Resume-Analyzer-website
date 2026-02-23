"""
AI Resume Analyzer — NLP Analysis Engine
=========================================
Provides TF-IDF + cosine-similarity based resume analysis with:
  - ATS compatibility scoring
  - Skill gap analysis
  - Section-wise scoring
  - Role-based keyword optimization
  - Keyword density analysis
  - Before vs After optimization estimation
  - Recruiter view simulation
  - Explainable AI breakdown
"""

import re
import math
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------
for resource in ("punkt_tab", "stopwords", "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else resource)
    except LookupError:
        nltk.download(resource, quiet=True)

# ---------------------------------------------------------------------------
# Role-specific keyword banks
# ---------------------------------------------------------------------------
ROLE_KEYWORDS: dict[str, list[str]] = {
    "Data Scientist": [
        "python", "r", "machine learning", "deep learning", "statistics",
        "tensorflow", "pytorch", "pandas", "numpy", "scikit-learn",
        "data visualization", "sql", "hypothesis testing", "regression",
        "classification", "nlp", "neural network", "feature engineering",
        "data mining", "jupyter", "matplotlib", "seaborn", "keras",
        "xgboost", "random forest", "clustering", "dimensionality reduction",
        "a/b testing", "big data", "spark",
    ],
    "ML Engineer": [
        "python", "machine learning", "deep learning", "tensorflow",
        "pytorch", "mlops", "docker", "kubernetes", "ci/cd",
        "model deployment", "aws", "gcp", "azure", "api",
        "microservices", "data pipeline", "feature store", "mlflow",
        "airflow", "sql", "nosql", "linux", "git", "flask",
        "fastapi", "model monitoring", "distributed systems",
        "model optimization", "onnx", "triton",
    ],
    "Backend Developer": [
        "python", "java", "node.js", "sql", "nosql", "rest api",
        "graphql", "docker", "kubernetes", "microservices", "aws",
        "gcp", "azure", "redis", "postgresql", "mongodb", "git",
        "ci/cd", "linux", "nginx", "rabbitmq", "kafka", "grpc",
        "flask", "django", "spring boot", "express", "testing",
        "security", "authentication", "caching",
    ],
    "Frontend Developer": [
        "javascript", "typescript", "react", "angular", "vue",
        "html", "css", "sass", "webpack", "vite", "responsive design",
        "accessibility", "performance", "testing", "jest", "cypress",
        "figma", "ui/ux", "graphql", "rest api", "git", "npm",
        "state management", "redux", "next.js", "tailwind",
        "web components", "pwa", "seo", "cross-browser",
    ],
    "Full Stack Developer": [
        "javascript", "typescript", "python", "react", "node.js",
        "sql", "nosql", "rest api", "graphql", "docker", "aws",
        "git", "ci/cd", "html", "css", "mongodb", "postgresql",
        "redis", "testing", "agile", "microservices", "linux",
        "webpack", "responsive design", "authentication", "security",
        "next.js", "express", "django",
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "ci/cd", "jenkins", "terraform",
        "ansible", "aws", "gcp", "azure", "linux", "bash",
        "python", "monitoring", "prometheus", "grafana", "elk",
        "git", "nginx", "security", "networking", "load balancing",
        "infrastructure as code", "helm", "argocd", "github actions",
        "cloud formation", "serverless", "lambda", "microservices",
    ],
}

# ---------------------------------------------------------------------------
# Section header patterns (case-insensitive)
# ---------------------------------------------------------------------------
SECTION_PATTERNS: dict[str, list[str]] = {
    "skills": [
        r"(?:technical\s+)?skills",
        r"core\s+competenc(?:ies|y)",
        r"technologies",
        r"tools?\s*(?:&|and)?\s*technologies",
        r"proficienc(?:ies|y)",
    ],
    "experience": [
        r"(?:work|professional)\s+experience",
        r"experience",
        r"employment\s+history",
        r"work\s+history",
    ],
    "projects": [
        r"projects",
        r"personal\s+projects",
        r"academic\s+projects",
        r"key\s+projects",
    ],
    "education": [
        r"education",
        r"academic\s+background",
        r"qualifications",
        r"degrees?",
    ],
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lower-case, strip non-alpha, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    stop = set(stopwords.words("english"))
    return " ".join(w for w in word_tokenize(text) if w not in stop)


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity (0-100) between two preprocessed texts."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vec = TfidfVectorizer()
    mtx = vec.fit_transform([text_a, text_b])
    return float(cosine_similarity(mtx[0:1], mtx[1:2])[0][0] * 100)

# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def extract_resume_sections(raw_text: str) -> dict[str, str]:
    """
    Extract Skills, Experience, Projects, Education sections from raw resume
    text using regex header matching.  Returns dict mapping section name to
    the text block that follows it.
    """
    lines = raw_text.split("\n")
    sections: dict[str, str] = {}
    current_section = None
    current_lines: list[str] = []

    combined_patterns: dict[str, re.Pattern] = {}
    for sec, pats in SECTION_PATTERNS.items():
        combined_patterns[sec] = re.compile(
            r"^\s*(?:" + "|".join(pats) + r")\s*:?\s*$", re.IGNORECASE
        )

    for line in lines:
        matched = False
        for sec, pat in combined_patterns.items():
            if pat.match(line.strip()):
                # save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_lines)
                current_section = sec
                current_lines = []
                matched = True
                break
        if not matched and current_section:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines)

    return sections

# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def calculate_ats_score(resume_text: str, job_description: str) -> float:
    """Overall ATS compatibility score (0-100)."""
    r = remove_stopwords(clean_text(resume_text))
    j = remove_stopwords(clean_text(job_description))
    return round(_tfidf_similarity(r, j), 2)


def score_sections(
    resume_text: str, job_description: str
) -> dict[str, float]:
    """Score each resume section against the full JD."""
    sections = extract_resume_sections(resume_text)
    jd_clean = remove_stopwords(clean_text(job_description))
    scores: dict[str, float] = {}
    for name in ("skills", "experience", "projects", "education"):
        sec_text = sections.get(name, "")
        if sec_text.strip():
            sec_clean = remove_stopwords(clean_text(sec_text))
            scores[name] = round(_tfidf_similarity(sec_clean, jd_clean), 2)
        else:
            scores[name] = 0.0
    return scores


def analyze_skills(
    resume_text: str, job_description: str, role: str | None = None
) -> dict:
    """
    Compare resume skills vs JD skills.
    Returns {matched, missing, extra, match_percent}.
    """
    jd_clean = clean_text(job_description)
    resume_clean = clean_text(resume_text)

    # Build skill pool from JD words + role keywords
    jd_tokens = set(word_tokenize(jd_clean))
    role_kw = set()
    if role and role in ROLE_KEYWORDS:
        role_kw = {k.lower() for k in ROLE_KEYWORDS[role]}

    # Candidate skill tokens from resume
    resume_tokens = set(word_tokenize(resume_clean))

    # Use role keywords + significant JD bigrams as the "required" set
    required: set[str] = set()
    # single-word skills from role bank that appear in JD
    if role_kw:
        for kw in role_kw:
            if kw in jd_clean:
                required.add(kw)
    # also pull all JD tokens > 2 chars that aren't pure stopwords
    stop = set(stopwords.words("english"))
    for t in jd_tokens:
        if len(t) > 2 and t not in stop:
            required.add(t)

    if not required:
        required = jd_tokens - stop

    matched = sorted(required & resume_tokens)
    missing = sorted(required - resume_tokens)

    # Extra: resume skills from role bank not in JD
    extra = sorted((role_kw & resume_tokens) - required) if role_kw else []

    total = len(required) if required else 1
    match_pct = round(len(matched) / total * 100, 1)

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_percent": match_pct,
    }


def keyword_density_analysis(
    resume_text: str, job_description: str, role: str | None = None
) -> list[dict]:
    """
    Classify important JD keywords by density in the resume.
    Each entry: {keyword, resume_count, status} where status is
    'optimal', 'underused', or 'missing'.
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)
    resume_words = word_tokenize(resume_clean)
    jd_words = word_tokenize(jd_clean)

    stop = set(stopwords.words("english"))

    # Important JD keywords (top by frequency)
    jd_freq = Counter(w for w in jd_words if w not in stop and len(w) > 2)
    top_keywords = [w for w, _ in jd_freq.most_common(25)]

    # Add role keywords present in JD
    if role and role in ROLE_KEYWORDS:
        for kw in ROLE_KEYWORDS[role]:
            kw_lower = kw.lower()
            if kw_lower in jd_clean and kw_lower not in top_keywords:
                top_keywords.append(kw_lower)

    resume_counter = Counter(resume_words)
    results = []
    for kw in top_keywords[:30]:
        count = resume_counter.get(kw, 0)
        # Also check multi-word
        if " " in kw:
            count = resume_clean.count(kw)
        if count == 0:
            status = "missing"
        elif count <= 1:
            status = "underused"
        else:
            status = "optimal"
        results.append({"keyword": kw, "resume_count": count, "status": status})

    return results


def generate_optimization_tips(
    skill_analysis: dict,
    section_scores: dict[str, float],
    keyword_analysis: list[dict],
) -> list[str]:
    """Produce actionable resume improvement suggestions."""
    tips: list[str] = []

    # Missing skills
    missing = skill_analysis.get("missing", [])
    if missing:
        sample = ", ".join(missing[:8])
        tips.append(
            f"Add these missing keywords to your resume: {sample}."
        )

    # Low section scores
    for sec, score in section_scores.items():
        if score < 20:
            tips.append(
                f"Your '{sec.title()}' section scored very low ({score}%). "
                f"Expand it with relevant details from the job description."
            )
        elif score < 45:
            tips.append(
                f"Consider strengthening your '{sec.title()}' section "
                f"(currently {score}%) with more JD-aligned content."
            )

    # Missing / underused keywords
    missing_kw = [e["keyword"] for e in keyword_analysis if e["status"] == "missing"]
    underused_kw = [e["keyword"] for e in keyword_analysis if e["status"] == "underused"]
    if missing_kw:
        tips.append(
            f"Incorporate these entirely missing keywords: "
            f"{', '.join(missing_kw[:6])}."
        )
    if underused_kw:
        tips.append(
            f"Use these keywords more frequently: "
            f"{', '.join(underused_kw[:6])}."
        )

    # Generic best-practice tips
    if not any(s > 60 for s in section_scores.values()):
        tips.append(
            "Tailor your resume more closely to this specific role — "
            "mirror the language and terminology of the job description."
        )

    tips.append(
        "Use strong action verbs (designed, implemented, optimized, "
        "deployed) at the start of bullet points."
    )
    tips.append(
        "Quantify achievements where possible (e.g., 'improved latency by 30%')."
    )

    return tips


def simulate_optimized_score(
    current_score: float, skill_analysis: dict, keyword_analysis: list[dict]
) -> float:
    """Estimate what the ATS score would be after applying optimizations."""
    missing_count = len(skill_analysis.get("missing", []))
    matched_count = len(skill_analysis.get("matched", []))
    total = missing_count + matched_count or 1

    # Fraction of gap that could be closed
    gap = 100 - current_score
    potential_gain = gap * (missing_count / total) * 0.7

    missing_kw = sum(1 for e in keyword_analysis if e["status"] == "missing")
    underused_kw = sum(1 for e in keyword_analysis if e["status"] == "underused")
    kw_bonus = min((missing_kw * 0.8 + underused_kw * 0.4), gap * 0.3)

    optimized = min(current_score + potential_gain + kw_bonus, 98.0)
    return round(max(optimized, current_score + 5), 2)


def recruiter_view(
    ats_score: float,
    skill_analysis: dict,
    section_scores: dict[str, float],
) -> dict:
    """Simulate a recruiter's evaluation."""
    red_flags: list[str] = []
    strengths: list[str] = []

    if ats_score < 35:
        decision = "Reject"
        band = "Low Match"
    elif ats_score < 55:
        decision = "Maybe — Needs Review"
        band = "Below Average"
    elif ats_score < 75:
        decision = "Shortlist — Worth Interviewing"
        band = "Good Match"
    else:
        decision = "Strong Shortlist"
        band = "Excellent Match"

    # Red flags
    match_pct = skill_analysis.get("match_percent", 0)
    if match_pct < 30:
        red_flags.append("Very low skill overlap with job requirements.")
    if section_scores.get("experience", 0) < 15:
        red_flags.append("Experience section appears weak or missing.")
    if section_scores.get("skills", 0) < 15:
        red_flags.append("Skills section appears weak or missing.")
    if section_scores.get("education", 0) < 10:
        red_flags.append("Education section may be missing or insufficient.")
    missing = skill_analysis.get("missing", [])
    if len(missing) > 10:
        red_flags.append(
            f"{len(missing)} required keywords are absent from the resume."
        )

    # Strengths
    if match_pct > 60:
        strengths.append("Strong skill alignment with the job description.")
    if section_scores.get("experience", 0) > 50:
        strengths.append("Experience section is well-tailored.")
    if section_scores.get("projects", 0) > 50:
        strengths.append("Relevant project work demonstrated.")
    if ats_score > 65:
        strengths.append("Overall resume content is well-matched.")

    if not red_flags:
        red_flags.append("No major red flags detected.")
    if not strengths:
        strengths.append("Resume could benefit from further tailoring.")

    return {
        "decision": decision,
        "band": band,
        "red_flags": red_flags,
        "strengths": strengths,
    }


def explain_score(
    ats_score: float,
    section_scores: dict[str, float],
    skill_analysis: dict,
    keyword_analysis: list[dict],
) -> list[dict]:
    """
    Break down the ATS score into understandable factors.
    Returns list of {factor, weight, value, explanation}.
    """
    explanations: list[dict] = []

    # 1. Overall TF-IDF similarity
    explanations.append({
        "factor": "Content Similarity (TF-IDF)",
        "weight": "40%",
        "value": f"{ats_score:.1f}%",
        "explanation": (
            "Measures how closely the overall language of your resume "
            "matches the job description using TF-IDF vectorization and "
            "cosine similarity. A higher score means your wording is "
            "well-aligned with what the ATS is scanning for."
        ),
    })

    # 2. Skill match
    match_pct = skill_analysis.get("match_percent", 0)
    explanations.append({
        "factor": "Skill Match Rate",
        "weight": "25%",
        "value": f"{match_pct:.1f}%",
        "explanation": (
            f"Out of the required skills identified in the job description, "
            f"{len(skill_analysis.get('matched', []))} were found in your resume "
            f"and {len(skill_analysis.get('missing', []))} are missing. "
            f"Adding missing skills is the fastest way to improve your score."
        ),
    })

    # 3. Section coverage
    avg_section = sum(section_scores.values()) / max(len(section_scores), 1)
    explanations.append({
        "factor": "Section Coverage",
        "weight": "20%",
        "value": f"{avg_section:.1f}%",
        "explanation": (
            "Evaluates how well each resume section (Skills, Experience, "
            "Projects, Education) individually matches the job description. "
            "Sections scoring below 30% should be expanded with relevant content."
        ),
    })

    # 4. Keyword optimization
    optimal_count = sum(1 for e in keyword_analysis if e["status"] == "optimal")
    total_kw = len(keyword_analysis) or 1
    kw_pct = round(optimal_count / total_kw * 100, 1)
    explanations.append({
        "factor": "Keyword Optimization",
        "weight": "15%",
        "value": f"{kw_pct:.1f}%",
        "explanation": (
            f"Of the {total_kw} important keywords from the job description, "
            f"{optimal_count} appear with optimal frequency in your resume. "
            f"Keywords marked 'missing' or 'underused' should be added or "
            f"used more frequently."
        ),
    })

    return explanations

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def full_analysis(
    resume_text: str, job_description: str, role: str | None = None
) -> dict:
    """Run every analysis and return a single result dict."""
    ats_score = calculate_ats_score(resume_text, job_description)
    sec_scores = score_sections(resume_text, job_description)
    skills = analyze_skills(resume_text, job_description, role)
    kw_density = keyword_density_analysis(resume_text, job_description, role)
    tips = generate_optimization_tips(skills, sec_scores, kw_density)
    optimized_score = simulate_optimized_score(ats_score, skills, kw_density)
    rec_view = recruiter_view(ats_score, skills, sec_scores)
    explanation = explain_score(ats_score, sec_scores, skills, kw_density)

    return {
        "ats_score": ats_score,
        "section_scores": sec_scores,
        "skill_analysis": skills,
        "keyword_density": kw_density,
        "optimization_tips": tips,
        "optimized_score": optimized_score,
        "recruiter_view": rec_view,
        "explanation": explanation,
        "role": role or "General",
    }
