"""
AI Resume Analyzer — Flask Application
=======================================
Serves the web UI and exposes a /analyze endpoint.
"""

import os
from flask import Flask, render_template, request, jsonify
import PyPDF2
from analyzer import full_analysis

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


def extract_text_from_pdf(file_storage) -> str:
    """Extract text from an uploaded PDF file."""
    try:
        reader = PyPDF2.PdfReader(file_storage)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    resume_file = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()
    role = request.form.get("role", "").strip() or None

    if not resume_file:
        return jsonify({"error": "Please upload a resume PDF."}), 400
    if not job_description:
        return jsonify({"error": "Please provide a job description."}), 400

    resume_text = extract_text_from_pdf(resume_file)
    if not resume_text:
        return jsonify({"error": "Could not extract text from PDF. Please try another file."}), 400

    results = full_analysis(resume_text, job_description, role)
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)