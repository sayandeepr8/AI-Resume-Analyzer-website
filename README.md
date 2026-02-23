# 🚀 AI Resume Analyzer & ATS Match Scorer

An AI-powered web application that analyzes your resume against job descriptions and provides an **ATS Compatibility Score** along with actionable insights to optimize your resume for Applicant Tracking Systems.

Built with **Flask**, **NLP (TF-IDF + Cosine Similarity)**, and a premium dark-mode UI.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| ✅ **ATS Compatibility Score** | Quantifies how well your resume matches a given job description (0–100) |
| 🧠 **Skill Gap Analysis** | Identifies matched, missing, and bonus skills vs job requirements |
| 📊 **Section-Wise Scoring** | Separately evaluates Skills, Experience, Projects, and Education |
| 🎯 **Role-Based Optimization** | Custom analysis for Data Scientist, ML Engineer, Backend/Frontend/Full Stack Dev, DevOps |
| 📈 **Keyword Density Analysis** | Detects missing, underused, and optimally used keywords for ATS ranking |
| 🔁 **Before vs After View** | Shows estimated score improvement with optimization tips |
| 🧑‍💼 **Recruiter View Mode** | Simulates recruiter decisions with shortlist/reject insights and red flags |
| 🔍 **Explainable AI Panel** | Breaks down exactly why your resume received a particular score |

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **NLP:** Scikit-learn (TF-IDF Vectorizer, Cosine Similarity), NLTK
- **PDF Parsing:** PyPDF2
- **Frontend:** HTML5, CSS3 (Glassmorphism, Dark Mode), Vanilla JavaScript
- **Typography:** Google Fonts (Inter)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/Saptomita/Resume-Analyzer.git
cd Resume-Analyzer

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open your browser and navigate to **http://127.0.0.1:5000**

---

## 📖 How to Use

1. **Upload** your resume (PDF format)
2. **Paste** the job description
3. **Select** a target role (optional — enhances analysis)
4. **Click** "Analyze Match"
5. **Review** your ATS score, skill gaps, keyword density, and optimization tips

---

## 📁 Project Structure

```
Resume-Analyzer/
├── app.py                 # Flask server (routes & PDF handling)
├── analyzer.py            # NLP analysis engine (12 functions)
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Main UI template
├── static/
│   ├── style.css          # Premium dark-mode stylesheet
│   └── script.js          # Frontend logic & animations
└── .gitignore
```

---

## 📸 Screenshot

![Resume Analyzer UI](https://raw.githubusercontent.com/Saptomita/Resume-Analyzer/main/screenshots/ui.png)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
