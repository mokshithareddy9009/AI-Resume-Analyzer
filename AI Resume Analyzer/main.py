from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re

app = Flask(__name__)

# ======================
# CONFIGURATION
# ======================

app.secret_key = "supersecretkey"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ======================
# DATABASE MODELS
# ======================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)


class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()

# ======================
# SKILL DATABASE
# ======================

SKILLS = [
    "python", "java", "c++", "flask", "django", "react", "javascript",
    "html", "css", "sql", "mongodb", "machine learning", "deep learning",
    "tensorflow", "pytorch", "data analysis", "pandas", "numpy",
    "scikit-learn", "git", "aws", "docker", "kubernetes",
    "node.js", "express", "typescript", "linux", "cybersecurity",
    "networking", "ci/cd", "graphql"
]

# ======================
# SYNONYMS
# ======================

SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "js": "javascript",
    "py": "python",
    "tf": "tensorflow"
}

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    words = text.split()
    normalized_words = []

    for word in words:
        if word in SYNONYMS:
            normalized_words.append(SYNONYMS[word])
        else:
            normalized_words.append(word)

    return " ".join(normalized_words)

# ======================
# HELPER FUNCTIONS
# ======================

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def extract_skills(text):
    text = normalize_text(text)
    return [skill for skill in SKILLS if skill in text]

# ======================
# HOME / ANALYZER
# ======================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    score = None
    resume_skills = []
    missing_skills = []
    job_skills = []
    feedback = ""

    if request.method == "POST":
        job_description = request.form.get("job_description")
        resume_text = request.form.get("resume")
        resume = ""

        # PDF upload
        if "resume_file" in request.files:
            file = request.files["resume_file"]
            if file and file.filename != "":
                resume = extract_text_from_pdf(file)

        # Text paste
        if not resume and resume_text:
            resume = resume_text

        if not resume:
            flash("⚠️ Please upload a resume PDF or paste resume text.")
            return redirect(url_for("index"))

        # Normalize both texts
        resume = normalize_text(resume)
        job_description = normalize_text(job_description)

        # -------------------------
        # TEXT SIMILARITY (40%)
        # -------------------------
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume, job_description])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])
        text_score = similarity[0][0] * 100

        # -------------------------
        # SKILL MATCH SCORE (60%)
        # -------------------------
        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job_description)

        matched_skills = [skill for skill in job_skills if skill in resume_skills]
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]

        if len(job_skills) > 0:
            skill_score = (len(matched_skills) / len(job_skills)) * 100
        else:
            skill_score = 0

        # -------------------------
        # FINAL HYBRID SCORE
        # -------------------------
        final_score = (0.6 * skill_score) + (0.4 * text_score)
        score = round(final_score, 2)

        # Feedback
        if score < 40:
            feedback = "⚠️ Your resume is poorly aligned with this job."
        elif 40 <= score < 70:
            feedback = "👍 You are partially aligned. Improve missing skills."
        else:
            feedback = "🎯 Strong match! Your resume aligns well."

        # Save to DB
        new_resume = Resume(score=score, user_id=current_user.id)
        db.session.add(new_resume)
        db.session.commit()

    return render_template("index.html",
                           score=score,
                           resume_skills=resume_skills,
                           missing_skills=missing_skills,
                           job_skills=job_skills,
                           feedback=feedback)

# ======================
# REGISTER
# ======================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("⚠️ Username already exists.")
            return redirect(url_for("register"))

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("✅ Registration successful! Please login.")
        return redirect(url_for("login"))

    return render_template("register.html")

# ======================
# LOGIN
# ======================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = User.query.filter_by(username=username).first()

        if not user:
            flash("⚠️ Account does not exist.")
            return redirect(url_for("login"))

        if user.password != password:
            flash("❌ Incorrect password.")
            return redirect(url_for("login"))

        login_user(user)
        flash("✅ Login successful!")
        return redirect(url_for("index"))

    return render_template("login.html")

# ======================
# DASHBOARD
# ======================

@app.route("/dashboard")
@login_required
def dashboard():
    resumes = Resume.query.filter_by(user_id=current_user.id).all()
    return render_template("dashboard.html", resumes=resumes)

# ======================
# LOGOUT
# ======================

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("👋 Logged out successfully.")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)