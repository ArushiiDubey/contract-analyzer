 Contract Simplifier

**Project conducted:** October 2025

An AI-powered system that simplifies complex contractual language and identifies biased or unfair terms using Natural Language Processing (NLP) and Large Language Models (LLMs).

---

## Introduction

The **Contract Simplifier** project aims to make legal documents more transparent, readable, and fair.
Contracts often contain technical jargon and subtle biases that can affect interpretation.
This system uses text simplification and bias detection techniques to automatically generate a **plain-language version** of contracts and highlight **potentially biased terms**.

The main objectives are:

* To simplify complex legal text into easy-to-understand language.
* To detect and mitigate bias in legal and contractual wording.
* To ensure fair and inclusive representation in professional documentation.

  <img width="1788" height="821" alt="image" src="https://github.com/user-attachments/assets/3ea873fe-ea4c-477d-8f25-dbae5f67fbfd" />


---

## Dataset

### Data Source

The text data for testing was compiled from publicly available **sample contract templates** (e.g., service agreements, NDAs, employment contracts).
These were cleaned and processed to extract clause-level content for model evaluation.

### Bias Reference List

A **custom dictionary** of biased or discriminatory terms (related to gender, race, or age) was manually curated and used for detection.

---

## Tools & Technologies

| Category                 | Tools / Libraries                        |
| ------------------------ | ---------------------------------------- |
| **Language**             | Python                                   |
| **Frontend**             | Streamlit                                |
| **NLP Framework**        | Hugging Face Transformers                |
| **Model Used**           | T5 Transformer (for text simplification) |
| **Supporting Libraries** | pandas, torch, re, nltk                  |
| **Environment**          | Local deployment via Streamlit interface |

---

## Project Workflow

### 1. Text Input

File: `app.py`
Users can upload contract text files (PDF/TXT/DOCX).
The text is extracted, cleaned, and segmented into meaningful clauses.

### 2. Simplification

The **Pegasus model** (`google/pegasus-xsum`) generates simplified summaries of each clause while retaining legal meaning and structure.

### 3. Bias Detection

A **rule-based NLP component** scans the text for biased or discriminatory terms based on a keyword dictionary and linguistic patterns.

### 4. Fair Rewrite

Detected biased sentences are automatically rephrased into fair, neutral alternatives to ensure inclusivity and ethical compliance.

### 5. Output Display

The Streamlit app displays three views for user comparison:

* **Original Clause**
* **Simplified Version**
* **Fair Rewrite (if applicable)**

All operations are performed in-memory—no CSV output or file downloads are required.

---

## Model Details

| Component          | Model / Method                | Description                                       |
| ------------------ | ----------------------------- | ------------------------------------------------- |
| **Simplification** | Pegasus (Transformer-based)   | Generates simplified versions of contract clauses |
| **Bias Detection** | Keyword-based NLP rules       | Detects and flags bias using custom dictionaries  |
| **Fair Rewrite**   | Text rephrasing using Pegasus | Produces neutral, unbiased rewrites               |

---

## Evaluation

The project was qualitatively evaluated based on:

* **Readability improvement** (ease of understanding)
* **Semantic preservation** (legal meaning retained)
* **Bias reduction** (removal of discriminatory or unfair language)

Future quantitative evaluation can include readability metrics (e.g., Flesch–Kincaid score) and human review studies.

---

## How to Use

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application**

   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**
   Go to `http://localhost:8501`
   Upload a document → View simplified and unbiased versions instantly.

---

## Learning Outcomes

* Implemented a hybrid NLP system combining **summarization and bias detection**.
* Gained experience in **Hugging Face Transformers**, **Streamlit UI**, and **ethical AI principles**.
* Designed a functional prototype to assist legal professionals and general users.
* Strengthened understanding of **document-level NLP** and **fair language models**.

---

## Future Scope

* Integrate a **deep learning-based bias detection model** trained on larger legal corpora.
* Extend to **multi-language support** (e.g., Hindi, Spanish, French).
* Deploy via a **cloud API service** for scalable access.
* Add explainable AI insights to justify simplification decisions.

---

Would you like me to make this version look **GitHub-ready** (formatted in Markdown with icons, tables, and headings like a repository README)?
I can also include your **name, role, and course info** at the bottom if you plan to submit it officially.

