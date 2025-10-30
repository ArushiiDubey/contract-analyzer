# 🧠 Contract Simplifier & Bias Detector

**A Streamlit-based NLP web application that simplifies legal contracts, detects biased language, and suggests fair rewrites.**

---

## 🚀 Overview

Legal contracts are often filled with complex and one-sided language.  
This project uses **Natural Language Processing (NLP)** to make contracts easier to understand and identify clauses that might be **biased or unfair**.

Built entirely with **Python** and **Streamlit**, the app performs:
1. **Simplification** – converts legal text into plain English  
2. **Bias Detection** – flags one-sided clauses  
3. **Fair Rewrite** – suggests neutral rewordings

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|----------------|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Backend** | Python |
| **NLP Models** | Hugging Face Transformers |
| **Model Backend** | PyTorch |
| **Libraries** | `transformers`, `torch`, `pdfplumber`, `nltk`, `pandas`, `re` |

---

## 🧩 Models Used

| Feature | Model / Technique | Description |
|----------|------------------|--------------|
| **Simplification** | `philschmid/distilbart-cnn-12-6-samsum` | Summarizes and simplifies legal clauses into plain English |
| **Bias Detection (ML)** | `facebook/bart-large-mnli` | Zero-shot classifier that detects potentially unfair or exploitative clauses |
| **Bias Detection (Rule-Based)** | Custom keyword dictionary (`BIAS_KEYWORDS`) | Flags known biased terms like *sole discretion*, *non-refundable*, *exclusive rights* |
| **Fair Rewrite** | Regex-based replacements | Suggests neutral, ethical rephrasings for biased clauses |

---

## 📚 Data Source

No model was trained from scratch.  
The system uses:
- Sample **legal contract templates** (NDAs, employment, service agreements) for testing.  
- A custom **bias dictionary** of ~30 flagged legal phrases.  
- Pre-trained models from **Hugging Face** for summarization and classification.

---

## 🧮 Workflow

### 1. **Input Stage**
Upload a **PDF or TXT** file or paste contract text directly.  
`pdfplumber` extracts and cleans text using regex and NLTK.

### 2. **Clause Segmentation**
The contract is split into **logical paragraphs or clauses** for analysis.

### 3. **Simplification**
Each clause is summarized using **DistilBART**, producing:
- Plain English text
- Optional bullet-point summary

### 4. **Bias Detection**
Two-layer bias detection:
1. **Keyword Scan:** Finds legal red-flag phrases.  
2. **ML Classification:** Zero-shot model (`bart-large-mnli`) calculates bias probability.

Bias score is categorized as:
| Label | Range |
|--------|--------|
| 🟢 Low Bias | 0–0.25 |
| 🟠 Moderate Bias | 0.25–0.55 |
| 🔴 High Bias | >0.55 |

### 5. **Fair Rewrite**
Automatically replaces biased terms with neutral ones.  
Example:
> “at its sole discretion” → “after mutual agreement or with reasonable notice”

### 6. **Output**
Interactive Streamlit UI displays:
- Simplified text  
- Original clause  
- Fair rewrite suggestion  
- Bias score and explanation  

No files are stored or uploaded to external servers.
 <img width="1788" height="821" alt="image" src="https://github.com/user-attachments/assets/3ea873fe-ea4c-477d-8f25-dbae5f67fbfd" />
---

## 🧠 Educational & Technical Learning Outcomes

- Implemented **transformer pipelines** (summarization + classification)  
- Built **rule-based bias correction** logic  
- Designed an **interactive NLP app** using Streamlit  
- Learned to evaluate **ethical and fairness aspects** in contract analysis  

---


---

## 🧰 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/contract-simplifier-bias-detector.git
cd contract-simplifier-bias-detector

pip install -r requirements.txt

streamlit
transformers
torch
pdfplumber
nltk
pandas
sentencepiece
---
##🔮 Future Improvements

Integrate readability scoring (e.g., Flesch–Kincaid index)

Add human evaluation for bias severity

Build explainability dashboard for clause-level insights

Extend support for multi-language contracts

##👩‍💻 Author

Arushi Dubey



