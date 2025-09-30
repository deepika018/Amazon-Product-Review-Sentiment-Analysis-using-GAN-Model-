# Amazon-Product-Review-Sentiment-Analysis-using-GAN-Model-

This Jupyter Notebook performs an exploratory sentiment analysis on a dataset of Amazon product reviews. The primary goal is to compare the sentiment scoring capabilities of a lexicon-based model (VADER) with a pre-trained transformer model (RoBERTa) against the known star ratings.

(Note: The notebook title mentions a "GAN Model," but the code provided focuses on VADER and RoBERTa for comparison. This README reflects the actual contents of the executed code cells.)

---

## 📌 Project Goal
The primary objectives of this project are:
- Load and explore the **Amazon Fine Food Reviews dataset**.
- Perform sentiment analysis using:
  - **VADER** (Valence Aware Dictionary and sEntiment Reasoner).
  - **RoBERTa** (pre-trained transformer model).
- Compare sentiment predictions from both models against the user-provided **star rating (Score)**.
- Identify examples where the models align or contradict the ratings, highlighting their **strengths and weaknesses** (e.g., sarcasm, negations, complex language).

---

## 📊 Dataset
- **Source:** Amazon Fine Food Reviews  
- **Original Size:** 568,454 reviews  
- **Sample Size:** First **500 reviews** (`df.head(500)`)  

**Key Columns:**
- `Id`, `ProductId`, `UserId`, `ProfileName`, `HelpfulnessNumerator`, `HelpfulnessDenominator`, `Time`, `Summary`
- `Score`: Ground truth star rating (1–5)
- `Text`: Raw review text used for sentiment analysis

---

## 🛠️ Tools & Models

### 🔹 VADER (Lexicon-based)
- Library: `nltk.sentiment.SentimentIntensityAnalyzer`  
- **Outputs:**
  - `vader_neg` → Negative sentiment score  
  - `vader_neu` → Neutral sentiment score  
  - `vader_pos` → Positive sentiment score  
  - `vader_compound` → Composite normalized score (-1 to +1)

### 🔹 RoBERTa (Transformer-based)
- Model: Fine-tuned **RoBERTa sentiment classifier**  
- Likely used: [`twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
- **Outputs (Softmax probabilities):**
  - `roberta_neg` → Negative sentiment probability  
  - `roberta_neu` → Neutral sentiment probability  
  - `roberta_pos` → Positive sentiment probability  

---

## 🔑 Key Steps

1. **Setup & Data Loading**
   - Import libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `transformers`)
   - Load dataset (`Reviews.csv`)
   - Sample first 500 rows

2. **Exploratory Data Analysis (EDA)**
   - Bar chart: Count of reviews by star rating (`Score`)

3. **Basic NLP Preprocessing**
   - Tokenization & Part-of-Speech (POS) tagging using NLTK

4. **VADER Analysis**
   - Calculate sentiment scores for each review
   - Visualize `vader_compound` vs. `Score`

5. **RoBERTa Analysis**
   - Load tokenizer & model from local path  
     `C:\\Users\\benis\\Downloads\\twitter-roberta-base-sentiment`
   - Define function `polarity_scores_roberta`  
   - Generate sentiment probabilities

6. **Combined Analysis**
   - Merge VADER & RoBERTa results into a single DataFrame (`results_df`)
   - Create pairplots to visualize relationships among six sentiment scores
   - Color plots by ground truth `Score`

7. **Case Studies**
   - Highest `roberta_pos` for `Score == 1`
   - Highest `vader_pos` for `Score == 1`
   - Highest `roberta_neg` for `Score == 5`
   - Highest `vader_neg` for `Score == 5`

---

## 📦 Requirements

### 🔹 Python
- Version: **3.x**

### 🔹 Libraries
```bash
pandas
numpy
matplotlib
seaborn
nltk
transformers
scipy
tqdm

