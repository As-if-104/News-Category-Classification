# Task 2: News Category Classification (Multiclass)

## 📌 Problem Definition
The goal of this task is to automatically categorize news articles into one of four distinct topics: **World, Sports, Business, or Sci/Tech**. Unlike Task 1 (binary), this is a multiclass classification problem requiring a Softmax approach.

## 📊 Dataset
- **Source:** Hugging Face `fancyzhx/ag_news`
- **Size Used:** 10,000 training samples / 2,000 test samples (subset for faster iteration).
- **Categories:** 0 (World), 1 (Sports), 2 (Business), 3 (Sci/Tech)

## 🛠️ Pipeline & Preprocessing
1. **Text Cleaning:** Lowercasing, punctuation removal, stopword removal, and lemmatization using `NLTK`.
2. **Vectorization:** `TfidfVectorizer` with `max_features=20000` and `ngram_range=(1,2)` to capture bigrams (e.g., "stock market", "white house").
3. **Dense Conversion:** TF-IDF sparse matrices were converted to dense arrays specifically for the Tree-based models (Random Forest, XGBoost, LightGBM) and the Neural Network, as they require dense inputs.

## 🤖 Model Benchmarking & Results
To find the best approach, I benchmarked 6 different algorithms. **Macro F1** was used as the primary metric to ensure the model performs equally well across all 4 classes, rather than just favoring the majority class.

| Model | Accuracy | Macro F1-Score |
| :--- | :--- | :--- |
| **Logistic Regression** | **0.8930** | **0.8933** |
| Linear SVM | 0.8925 | 0.8930 |
| Neural Network (Keras) | 0.8800 | 0.8805 |
| LightGBM | 0.8745 | 0.8750 |
| XGBoost | 0.8670 | 0.8676 |
| Random Forest | 0.8535 | 0.8537 |

## 💡 Key Insights & Analysis
1. **Linear Models Dominate:** Logistic Regression and Linear SVM outperformed all other models. This proves that for sparse, high-dimensional TF-IDF representations, linear decision boundaries are highly effective and computationally efficient.
2. **Neural Networks:** A simple feedforward Neural Network (128 -> 64 -> 4 nodes with Dropout) performed well but didn't beat linear models. (Note: Non-linear models typically require dense word embeddings like Word2Vec or BERT to outperform linear models on text).
3. **Tree-Based Models Struggle:** LightGBM, XGBoost, and Random Forest performed worse. Decision trees struggle with high-dimensional sparse data because they make splits on one word at a time, whereas linear models can weigh combinations of words simultaneously.
4. **Confusion Matrix Insights (Domain Overlap):** 
   - **Sports** was the easiest category to classify due to highly distinct vocabulary.
   - **Business vs. Sci/Tech** had the highest confusion rate. For example, an article about Apple's quarterly earnings uses vocabulary from both domains, causing overlap.

## 🎨 Bonus Tasks Completed
- [x] Visualized the top 15 most frequent words per category using bar plots.
- [x] Trained a simple feedforward Neural Network with Keras and compared it against traditional ML models.
