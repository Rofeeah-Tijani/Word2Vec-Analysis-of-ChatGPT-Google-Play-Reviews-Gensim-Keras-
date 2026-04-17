# Word2Vec-Analysis-of-ChatGPT-Google-Play-Reviews-Gensim-Keras-
# 🧠 Word2Vec Analysis of ChatGPT Google Play Reviews (Gensim + Keras)

## 📌 Project Overview

This project explores how machines learn the meaning of words using **Word2Vec**, applied to real-world user reviews of the ChatGPT application collected from the Google Play Store.

The goal is to understand how different Word Embedding approaches (Gensim vs Keras) capture semantic relationships between words like *helpful*.

---

## 🎯 Objectives

- Collect real user reviews of ChatGPT from Google Play Store
- Preprocess and clean textual data
- Train Word2Vec models using:
  - Gensim (high-level implementation)
  - Keras (manual implementation from scratch)
- Compare CBOW and Skip-gram approaches
- Analyze semantic relationships between words
- Extract and interpret word embeddings

---

## 📊 Dataset

- Source: Google Play Store (ChatGPT app reviews)
- Data Type: User-generated text reviews
- Format: Manually copied textual reviews
- Size: ~30+ reviews (for learning purposes)

Example:

- "ChatGPT is very helpful and saves time"
- "The app is amazing and easy to use"
- "It sometimes gives wrong information"
- "Very useful but not always reliable"

---

## 🧹 Data Preprocessing

Before training the models, the text was cleaned and prepared:

### Steps:
- Converted text to lowercase
- Removed punctuation using `re`
- Tokenized sentences into words using `.split()`
- Converted text into structured format for Word2Vec

### Example:

**Before:**

ChatGPT is AMAZING!!!

**After:**

["chatgpt", "is", "amazing"]


---

## 🧠 Word Embedding Techniques Used

## 1️⃣ Gensim Word2Vec (High-Level API)

Gensim was used for fast and efficient training of Word2Vec models.

### Model Types:
- CBOW (Continuous Bag of Words)
- Skip-gram

### Key Parameters:
- `vector_size = 100`
- `window = 2`
- `min_count = 1`
- `sg = 0 (CBOW), sg = 1 (Skip-gram)`

### Example Code:
```python
from gensim.models import Word2Vec
````
model = Word2Vec(
    sentences=tokenized_reviews,
    vector_size=100,
    window=2,
    min_count=1,
    sg=0
)

🔍 Insights from Gensim

model_sg.wv.most_similar("helpful")

| Word        | Similarity Score |
| ----------- | ---------------- |
| ever        | 0.3493           |
| information | 0.3043           |
| helps       | 0.2487           |
| truly       | 0.2221           |
| it          | 0.1779           |
| much        | 0.1643           |
| incredibly  | 0.1622           |
| life        | 0.1548           |
| satisfying  | 0.1524           |
| ChatGPT     | 0.1466           |


model.wv.most_similar("disappointing")

| Word       | Similarity Score |
| ---------- | ---------------- |
| incredibly | 0.2163           |
| Good       | 0.1924           |
| but        | 0.1562           |
| tool       | 0.1487           |
| content    | 0.1385           |
| filter     | 0.1191           |
| so         | 0.1157           |
| navigate   | 0.1120           |
| gives      | 0.1023           |
| very       | 0.0982           |

model.wv.most_similar(positive=["excellent", "helpful"], negative=["disappointing"])

| Word       | Similarity Score |
| ---------- | ---------------- |
| in         | 0.2493           |
| life       | 0.2099           |
| ever       | 0.1984           |
| one        | 0.1976           |
| truly      | 0.1943           |
| satisfying | 0.1914           |
| deep       | 0.1841           |
| too        | 0.1661           |
| Best       | 0.1661           |
| unmatched  | 0.1550           |

-----

2️⃣ Keras Word2Vec (From Scratch Implementation)

A custom Word2Vec model was built using Keras to understand the internal mechanism of embedding learning.

-----

🔹 Step 1: Tokenization

Words were converted into integer sequences using Keras Tokenizer.

-----

🔹 Step 2: Skip-gram Dataset Creation

Using skipgrams() function, training pairs were generated:

Target word → Context word
Context word → Target word

-----

🔹 Step 3: Neural Network Architecture

The model consists of:

Input layer (target & context words)
Embedding layer (word vectors)
Dot product layer (similarity computation)
Output layer (binary classification)
Model Flow:
Input Word Pair → Embedding Layer → Dot Product → Similarity Score

-----

🔹 Step 4: Training Objective

The model learns to:

Predict whether two words appear in the same context window.

Loss Function: Binary Crossentropy

Optimizer: Adam

-----

🔹 Step 5: Extracting Word Embeddings

After training:

word_vectors = model.get_layer("embedding").get_weights()[0]

These vectors represent learned word meanings.

📈 Key Learnings
Word meaning can be represented numerically as vectors
Words appearing in similar contexts have similar embeddings
CBOW predicts center word from context
Skip-gram predicts context from center word
Neural networks can learn language structure without explicit rules

------

🚀 Conclusion

This project demonstrates how machine learning models can learn semantic relationships between words using real-world text data.

It also highlights the difference between using a prebuilt library (Gensim) and building a model from scratch (Keras), reinforcing a deeper understanding of Word2Vec.
