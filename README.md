## Book Recommendation

### Overview

This project builds a Content-Based Book Recommendation System using TF-IDF Vectorization and Cosine Similarity. The system suggests books similar to a user’s search query by analyzing textual metadata such as book title, author, and publisher.
The model is evaluated using unsupervised metrics — Diversity, Novelty, Coverage, and Popularity Bias — to ensure recommendation quality and variety.

### Features

* Cleans and preprocesses large-scale book datasets.
* Combines textual metadata (Book-Title, Book-Author, Publisher) for feature creation.
* Uses TF-IDF to convert text data into numerical vectors.
* Computes Cosine Similarity between user queries and book metadata.
* Evaluates recommendation quality using advanced unsupervised metrics.
* Returns detailed book recommendations with titles, authors, and scores.

### Project Pipeline

1. **Data Loading & Cleaning** – Merge and preprocess datasets, remove duplicates, and filter active users/books.
2. **Feature Engineering** – Combine book metadata into a unified text feature.
3. **Vectorization** – Transform textual data using TF-IDF.
4. **Similarity** Computation – Use cosine similarity for book-query matching.
5. **Evaluation** – Assess system with diversity, novelty, coverage, and popularity metrics.
6. **Deployment** - Deployed using steamlit with interactive user interface

### Evaluation Metrics

**Diversity** – Measures variety among recommended books.
**Novelty** – Rewards less popular, unique books.
**Coverage** – Determines catalog breadth.
**Popularity Bias** – Checks over-recommendation of famous titles.

### Tools Used

- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib / Seaborn (for visualization)
- Jupyter Notebook / Google Colab
- Streamlit

### Example Queries
  ```
     "harry potter"
     "lord of the rings"
     "four blind mice"
  ```
### Results Summary

- Good diversity and novelty across recommendations.
- Low popularity bias ensuring fair exposure.
- Moderate coverage — room for improvement in catalog exploration.

---

<h3 align="center">By</h3>

<h4 align="center">Sucharita Lakkavajhala</h4>
<h4 align="center">Shyam Kumar Kampelly</h4>
<h4 align="center">Uday Kumar Barigela</h4>
<h4 align="center">Pravalika Challuri</h4>
