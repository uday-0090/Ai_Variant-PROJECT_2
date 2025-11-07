import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    books = pd.read_csv('Books (2).csv')
    ratings = pd.read_csv('Ratings (1).csv')
    users = pd.read_csv('Users (2).csv')

    df = pd.merge(pd.merge(ratings, users, on='User-ID', how='left'), books, on='ISBN', how='left')
    df.dropna(inplace=True)
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df['Age'] = df['Age'].astype(int)

    Q1, Q3 = df['Age'].quantile(0.25), df['Age'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['Age'] >= Q1 - 1.5 * IQR) & (df['Age'] <= Q3 + 1.5 * IQR)]

    stats = df.groupby('ISBN').agg(
        avg_rating=('Book-Rating', 'mean'),
        num_ratings=('Book-Rating', 'count')
    ).reset_index()
    df = df.merge(stats, on='ISBN', how='left')
    df['popularity_score'] = df['avg_rating'] * np.sqrt(df['num_ratings'])
    return df

def recommend_books_by_search(df, search_query, top_n=10, min_rating=0, min_num_ratings=0):
    unique_books = df.drop_duplicates(subset='ISBN').copy()
    unique_books = unique_books[
        (unique_books['avg_rating'] >= min_rating) &
        (unique_books['num_ratings'] >= min_num_ratings)
    ].reset_index(drop=True)

    unique_books['text_search'] = unique_books[['Book-Title', 'Book-Author', 'Publisher']].fillna('').agg(' '.join, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(unique_books['text_search'])

    query_vector = tfidf.transform([search_query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_n]

    similar_books = unique_books.iloc[top_indices].copy()
    similar_books['similarity_score'] = cosine_similarities[top_indices]

    top_rated_books = unique_books.sort_values(by='avg_rating', ascending=False).sample(top_n,random_state=42)
    most_popular_books = unique_books.sort_values(by='popularity_score', ascending=False).sample(top_n,random_state=42)

    return {
        'similar_books': similar_books,
        'top_rated_books': top_rated_books,
        'most_popular_books': most_popular_books
    }


def render_book_grid(books_df, section_title):
    st.markdown(f"### {section_title}")
    st.markdown("---")

    rows = [books_df[i:i + 5] for i in range(0, len(books_df), 5)]

    for row in rows:
        cols = st.columns(6)
        for col, (idx, book) in zip(cols, row.iterrows()):
            with col:
                isbn = book['ISBN']
                title = f"{book['Book-Title']} ({int(book['Year-Of-Publication'])})"
                rating = round(book['avg_rating'], 2)

                st.image(book['Image-URL-M'], use_container_width=True)
                st.caption(f"⭐ {rating}")
                if st.button(title, key=f"btn_{isbn}_{idx}"):  # ← uses idx from iterrows()
                    st.session_state.selected_book = isbn

def show_book_details(df, isbn):
    book = df[df['ISBN'] == isbn].iloc[0]
    st.markdown("### 📘 Book Details")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(book['Image-URL-L'], width=180)

    with col2:
        st.markdown(f"### {book['Book-Title']} ({int(book['Year-Of-Publication'])})")
        st.markdown(f"**Author:** {book['Book-Author']}")
        st.markdown(f"**Publisher:** {book['Publisher']}")
        st.markdown(f"**Average Rating:** {round(book['avg_rating'], 2)}")
        st.markdown(f"**Number of Ratings:** {book['num_ratings']}")
        if st.button("🔙 Back to Recommendations"):
            st.session_state.selected_book = None

    st.markdown("---")
    st.markdown(f"### 📚 More books by {book['Book-Author']}")
    author_books = df[(df['Book-Author'] == book['Book-Author']) & (df['ISBN'] != isbn)]
    author_books = author_books.sort_values(by='popularity_score', ascending=False).drop_duplicates('ISBN').head(10)

    render_book_grid(author_books, "")

def main():
    st.set_page_config(page_title="📚 Book Recommender", layout="wide")
    st.title("📚 Book Recommendation System")

    df = load_data()
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None

    search_query = st.text_input("🔍 Search for books (by title, author, or publisher)", "lord of the rings")

    with st.sidebar:
        st.header("🎛️ Filters")
        min_rating = st.slider("Minimum Average Rating", 0.0, 5.0, 3.5, 0.1)
        min_num_ratings = st.slider("Minimum Number of Ratings", 0, 100, 10)
        top_n = st.slider("Number of Recommendations", 5, 20, 10)

    if st.session_state.selected_book:
        show_book_details(df, st.session_state.selected_book)
    elif search_query:
        results = recommend_books_by_search(df, search_query, top_n, min_rating, min_num_ratings)
        render_book_grid(results['similar_books'], "🔍 Similar Books")
        render_book_grid(results['top_rated_books'], "⭐ Top Rated Books")
        render_book_grid(results['most_popular_books'], "🔥 Most Popular Books")

if __name__ == "__main__":
    main()
