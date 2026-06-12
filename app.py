import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

st.title("🎬 Movie Recommendation System")

# Load data
movies_data = pd.read_csv("movies.csv")

# Fill missing values
features = [
    'genres',
    'keywords',
    'tagline',
    'cast',
    'director'
]

for feature in features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine features
combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

# TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Similarity matrix
similarity = cosine_similarity(feature_vectors)

movie_name = st.text_input("Enter your favorite movie")

if st.button("Recommend"):

    list_of_titles = movies_data['title'].tolist()

    close_match = difflib.get_close_matches(
        movie_name,
        list_of_titles
    )

    if len(close_match) == 0:
        st.error("Movie not found")
    else:
        close_movie = close_match[0]

        index = movies_data[
            movies_data.title == close_movie
        ].index[0]

        similarity_score = list(
            enumerate(similarity[index])
        )

        sorted_movies = sorted(
            similarity_score,
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader(
            f"Movies similar to {close_movie}"
        )

        count = 0

        for movie in sorted_movies[1:11]:

            idx = movie[0]

            title = movies_data[
                movies_data.index == idx
            ]['title'].values[0]

            st.write(title)

            count += 1

            if count == 10:
                break
