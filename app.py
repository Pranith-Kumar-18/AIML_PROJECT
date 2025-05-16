import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv('ml-latest-small/movies.csv')
movie_genres = movies.copy()
movie_genres['genres'] = movie_genres['genres'].str.replace('|', ' ')

# Compute TF-IDF and cosine similarity matrix
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(movie_genres['genres'])
genre_sim = cosine_similarity(genre_matrix)

# Recommend movies function
def recommend_movies(movie_title, top_n=5):
    try:
        idx = movie_genres[movie_genres['title'] == movie_title].index[0]
        sim_scores = list(enumerate(genre_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return movie_genres['title'].iloc[movie_indices].tolist()
    except:
        return ["Movie not found. Please try another title."]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Get similar movie suggestions based on genre.")

movie_list = movie_genres['title'].tolist()
selected_movie = st.selectbox("Choose a Movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)
    st.write("Top 5 Recommendations:")
    for i, rec in enumerate(recommendations):
        st.write(f"{i+1}. {rec}")
