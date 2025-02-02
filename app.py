import numpy as np
import pandas as pd
import streamlit as st
import warnings
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

warnings.simplefilter(action='ignore', category=FutureWarning)

# Initialize session state if not already initialized
if 'page' not in st.session_state:
    st.session_state.page = "Home"  # Default page

# Load datasets
@st.cache_data  # Cache the data loading for performance
def load_data():
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    return ratings, movies

ratings, movies = load_data()

# Create user-item matrix
def create_matrix(df):
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
    
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
    
    user_inv_mapper = {v: k for k, v in user_mapper.items()}
    movie_inv_mapper = {v: k for k, v in movie_mapper.items()}
    
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
    
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
    
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings)

# Find similar movies
def find_similar_movies(movie_id, X, k=10, metric='cosine'):
    if movie_id not in movie_mapper:
        return []
    
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    
    distances, indices = kNN.kneighbors(movie_vec)
    neighbor_ids = [movie_inv_mapper[idx] for idx in indices[0][1:]]  # Exclude first (itself)
    return neighbor_ids

# TF-IDF vectorizer for content-based filtering
vectorizer = TfidfVectorizer()
movies['genres_str'] = movies['genres'].fillna('')
movies_genres_matrix = vectorizer.fit_transform(movies['genres_str'])

# Sidebar navigation with session state
def sidebar_navigation():
    st.sidebar.title("Navigation")
    
    # Add buttons to select pages
    home_button = st.sidebar.button("Home")
    info_button = st.sidebar.button("Info")
    movie_recommendations_button = st.sidebar.button("Movie Recommendations")
    movie_reviews_button = st.sidebar.button("Movie Reviews")
    database_button = st.sidebar.button("Database")
    
    if home_button:
        st.session_state.page = "Home"
    elif info_button:
        st.session_state.page = "Info"
    elif movie_recommendations_button:
        st.session_state.page = "Movie Recommendations"
    elif movie_reviews_button:
        st.session_state.page = "Movie Reviews"
    elif database_button:
        st.session_state.page = "Database"

# Sidebar fixed positioning style
st.sidebar.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 40px;
        font-size: 16px;
        margin-bottom: 10px;  /* Adds space between buttons */
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation call
sidebar_navigation()

# Movie title mapper
movies_title_mapper = dict(zip(movies['title'].str.strip(), movies['movieId']))

# Movie Recommendations Page
if st.session_state.page == "Movie Recommendations":
    st.header("Movie Recommendation System")
    
    # User input for movie title
    title = st.text_input("Enter Movie Title:").strip()
    search_clicked = st.button("Search")
    
    if search_clicked and title:
        if title in movies_title_mapper:
            selected_movie = movies[movies['title'].str.strip() == title]
            
            if not selected_movie.empty:
                movie_info = selected_movie.iloc[0]
                st.subheader("Movie Details")
                st.markdown(f"**Title:** {movie_info['title']}")
                st.markdown(f"**Genres:** {movie_info['genres']}")
                st.markdown(f"**Movie ID:** {movie_info['movieId']}")
                
                # Movie Recommendations Section
                # Get content-based filtering similar movies
                movie_vec = vectorizer.transform([movie_info['genres_str']]).toarray()
                content_similarities = cosine_similarity(movie_vec, movies_genres_matrix).flatten()
                similar_movies_cb_indices = np.argsort(content_similarities)[-11:-1][::-1]  # Top 10 content-based recommendations
                
                # Get collaborative filtering similar movies
                movie_id = movies_title_mapper[title]
                similar_movies_cf = find_similar_movies(movie_id, X, k=10)
                
                # Combine both CBF and CF results as sets of movie indices
                all_similar_movies_indices = set(similar_movies_cb_indices.tolist()) | set(similar_movies_cf)  # Removed duplicates
                
                # Get the top 10 movie titles safely
                recommended_movie_titles = [movies['title'].iloc[idx] for idx in all_similar_movies_indices if idx < len(movies)]  # Ensure idx is valid
                
                st.subheader("Top 10 Recommended Movies")
                for movie_title in recommended_movie_titles[:10]:  # Display only top 10
                    st.markdown(f"🎬 **{movie_title}**")
        else:
            st.error("Movie not found. Please enter a valid title.")

# Movie Reviews Page
elif st.session_state.page == "Movie Reviews":
    st.header("Movie Reviews")
    
    # User input for movie title
    title = st.text_input("Enter Movie Title for Reviews:").strip()
    search_clicked = st.button("Search Reviews")
    
    if search_clicked and title:
        if title in movies_title_mapper:
            selected_movie = movies[movies['title'].str.strip() == title]
            
            if not selected_movie.empty:
                movie_info = selected_movie.iloc[0]
                st.subheader("Movie Details")
                st.markdown(f"**Title:** {movie_info['title']}")
                st.markdown(f"**Genres:** {movie_info['genres']}")
                st.markdown(f"**Movie ID:** {movie_info['movieId']}")
                
                # Movie Reviews Section
                st.subheader("Reviews Data")
                
                # Merge reviews with movie titles
                movie_reviews = ratings[ratings['movieId'] == movie_info['movieId']]  # Using ratings as reviews
                
                if not movie_reviews.empty:
                    unique_user_ids = movie_reviews['userId'].unique()
                    unique_user_reviews = []
                    
                    for user_id in unique_user_ids:
                        user_reviews = movie_reviews[movie_reviews['userId'] == user_id]
                        unique_user_reviews.append(user_reviews)
                        
                    st.subheader("Reviews")
                    
                    for i, user_reviews in enumerate(unique_user_reviews, start=1):
                        if not user_reviews.empty:
                            st.subheader(f"Reviews from User {i}:")
                            st.write(user_reviews)
                        else:
                            st.write(f"No reviews available from user {i}.")
                else:
                    st.write("No reviews available for this movie.")
        else:
            st.error("Movie not found. Please enter a valid title.")

# Database Page
elif st.session_state.page == "Database":
    st.header("Movie and Rating Database")
    
    # Display Movies and Ratings DataFrames
    st.write("### Movies Database")
    st.dataframe(movies)
    
    st.write("### Ratings Database")
    st.dataframe(ratings)

# Home Page
elif st.session_state.page == "Home":
    st.header("Welcome to the Movie Recommendation System!")

# Info Page
elif st.session_state.page == "Info":
    st.header("About This Project")
    st.write("This system provides movie recommendations using content-based and collaborative filtering, along with reviews.")
