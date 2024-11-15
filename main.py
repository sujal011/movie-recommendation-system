# pip install pandas scikit-learn

import pandas as pd
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

tmdb_api_key = os.getenv('TMDB_API_KEY')

movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Data Preprocessing

data = pd.merge(ratings, movies, on='movieId')

#  Collaborative Filtering (Using the Surprise Library)

# from surprise import Dataset, Reader
# from surprise import SVD
# from surprise.model_selection import train_test_split
# from surprise import accuracy

# # Prepare the data for Surprise
# reader = Reader(rating_scale=(0.5, 5.0))
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Content-Based Filtering

# Vectorize movie genres using TF-IDF:
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

print(tfidf_matrix.shape)

# Compute the cosine similarity between movies:

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping from movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to fetch movie poster URLs from TMDB
import time
import re

def fetch_poster_url(title):
    api_key = tmdb_api_key
    title = re.sub(r'\s\(\d{4}\)$', '', title)
    print(title)
    url = f'https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}'
    
    # Try up to 3 attempts if the request fails
    for attempt in range(1):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            if data['results']:
                poster_path = data['results'][0]['poster_path']
                return f"https://image.tmdb.org/t/p/w200{poster_path}"
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}: Error fetching poster - {e}")
            time.sleep(2)  # Wait before retrying
    return None
  # Return None if poster is not found
# Placeholder image for movies without a poster
placeholder_image = "https://via.placeholder.com/100x150?text=No+Image"

# Example: Recommend movies similar to a given movie
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# print(get_recommendations('Toy Story (1995)'))

import streamlit as st

st.title("Movie Recommendation System")

# Searchable dropdown for selecting a movie using `multiselect`
movie_list = movies['title'].values  # List of movies for dropdown
user_choice = st.multiselect("Search and select your favourite movie", movie_list, default=movie_list[0])

# Button to get recommendations
if st.button("Get Recommendation"):
    if user_choice:
        recommendations = get_recommendations(user_choice[0])  # Take the first selected movie
        st.write("**Recommended Movies:**")
        
        # Display recommended movies in a grid layout
        cols = st.columns(5)  # Adjust number of columns as needed
        for idx, movie in enumerate(recommendations):
            with cols[idx % 5]:  # Distribute movies across columns
                poster_url = fetch_poster_url(movie) or placeholder_image
                st.image(poster_url, width=100, use_container_width=True)  # Set consistent width
                st.write(movie)
    else:
        st.write("Please select a movie.")

    