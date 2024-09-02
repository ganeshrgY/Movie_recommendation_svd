import os
from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity
import pickle  # Import pickle for loading preprocessed data

app = Flask(__name__, template_folder="templates", static_folder="staticfiles")

# Define the path to the data directory relative to the current script's directory
data_path = "data.pkl"

try:
    # Load preprocessed data from the pickle file
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    movies = data.get('movies')
    movie_idx = data.get('movie_idx')
    movie_mapper = data.get('movie_mapper')
    cosine_sim = data.get('cosine_sim')

except FileNotFoundError:
    print("Data file not found at:", data_path)
    movies = None
    movie_idx = None
    movie_mapper = None
    cosine_sim = None
except Exception as e:
    print("Error loading data file:", e)
    movies = None
    movie_idx = None
    movie_mapper = None
    cosine_sim = None

# Function to get movie recommendations
def get_recommendations(movie_title):
    if movies is None or movie_idx is None or movie_mapper is None or cosine_sim is None:
        return None  # Return None if movies data is not available

    matched_title, _ = process.extractOne(movie_title, movie_idx.keys())
    idx = movie_idx[matched_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    similar_movies = [movie_mapper[i[0]] for i in sim_scores]
    return similar_movies

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations_route():
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        if movie_title:
            print("Movie title:", movie_title)  # Print the movie title
            recommendations = get_recommendations(movie_title)
            if recommendations is not None:
                return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)
            else:
                return "Error: Failed to get recommendations. Data not available."
        else:
            return "Error: Movie title not provided."
    else:
        return 'Method Not Allowed', 405

if __name__ == '__main__':
    app.run(debug=True)
