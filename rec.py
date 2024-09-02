from flask import Flask, render_template, request
from fuzzywuzzy import process
import pickle

app = Flask(__name__, template_folder="templates", static_folder="staticfiles")

# Load preprocessed data from pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

movie_idx = data['movie_idx']
movie_mapper = data['movie_mapper']
movie_inv_mapper = data['movie_inv_mapper']
movie_titles = data['movie_titles']
similar_movies_titles = data['similar_movies_titles']
Q = data['Q']

def find_similar_movies(movie_id, Q_T, movie_mapper, movie_inv_mapper, metric='cosine', k=10):
    X = Q_T.T
    neighbour_ids = []
    movie_ind = movie_mapper.get(movie_id)
    if movie_ind is None:
        print(f"Movie ID {movie_id} not found in movie_mapper.")
        return []
    movie_vec = X[movie_ind]
    if isinstance(movie_vec, (np.ndarray)):
        movie_vec = movie_vec.reshape(1, -1)
    kNN = NearestNeighbors(n_neighbors=k+1, algorithm="brute", metric=metric)
    kNN.fit(X)
    neighbour = kNN.kneighbors(movie_vec, return_distance=False)
    for i in range(0, k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper.get(n))
    neighbour_ids.pop(0)
    return neighbour_ids

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommendations():
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        print("Received movie title:", movie_title)
        matched_title, score = process.extractOne(movie_title, movie_idx.keys())
        print("Matched title:", matched_title)
        print("Matching score:", score)
        movie_id = movie_idx.get(matched_title)
        print("Matched movie ID:", movie_id)
        if movie_id is None:
            return render_template('error.html', error_message='Movie title not found')
        similar_movies = find_similar_movies(movie_id, Q.T, movie_mapper, movie_inv_mapper, metric='cosine', k=10)
        similar_movies_titles = [movie_titles[i] for i in similar_movies]
        print("Similar movies:", similar_movies_titles)
        return render_template('index.html', movie_title=matched_title, recommendations=similar_movies_titles)
    else:
        return 'Method Not Allowed', 405

if __name__ == '__main__':
    app.run(debug=True)
