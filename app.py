from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import difflib

# Load the model and data
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))
movies = pickle.load(open('movies.pkl', 'rb'))

# Create a reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Find the closest match for the movie title
    find_close_match = difflib.get_close_matches(title, movies['title'])
    
    if not find_close_match:
        return None

    close_match = find_close_match[0]
    idx = indices[close_match]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices].tolist(), close_match

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations, close_match = get_recommendations(title)
    
    if recommendations is None:
        return render_template('index.html', recommendations=[], message="No close match found for the movie title.")
    
    return render_template('index.html', recommendations=recommendations, movie_title=close_match)

if __name__ == '__main__':
    app.run(debug=True)
