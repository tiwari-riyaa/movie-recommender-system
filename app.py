from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

def fetch_poster(movie_id):
    try:
        api_key = "c51b942f2c159e5aedb294eb6dc0676c"

        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"

        response = requests.get(url)
        data = response.json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        else:
            return "https://picsum.photos/200/300"

    except:
        return "https://picsum.photos/200/300"


app = Flask(__name__)

# Load dataset
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')

# Keep important columns
movies = movies[['id','title', 'genres', 'overview']]

# Fill missing values
movies.fillna('', inplace=True)

# Convert everything to string (IMPORTANT)
movies['genres'] = movies['genres'].astype(str)
movies['overview'] = movies['overview'].astype(str)

# Combine features
movies['tags'] = (movies['genres'] + " " + movies['overview']).str.lower()

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.strip().title()

    if movie_name not in movies['title'].values:
        return ["Movie not found"], ["https://picsum.photos/200/300"]

    index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[index]

    movie_list = sorted(list(enumerate(distances)),
                        key=lambda x: x[1],
                        reverse=True)[1:6]

    recommended_movies = []
    posters = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
        movie_id = movies.iloc[i[0]].id
        posters.append(fetch_poster(movie_id))


    return recommended_movies, posters




@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    posters = []

    if request.method == "POST":
        movie_name = request.form["movie"]
        recommendations, posters = recommend(movie_name)

    return render_template("index.html",
                           recommendations=recommendations,
                           posters=posters)


if __name__ == "__main__":
    app.run(debug=False, port=5001)

