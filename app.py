"""
Content-Based Movie Recommender System
----------------------------------------
Streamlit front-end that serves recommendations produced by the
NLP/cosine-similarity pipeline in notebooks/movie_recommender_eda.ipynb.

Run locally:
    streamlit run app.py

Requires a TMDB API key stored in .streamlit/secrets.toml (see
.streamlit/secrets.toml.example) or as an environment variable TMDB_API_KEY.
"""

import os
import pickle

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

DATA_DIR = "data"
MOVIE_DICT_PATH = os.path.join(DATA_DIR, "movie_dict.pkl")
SIMILARITY_PATH = os.path.join(DATA_DIR, "similarity.pkl")
PLACEHOLDER_POSTER = "https://placehold.co/500x750?text=No+Poster"


def get_api_key() -> str:
    """Fetch the TMDB API key from Streamlit secrets first, then env vars.

    Never hardcode the key in source. For local dev, put it in
    .streamlit/secrets.toml. For Streamlit Community Cloud, set it under
    App settings -> Secrets.
    """
    if "TMDB_API_KEY" in st.secrets:
        return st.secrets["TMDB_API_KEY"]
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        st.error(
            "TMDB API key not found. Add it to .streamlit/secrets.toml "
            "(local) or your host's secrets manager (deployed)."
        )
        st.stop()
    return key


# --------------------------------------------------------------------------
# Cached data loaders (avoid re-reading pickles on every rerun)
# --------------------------------------------------------------------------
@st.cache_resource
def load_data():
    movie_dict = pickle.load(open(MOVIE_DICT_PATH, "rb"))
    movies_df = pd.DataFrame(movie_dict)
    similarity_matrix = pickle.load(open(SIMILARITY_PATH, "rb"))
    return movies_df, similarity_matrix


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # cache posters for 24h
def fetch_poster(movie_id: int, api_key: str) -> str:
    """Fetch a poster URL from TMDB, falling back gracefully on any failure."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    try:
        response = requests.get(url, params={"api_key": api_key}, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return PLACEHOLDER_POSTER
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except (requests.RequestException, ValueError):
        return PLACEHOLDER_POSTER


def recommend(movie_name: str, movies_df: pd.DataFrame, similarity_matrix, api_key: str):
    """Return top-5 similar movies (titles + poster URLs) for a given title."""
    movie_index = movies_df[movies_df["title"] == movie_name].index[0]
    distances = similarity_matrix[movie_index]
    top_matches = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    names, posters = [], []
    for i, _score in top_matches:
        movie_id = movies_df.iloc[i].movie_id
        names.append(movies_df.iloc[i].title)
        posters.append(fetch_poster(movie_id, api_key))
    return names, posters


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def main():
    st.title("🎬 Movie Recommender System")
    st.caption(
        "Content-based recommendations using TF-style tags (overview, genre, "
        "cast, crew, keywords) and cosine similarity. "
        "[Notebook & write-up](https://github.com/YOUR_USERNAME/movie-recommender-system)"
    )

    movies_df, similarity_matrix = load_data()
    api_key = get_api_key()

    selected_movie = st.selectbox(
        "Pick a movie you like:", movies_df["title"].values
    )

    if st.button("Recommend", type="primary"):
        with st.spinner("Finding similar movies..."):
            names, posters = recommend(
                selected_movie, movies_df, similarity_matrix, api_key
            )

        cols = st.columns(5)
        for col, name, poster in zip(cols, names, posters):
            with col:
                st.image(poster, use_container_width=True)
                st.caption(name)


if __name__ == "__main__":
    main()
