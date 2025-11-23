import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import difflib
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Music Recommendation System", layout="wide")


# -------------------------------------------------------
# SPOTIFY THEME FIXED (MODERN, CLEAN, HIGH CONTRAST)
# -------------------------------------------------------
def spotify_theme():
    st.markdown("""
    <style>

    /* Full App Background */
    .stApp {
        background: linear-gradient(145deg, #000000 20%, #003300 100%) !important;
        color: #1DB954 !important;
        font-family: "Segoe UI", sans-serif;
    }

    /* Main Container (Glassmorphism) */
    .block-container {
        padding: 2rem !important;
        border-radius: 12px;
        background: rgba(0, 0, 0, 0.55) !important;
        backdrop-filter: blur(8px);
    }

    /* Widgets */
    .stTextInput>div>div>input,
    .stSelectbox>div>div,
    .stSlider>div>div,
    .stNumberInput>div>div>input {
        background-color: rgba(255,255,255,0.10) !important;
        border-radius: 10px;
        color: #FFFFFF !important;
        padding: 10px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #1DB954 !important;
        color: black !important;
        border-radius: 10px;
        padding: 0.7em 1.4em;
        font-weight: bold;
        border: none;
        transition: 0.2s;
        font-size: 18px;
    }

    .stButton>button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.06);
    }

    /* Title */
    h1 {
        color: #1DB954 !important;
        text-shadow: 0px 0px 8px #1DB954;
        font-weight: 900;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.95) !important;
        border-right: 2px solid #1DB954;
    }
    section[data-testid="stSidebar"] * {
        color: #1DB954 !important;
    }

    /* DataFrame Theme */
    .stDataFrame tbody tr td {
        color: white !important;
    }

    .stDataFrame thead tr th {
        background-color: #1DB954 !important;
        color: black !important;
        font-weight: bold;
    }

    </style>
    """, unsafe_allow_html=True)

spotify_theme()


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
DATA_PATH = "data (2).csv"

@st.cache_data(show_spinner=True)
def load_data(path):
    df = pd.read_csv(path)

    # normalize
    df["name"] = df["name"].astype(str).str.lower().str.strip()
    df["artists"] = df["artists"].astype(str)

    # choose genre column
    genre_col = "genres" if "genres" in df.columns else "genre" if "genre" in df.columns else None
    df["genres_text"] = df[genre_col].astype(str) if genre_col else "unknown"

    # text field for TF-IDF
    df["text"] = df["name"] + " " + df["artists"] + " " + df["genres_text"]

    # duration
    if "duration_ms" in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration_ms"], errors="coerce") / 1000

    # audio fields
    audio_cols = [
        "danceability","energy","acousticness","instrumentalness","liveness",
        "speechiness","valence","tempo","loudness","duration_sec"
    ]
    audio_features = [c for c in audio_cols if c in df.columns]

    for col in audio_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    df["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0)

    return df, audio_features, genre_col


data, audio_features, genre_col = load_data(DATA_PATH)


# -------------------------------------------------------
# TEXT MODEL
# -------------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_text_model(texts, n_components=45):
    tfidf = TfidfVectorizer(max_features=15000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    knn = NearestNeighbors(metric="cosine")
    knn.fit(reduced)

    return tfidf, svd, knn, reduced


tfidf, svd, text_knn, text_emb = build_text_model(data["text"])


# -------------------------------------------------------
# AUDIO MODEL
# -------------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_audio_model(df, audio_cols):
    if not audio_cols:
        return None, None, None

    scaler = StandardScaler()
    X = scaler.fit_transform(df[audio_cols])

    knn = NearestNeighbors(metric="cosine")
    knn.fit(X)

    return scaler, knn, X

scaler_audio, audio_knn, audio_X = build_audio_model(data, audio_features)


# -------------------------------------------------------
# FUZZY MATCH
# -------------------------------------------------------
def find_best_match(name):
    choices = data["name"].tolist()
    match = difflib.get_close_matches(name.lower().strip(), choices, n=1, cutoff=0.55)
    return match[0] if match else None


# -------------------------------------------------------
# CLEAN RESULT
# -------------------------------------------------------
def clean_output(df):
    cols = ["name", "artists", genre_col, "popularity"] if genre_col else ["name","artists","popularity"]
    result = df[cols].copy()
    result.columns = ["Song", "Artist", "Genre", "Popularity"] if genre_col else ["Song","Artist","Popularity"]
    return result.sort_values(by="Popularity", ascending=False).reset_index(drop=True)


# -------------------------------------------------------
# RECOMMENDER FUNCTIONS
# -------------------------------------------------------
def recommend_text(song, top_n):
    match = find_best_match(song)
    if not match:
        return pd.DataFrame()

    idx = data.index[data["name"] == match][0]
    _, indices = text_knn.kneighbors([text_emb[idx]], n_neighbors=top_n+1)
    return clean_output(data.iloc[indices[0][1:]])


def recommend_audio(song, top_n):
    if audio_knn is None:
        return pd.DataFrame()

    match = find_best_match(song)
    if not match:
        return pd.DataFrame()

    idx = data.index[data["name"] == match][0]
    _, indices = audio_knn.kneighbors([audio_X[idx]], n_neighbors=top_n+1)
    return clean_output(data.iloc[indices[0][1:]])


def recommend_hybrid(song, top_n):
    t = recommend_text(song, top_n * 2)
    a = recommend_audio(song, top_n * 2)
    combined = pd.concat([t, a]).drop_duplicates("Song").head(top_n)
    return combined


# -------------------------------------------------------
# UI
# -------------------------------------------------------
st.title("🎧Music Recommendation System")

mode = st.sidebar.radio(
    "Choose Recommendation Mode",
    ["🎶 Text Based", "🔊 Audio Based", "🚀 Hybrid"],
    index=2
)

song_input = st.text_input("Enter song name:")
top_k = st.slider("Number of recommendations", 1, 20, 10)

if st.button("Get Recommendations"):
    if not song_input:
        st.error("Please enter a song name!")
    else:
        if mode == "🎶 Text Based":
            result = recommend_text(song_input, top_k)
        elif mode == "🔊 Audio Based":
            result = recommend_audio(song_input, top_k)
        else:
            result = recommend_hybrid(song_input, top_k)

        if result.empty:
            st.error("❌ No similar songs found. Try another input.")
        else:
            st.success("🎯 Recommendations for: " + song_input.title())
            st.dataframe(result, use_container_width=True)
