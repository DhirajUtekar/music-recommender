import streamlit as st
import pandas as pd
import numpy as np
import difflib
import base64
import io
import textwrap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data (2).csv"
DATA_BY_GENRE_PATH = "data_by_genres.csv"
DATA_BY_YEAR_PATH = "data_by_year.csv"

st.set_page_config(page_title="Music Recommender 🎧", page_icon="🎵", layout="wide")

st.markdown(
    """
    <style>
      :root { --accent: #FFB86B; --bg:#071023; --panel:#0b1320; --muted:#94a3b8; }
      .stApp { background: linear-gradient(180deg,#071023,#0b1320); color: #e6eef8; }
      .block-container { padding: 1.5rem 2rem; }
      .panel { background: rgba(255,255,255,0.03); padding: 16px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.03); }
      .title { color: var(--accent); font-size: 36px; font-weight: 800; text-shadow: 0 0 8px rgba(255,184,107,0.12); }
      .muted { color: var(--muted); }
      .btn { background-color: var(--accent) !important; color: #04121a !important; font-weight: 700; }
      .small { font-size: 0.9rem; color: var(--muted); }
      .song-card { background: linear-gradient(90deg, rgba(255,184,107,0.03), rgba(255,255,255,0.02)); padding: 12px; border-radius: 8px; margin-bottom: 6px; }
      .chip { display:inline-block; padding:4px 8px; border-radius:999px; background: rgba(255,255,255,0.04); margin-right:6px; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>Music Recommender 🎧</div>", unsafe_allow_html=True)
st.markdown("<div class='small'>Calm, explainable music recommendations — think like a human, choose with confidence.</div>", unsafe_allow_html=True)
st.markdown("---")

@st.cache_data(show_spinner=True)
def load_data(main_path=DATA_PATH, genre_path=DATA_BY_GENRE_PATH, year_path=DATA_BY_YEAR_PATH):
    df = pd.read_csv(main_path)
   
    df['name'] = df.get('name', df.index.astype(str)).astype(str).str.lower().str.strip()
    df['artists'] = df.get('artists', "").astype(str)
 
    if 'genres' in df.columns:
        df['genres_text'] = df['genres'].astype(str)
    elif 'genre' in df.columns:
        df['genres_text'] = df['genre'].astype(str)
    else:
        df['genres_text'] = "unknown"
  
    df['text'] = df['name'] + " " + df['artists'] + " " + df['genres_text']
   
    if 'duration_ms' in df.columns and 'duration_sec' not in df.columns:
        df['duration_sec'] = pd.to_numeric(df['duration_ms'], errors='coerce') / 1000.0
  
    audio_cols = [c for c in ['danceability','energy','acousticness','instrumentalness','liveness','speechiness','valence','tempo','loudness','duration_sec'] if c in df.columns]
    for c in audio_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].fillna(df[c].median(), inplace=True)
    df['popularity'] = pd.to_numeric(df.get('popularity', 0), errors='coerce').fillna(0)

    try:
        df_genre = pd.read_csv(genre_path)
    except Exception:
        df_genre = None
    try:
        df_year = pd.read_csv(year_path)
    except Exception:
        df_year = None
    return df, audio_cols, df_genre, df_year

data, audio_features, data_by_genres, data_by_year = load_data()


name_to_index = {}
for i, nm in enumerate(data['name'].values):
    if nm not in name_to_index:
        name_to_index[nm] = i


@st.cache_resource(show_spinner=False)
def infer_genres_if_missing(df, num_clusters=12):

    if 'genres_text' not in df.columns:
        return df
    unknown_rate = (df['genres_text'].astype(str).str.lower().isin(['', 'nan', 'none', '[]', 'unknown'])).mean()
    if unknown_rate < 0.2:
        return df
    small_tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    texts = (df['artists'].fillna('') + " " + df['name']).values
    M = small_tfidf.fit_transform(texts)
    k = min(num_clusters, max(2, M.shape[0]//1000))
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    labels = kmeans.fit_predict(M)
    df['inferred_genre'] = ["inferred_" + str(int(l)) for l in labels]
    
    df['genres_text'] = df.apply(lambda r: r['inferred_genre'] if str(r['genres_text']).lower() in ['','nan','none','[]','unknown'] else r['genres_text'], axis=1)
    return df

data = infer_genres_if_missing(data)


@st.cache_resource(show_spinner=True)
def build_text_index(texts, svd_dims=64):
    tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
    tfid = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(svd_dims, tfid.shape[1]-1), random_state=42)
    embed = svd.fit_transform(tfid)
    knn = NearestNeighbors(metric='cosine', n_jobs=-1)
    knn.fit(embed)
    return tfidf, svd, embed, knn

@st.cache_resource(show_spinner=True)
def build_audio_index(df, audio_cols, svd_dims=32):
    if not audio_cols:
        return None, None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(df[audio_cols])
    svd = TruncatedSVD(n_components=min(svd_dims, X.shape[1]-1), random_state=42)
    embed = svd.fit_transform(X)
    knn = NearestNeighbors(metric='cosine', n_jobs=-1)
    knn.fit(embed)
    return scaler, svd, embed, knn

tfidf, text_svd, text_embed, text_knn = build_text_index(data['text'].values, svd_dims=64)
audio_scaler, audio_svd, audio_embed, audio_knn = build_audio_index(data, audio_features, svd_dims=32)

def fuzzy_suggestions(query, k=6):
    if not query:
        return []
    q = query.lower().strip()
    substr = data[data['name'].str.contains(q, na=False)]
    if not substr.empty:
        return list(substr['name'].unique()[:k])

    return difflib.get_close_matches(q, data['name'].tolist(), n=k, cutoff=0.45)

def set_query_song(song_name):
 
    st.query_params = {"song": [song_name]}


def top_features_similarity(base_vec, cand_vecs, feature_names, top_k=3):

    diffs = cand_vecs - base_vec
    explanations = []
    for diff in diffs:
       
        sorted_feats = sorted(zip(feature_names, diff), key=lambda x: abs(x[1]))[:top_k]
        explanations.append(", ".join([f"{f} ({v:+.2f})" for f, v in sorted_feats]))
    return explanations

def recommend_text(song_lower, top_n=10):
    if song_lower not in name_to_index:
        return pd.DataFrame()
    idx = name_to_index[song_lower]
    dists, inds = text_knn.kneighbors([text_embed[idx]], n_neighbors=top_n+1)
    rec_idx = list(inds[0][1:])
    recs = data.iloc[rec_idx].copy().reset_index(drop=True)
    recs['score'] = (1 - dists[0][1:])
 
    if audio_features:
        base = data.loc[idx, audio_features].values.astype(float)
        cand_feats = recs[audio_features].values.astype(float)
        recs['why'] = top_features_similarity(base, cand_feats, audio_features, top_k=3)
    else:
        recs['why'] = "Text-similar"
    return recs

def recommend_audio(song_lower, top_n=10):
    if audio_knn is None or song_lower not in name_to_index:
        return pd.DataFrame()
    idx = name_to_index[song_lower]
    dists, inds = audio_knn.kneighbors([audio_embed[idx]], n_neighbors=top_n+1)
    rec_idx = list(inds[0][1:])
    recs = data.iloc[rec_idx].copy().reset_index(drop=True)
    recs['score'] = (1 - dists[0][1:])
   
    base = data.loc[idx, audio_features].values.astype(float)
    cand_feats = recs[audio_features].values.astype(float)
    recs['why'] = top_features_similarity(base, cand_feats, audio_features, top_k=3)
    return recs

def recommend_hybrid(song_lower, top_n=10, text_weight=0.6, audio_weight=0.4):

    idx = name_to_index.get(song_lower)
    if idx is None:
        return pd.DataFrame()
    t_d, t_i = text_knn.kneighbors([text_embed[idx]], n_neighbors=top_n*2+1)
    text_scores = {int(i): (1 - float(d)) * text_weight for d, i in zip(t_d[0][1:], t_i[0][1:])}
    audio_scores = {}
    if audio_knn is not None:
        a_d, a_i = audio_knn.kneighbors([audio_embed[idx]], n_neighbors=top_n*2+1)
        audio_scores = {int(i): (1 - float(d)) * audio_weight for d, i in zip(a_d[0][1:], a_i[0][1:])}

    agg = {}
    for k, v in text_scores.items():
        agg[k] = agg.get(k, 0) + v
    for k, v in audio_scores.items():
        agg[k] = agg.get(k, 0) + v
    sorted_idx = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ids = [i for i, _ in sorted_idx]
    recs = data.iloc[ids].copy().reset_index(drop=True)
    recs['score'] = [s for _, s in sorted_idx]
    if audio_features:
        base = data.loc[idx, audio_features].values.astype(float)
        cand_feats = recs[audio_features].values.astype(float)
        recs['why'] = top_features_similarity(base, cand_feats, audio_features, top_k=3)
    else:
        recs['why'] = "Hybrid similarity (text-weighted)"
    return recs


left, right = st.columns([1, 2], gap="small")

with left:
    st.markdown("### 1) Search")
    user_query = st.text_input("Song title or artist (partial ok)", value=st.query_params.get("song", [""])[0])

    if user_query:
        suggestions = fuzzy_suggestions(user_query, k=6)
        if suggestions:
            st.markdown("**Suggestions:**")
            for s in suggestions:
                if st.button(s, key="suggest_" + s):
                    set_query_song(s)
               
                    user_query = s

    st.markdown("### 2) Mood")
    mood = st.radio("How do you feel right now?", ["Calm / Chill", "Neutral", "Energetic / Party"], index=1)
    mood_map = {
        "Calm / Chill": {"energy": -0.35, "danceability": -0.2, "valence": -0.15},
        "Neutral": {"energy": 0.0, "danceability": 0.0, "valence": 0.0},
        "Energetic / Party": {"energy": 0.45, "danceability": 0.3, "valence": 0.2},
    }
    bias = mood_map[mood]

    st.markdown("### 3) Filters")
    
    available_genres = sorted(set([g.strip() for s in data['genres_text'].astype(str).unique() for g in str(s).split(',') if g and g.strip()]))
    selected_genres = st.multiselect("Filter by genre (optional)", options=available_genres, default=[])

    st.markdown("### 4) Mode & count")
    mode = st.selectbox("Recommendation mode", ["Hybrid (recommended)", "Text", "Audio"])
    k = st.slider("Number of recommendations", min_value=3, max_value=20, value=8)

    with st.expander("Advanced (rebuild embeddings)"):
        text_dims = st.number_input("Text SVD dims", min_value=16, max_value=256, value=64)
        audio_dims = st.number_input("Audio SVD dims", min_value=8, max_value=128, value=32)
        if st.button("Rebuild indexes (advanced)"):
            
            _ = build_text_index(data['text'].values, svd_dims=text_dims)
            _ = build_audio_index(data, audio_features, svd_dims=audio_dims)
            st.success("Rebuilt embeddings — done.")

    st.markdown("### Favorites")
    if 'favorites' not in st.session_state:
        st.session_state['favorites'] = []
    if st.button("Save current (best match) to favorites") and user_query:
        candidates = fuzzy_suggestions(user_query, k=1)
        if candidates:
            st.session_state['favorites'].append(candidates[0])
            st.success("Saved to favorites")
        else:
            st.error("No matching song to save")

    if st.session_state['favorites']:
        st.markdown("**Your favorites**")
        for f in st.session_state['favorites']:
            st.write("•  " + f.title())
        if st.button("Download favorites CSV"):
            df_fav = pd.DataFrame({"song": st.session_state['favorites']})
            csv = df_fav.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="favorites.csv">Download favorites.csv</a>'
            st.markdown(href, unsafe_allow_html=True)

with right:
    st.markdown("## Results")
    if not user_query:
        st.info("Start by typing a song title or artist on the left. Try partial text — we handle fuzzy matches.")
    else:
 
        candidates = fuzzy_suggestions(user_query, k=6)
        chosen = None
      
        qp = st.query_params
        if "song" in qp and qp["song"]:
            chosen = qp["song"][0]
        elif candidates:
            chosen = candidates[0]
        else:
            chosen = None

        if chosen is None:
            st.warning("No matches found for your input.")
        else:
            chosen = chosen.lower()
          
            if mode.startswith("Text"):
                recs = recommend_text(chosen, top_n=k*2)
            elif mode.startswith("Audio"):
                recs = recommend_audio(chosen, top_n=k*2)
            else:
                recs = recommend_hybrid(chosen, top_n=k*2)

         
            if selected_genres:
                recs = recs[recs['genres_text'].apply(lambda g: any(sel.lower() in g.lower() for sel in selected_genres))]
     
            if audio_features and not recs.empty:
                def mood_boost_row(row):
                    base_boost = row.get('score', 0)
                    boost = 0.0
                    for f, w in bias.items():
                        if f in audio_features:
                            val = float(row[f])
                            med = data[f].median()
                            rng = (data[f].max() - data[f].min()) if (data[f].max() != data[f].min()) else 1.0
                            norm = (val - med) / rng
                            boost += w * norm
                    return base_boost + 0.35 * boost
                recs['adj_score'] = recs.apply(mood_boost_row, axis=1)
                recs = recs.sort_values(by=['adj_score','score','popularity'], ascending=False)
            else:
                recs = recs.sort_values(by=['score','popularity'], ascending=False)

            recs = recs.head(k).reset_index(drop=True)


            q_idx = name_to_index.get(chosen)
            if q_idx is not None:
                qrow = data.iloc[q_idx]
                st.markdown(f"**You searched:** `{chosen.title()}` — mode: **{mode}**, mood: **{mood}**")
                st.caption(f"{qrow.get('artists','')} • {qrow.get('genres_text','')}")

                if 'preview_url' in data.columns and pd.notna(qrow.get('preview_url')):
                    st.audio(qrow.get('preview_url'))

       
            for i, r in recs.iterrows():
                st.markdown("<div class='song-card'>", unsafe_allow_html=True)
                cols = st.columns([0.08, 1, 0.35])
                with cols[0]:
                    st.markdown(f"**#{i+1}**")
                with cols[1]:
                    st.markdown(f"### {r['name'].title()}")
                    st.markdown(f"*{r.get('artists','')}*")
                    st.markdown(f"<span class='chip'>{r.get('genres_text','')}</span>", unsafe_allow_html=True)
                    if audio_features:
                        feat_snip = " • ".join([f"{f}:{r[f]:.2f}" for f in audio_features[:4]])
                        st.caption(feat_snip)
 
                    why = r.get('why', None)
                    if isinstance(why, (list, tuple)):
                        why_text = "; ".join(why)
                    else:
                        why_text = str(why)
                    st.markdown(f"**Why:** {why_text}")
                with cols[2]:
                    st.markdown(f"Popularity: **{int(r.get('popularity',0))}**")
                    if 'preview_url' in data.columns and pd.notna(r.get('preview_url')):
                        st.audio(r.get('preview_url'))
                    if st.button("Add to favorites", key=f"fav_{r['name']}_{i}"):
                        if r['name'] not in st.session_state['favorites']:
                            st.session_state['favorites'].append(r['name'])
                            st.success("Added to favorites")
                st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")
c1 = st.columns([1])[0]

with c1:
    st.markdown("### Quick insights")
    st.markdown(f"- Dataset songs: **{len(data):,}**")
    st.markdown(f"- Audio features used: **{', '.join(audio_features) if audio_features else 'None'}**")
    st.markdown(f"- Unique genres: **{len(set(data['genres_text'].astype(str)))}**")
