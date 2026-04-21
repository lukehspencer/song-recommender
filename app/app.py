import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from model import recommend_songs
import os
 
# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Song Recommender",
    page_icon="🎵",
    layout="wide",
)
 
# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6f0;
}
 
.stApp {
    background: #0a0a0f;
}
 
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}
 
/* Header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
}
.hero h1 {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #e8e6f0 30%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #6b6880;
    font-size: 1.1rem;
    font-weight: 300;
}
 
/* Song input cards */
.song-card {
    background: #13121a;
    border: 1px solid #2a2735;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.song-card:hover {
    border-color: #a78bfa55;
}
.song-number {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.75rem;
}
 
/* Result cards */
.rec-card {
    background: linear-gradient(135deg, #13121a 0%, #1a1625 100%);
    border: 1px solid #2a2735;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.rec-rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #2a2735;
    min-width: 2.5rem;
}
.rec-info h4 {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.2rem 0;
    color: #e8e6f0;
}
.rec-info p {
    font-size: 0.85rem;
    color: #6b6880;
    margin: 0;
}
.sim-badge {
    margin-left: auto;
    background: #a78bfa22;
    border: 1px solid #a78bfa44;
    color: #a78bfa;
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    padding: 0.3rem 0.75rem;
    border-radius: 99px;
}
 
/* Section headers */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1.25rem;
    margin-top: 2.5rem;
}
 
/* Streamlit widget overrides */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #1a1825 !important;
    border: 1px solid #2a2735 !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
}
.stSlider > div > div > div > div {
    background: #a78bfa !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}
 
/* Divider */
hr {
    border-color: #2a2735 !important;
    margin: 2rem 0 !important;
}
</style>
""", unsafe_allow_html=True)
 
 
# ─── Load data ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, "data", "metadata.csv")
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df
 
FEATURE_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
 
# ─── UI ─────────────────────────────────────────────────────────────────────── ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎵 Song Recommender</h1>
    <p>Tell us what you love. We'll find what's next.</p>
</div>
""", unsafe_allow_html=True)
 
try:
    metadata_df = load_data()
    all_tracks = sorted(metadata_df["track_name"].dropna().unique().tolist())
    data_loaded = True
except Exception as e:
    st.error(f"Could not load dataset: {e}. Make sure `your_dataset.csv` is in the same directory.")
    data_loaded = False
 
if data_loaded:
    # ── Song inputs ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Your Songs</div>', unsafe_allow_html=True)
 
    songs_input, artists_input, ratings_input = [], [], []
    cols = st.columns(3)
 
    for i, col in enumerate(cols):
        with col:
            st.markdown(f'<div class="song-number">Song {i+1}</div>', unsafe_allow_html=True)
            song = st.selectbox(
                f"Track name",
                options=[""] + all_tracks,
                key=f"song_{i}",
                label_visibility="collapsed",
            )
            artist = st.text_input(
                "Artist",
                placeholder="Artist name",
                key=f"artist_{i}",
                label_visibility="collapsed",
            )
            rating = st.slider(
                "Rating",
                min_value=1, max_value=5, value=3,
                key=f"rating_{i}",
                help="1 = dislike, 5 = love it"
            )
            st.caption(f"{'⭐' * rating}")
            songs_input.append(song)
            artists_input.append(artist)
            ratings_input.append(rating)
 
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("✦ Find My Songs")
 
    # ── Results ──────────────────────────────────────────────────────────────
    if run:
        valid = [(s, a) for s, a in zip(songs_input, artists_input) if s and a]
 
        if len(valid) < 1:
            st.warning("Please enter at least one song and artist.")
        else:
            songs_v, artists_v, ratings_v = zip(*[
                (s, a, r) for s, a, r in zip(songs_input, artists_input, ratings_input) if s and a
            ])
 
            with st.spinner("Finding your next favorites..."):
                results = recommend_songs(
                    list(songs_v), list(artists_v), list(ratings_v), metadata_df
                )
 
            if results.empty:
                st.error("No recommendations found. Check that your song names and artists match the dataset exactly.")
            else:
                st.markdown('<div class="section-label">Recommended For You</div>', unsafe_allow_html=True)
 
                # Recommendation cards
                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    sim_pct = f"{row['similarity'] * 100:.0f}% match"
                    st.markdown(f"""
                    <div class="rec-card">
                        <div class="rec-rank">0{rank}</div>
                        <div class="rec-info">
                            <h4>{row['track_name']}</h4>
                            <p>{row['artists']}</p>
                        </div>
                        <div class="sim-badge">{sim_pct}</div>
                    </div>
                    """, unsafe_allow_html=True)
 
                st.markdown("---")
 
                # ── Feature table ────────────────────────────────────────────
                st.markdown('<div class="section-label">Feature Breakdown</div>', unsafe_allow_html=True)
                display_cols = ["track_name", "artists"] + FEATURE_COLS + ["similarity"]
                available_cols = [c for c in display_cols if c in results.columns]
                st.dataframe(
                    results[available_cols].reset_index(drop=True).style.format({
                        col: "{:.3f}" for col in FEATURE_COLS + ["similarity"] if col in results.columns
                    }).background_gradient(subset=["similarity"], cmap="Purples"),
                    use_container_width=True,
                    height=220,
                )
 
                st.markdown("---")
 
                # ── Visualizations ───────────────────────────────────────────
                st.markdown('<div class="section-label">Visualizations</div>', unsafe_allow_html=True)
 
                tab1, tab2, tab3 = st.tabs(["🟣 Feature Heatmap", "📊 Radar Chart", "🔵 Feature Distribution"])
 
                with tab1:
                    feat_data = results[FEATURE_COLS].copy()
                    feat_data.index = results["track_name"].values
                    fig_heat = px.imshow(
                        feat_data,
                        color_continuous_scale="RdPu",
                        aspect="auto",
                        title="Audio Feature Heatmap — Recommended Songs",
                        labels=dict(color="Scaled Value"),
                    )
                    fig_heat.update_layout(
                        paper_bgcolor="#0a0a0f",
                        plot_bgcolor="#0a0a0f",
                        font=dict(color="#e8e6f0", family="DM Sans"),
                        title_font=dict(family="Syne", size=16),
                        coloraxis_colorbar=dict(tickfont=dict(color="#e8e6f0")),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
 
                with tab2:
                    fig_radar = go.Figure()
                    colors = ["#a78bfa", "#f472b6", "#34d399", "#fb923c", "#60a5fa"]
                    for idx, (_, row) in enumerate(results.iterrows()):
                        values = [row[f] for f in FEATURE_COLS] + [row[FEATURE_COLS[0]]]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=values,
                            theta=FEATURE_COLS + [FEATURE_COLS[0]],
                            fill='toself',
                            name=row['track_name'],
                            line=dict(color=colors[idx % len(colors)], width=2),
                            fillcolor=f"rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.13)",
                        ))
                    fig_radar.update_layout(
                        polar=dict(
                            bgcolor="#13121a",
                            radialaxis=dict(visible=True, color="#2a2735", gridcolor="#2a2735"),
                            angularaxis=dict(color="#6b6880", gridcolor="#2a2735"),
                        ),
                        paper_bgcolor="#0a0a0f",
                        font=dict(color="#e8e6f0", family="DM Sans"),
                        title=dict(text="Audio Feature Radar", font=dict(family="Syne", size=16)),
                        legend=dict(bgcolor="#13121a", bordercolor="#2a2735"),
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
 
                with tab3:
                    feat_long = results[["track_name"] + FEATURE_COLS].melt(
                        id_vars="track_name", var_name="Feature", value_name="Value"
                    )
                    fig_bar = px.bar(
                        feat_long,
                        x="Feature", y="Value", color="track_name",
                        barmode="group",
                        title="Feature Comparison Across Recommended Songs",
                        color_discrete_sequence=["#a78bfa", "#f472b6", "#34d399", "#fb923c", "#60a5fa"],
                    )
                    fig_bar.update_layout(
                        paper_bgcolor="#0a0a0f",
                        plot_bgcolor="#0a0a0f",
                        font=dict(color="#e8e6f0", family="DM Sans"),
                        title_font=dict(family="Syne", size=16),
                        legend=dict(bgcolor="#13121a", bordercolor="#2a2735"),
                        xaxis=dict(gridcolor="#2a2735"),
                        yaxis=dict(gridcolor="#2a2735"),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)