
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from static_embeddings import load_static_embeddings
from similarity import (
    prepare_embedding_data,
    calculate_similarity,
    find_similar_words
)
from analogy import find_analogy
from clustering import cluster_words
from exploration import explore_word

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="LEXIS | Language Universe",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# 2. DESIGN SYSTEM
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 15% 0%, #172044 0%, transparent 40%),
        radial-gradient(ellipse at 90% 15%, #241342 0%, transparent 35%),
        #080b17;
    color: #edf0ff;
    font-family: 'DM Sans', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f4f3ff !important;
}

p, label, .stMarkdown {
    color: #c3c9e0;
}

.hero {
    padding: 35px 38px;
    border-radius: 24px;
    background: linear-gradient(
        120deg,
        rgba(38, 49, 91, 0.88),
        rgba(39, 23, 73, 0.88),
        rgba(12, 31, 54, 0.92)
    );
    border: 1px solid rgba(145, 140, 255, 0.25);
    box-shadow: 0 15px 55px rgba(0, 0, 0, 0.25);
    margin-bottom: 25px;
}

.hero-label {
    color: #9d9aff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 58px;
    font-weight: 700;
    letter-spacing: -2px;
    color: #ffffff;
    margin: 8px 0;
}

.hero-subtitle {
    color: #c4c9ed;
    font-size: 17px;
    margin-top: 5px;
}

.section-label {
    color: #a69bff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.info-card {
    background: rgba(23, 29, 54, 0.75);
    border: 1px solid rgba(143, 145, 220, 0.18);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
}

.info-card h3 {
    margin-top: 0;
    font-size: 17px;
}

.info-card p {
    color: #aeb8d7;
    font-size: 14px;
}

div[data-testid="stMetric"] {
    background: rgba(25, 32, 61, 0.8);
    border: 1px solid rgba(139, 143, 225, 0.2);
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #aab4d7;
}

div[data-testid="stMetricValue"] {
    color: #a9a1ff;
    font-family: 'Space Grotesk', sans-serif;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(18, 23, 45, 0.85);
    padding: 9px;
    border-radius: 14px;
    border: 1px solid rgba(137, 139, 216, 0.15);
}

.stTabs [data-baseweb="tab"] {
    color: #aab3d2;
    border-radius: 9px;
    padding: 12px 15px;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    background: #343064 !important;
    color: #ffffff !important;
}

.stButton > button {
    background: linear-gradient(110deg, #7665e9, #a45de5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #c1a5ff;
    color: white;
    box-shadow: 0 0 20px rgba(137, 102, 255, 0.35);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stTextArea textarea {
    background: #151c35 !important;
    border-color: #41466e !important;
    color: white !important;
    border-radius: 10px !important;
}

.stSlider [data-baseweb="slider"] {
    color: #9a82ff;
}

hr {
    border-color: rgba(151, 150, 220, 0.16);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# 3. LOAD STATIC EMBEDDINGS
# ==========================================================

@st.cache_data
def get_static_data():
    return load_static_embeddings()

@st.cache_data
def get_word_data():
    words, vectors, word_to_index = prepare_embedding_data()
    return words, vectors, word_to_index

records, vocabulary, ppmi, reducer = get_static_data()
words, vectors, word_to_index = get_word_data()

# ==========================================================
# 4. SHARED PLOTLY DESIGN
# ==========================================================

def style_3d_figure(fig, height=650):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=0, r=0, b=0, t=20),
        font=dict(
            family="DM Sans",
            color="#c8cce8",
            size=12
        ),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            
            xaxis=dict(
                title="Dimension 1",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(117,137,194,0.20)",
                zerolinecolor="rgba(151,145,255,0.35)",
                showbackground=False
            ),
            yaxis=dict(
                title="Dimension 2",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(117,137,194,0.20)",
                zerolinecolor="rgba(151,145,255,0.35)",
                showbackground=False
            ),
            zaxis=dict(
                title="Dimension 3",
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(117,137,194,0.20)",
                zerolinecolor="rgba(151,145,255,0.35)",
                showbackground=False
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.25)
            ),
        )
    )

    return fig

# ==========================================================
# 5. HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">An interactive NLP experience</div>
    <div class="hero-title">LEXIS <span style="color:#a69bff">✦</span></div>
    <div class="hero-subtitle">
        An interactive universe of language.
        Explore the hidden geometry of meaning.
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

col1.metric("WORDS IN UNIVERSE", len(vocabulary))
col2.metric("VECTOR DIMENSIONS", vectors.shape[1])
col3.metric("LANGUAGE MODEL", "LEXIS / BERT")

st.markdown("---")

# ==========================================================
# 6. NAVIGATION
# ==========================================================

tabs = st.tabs([
    "🌌 Universe",
    "🔎 Explorer",
    "↔ Similarity",
    "🧮 Analogy",
    "🧩 Clusters",
    "🧠 Contextual Lab"
])


# ==========================================================
# TAB 1 — LEXIS COSMIC LANGUAGE UNIVERSE
# ==========================================================

with tabs[0]:

    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="
            color:#a995ff;
            letter-spacing:4px;
            font-size:11px;
            font-weight:700;">
            THE SEMANTIC GALAXY / 01
        </div>
        <div style="
            font-size:35px;
            font-weight:700;
            color:#f5f2ff;
            font-family:Space Grotesk,sans-serif;">
            Explore the universe of meaning.
        </div>
        <div style="color:#9da6c8;font-size:14px;">
            Every particle is a word. Every position represents
            its location in the projected embedding space.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------
    # SEARCH AND SELECTION
    # ------------------------------------------------------

    control_col, info_col = st.columns([1.4, 1])

    with control_col:

        search_word = st.selectbox(
            "Find a word in the universe",
            options=["None"] + vocabulary,
            key="universe_search"
        )

    selected_neighbours = []

    if search_word != "None":
        selected_neighbours = find_similar_words(search_word, 5)

    # ------------------------------------------------------
    # WORD COORDINATES
    # ------------------------------------------------------

    x = np.array([r["x"] for r in records], dtype=float)
    y = np.array([r["y"] for r in records], dtype=float)
    z = np.array([r["z"] for r in records], dtype=float)

    labels = [r["word"] for r in records]

    # ------------------------------------------------------
    # COORDINATE BOUNDS
    # ------------------------------------------------------

    x_range = max(np.ptp(x), 1.0)
    y_range = max(np.ptp(y), 1.0)
    z_range = max(np.ptp(z), 1.0)

    # ------------------------------------------------------
    # CREATE FIGURE
    # ------------------------------------------------------

    fig = go.Figure()

    # ------------------------------------------------------
    # BACKGROUND STARFIELD
    # Decorative particles only. They are NOT word embeddings.
    # ------------------------------------------------------

    rng = np.random.default_rng(42)

    star_count = 280

    star_x = rng.uniform(
        x.min() - x_range * 0.35,
        x.max() + x_range * 0.35,
        star_count
    )

    star_y = rng.uniform(
        y.min() - y_range * 0.35,
        y.max() + y_range * 0.35,
        star_count
    )

    star_z = rng.uniform(
        z.min() - z_range * 0.35,
        z.max() + z_range * 0.35,
        star_count
    )

    fig.add_trace(
        go.Scatter3d(
            x=star_x,
            y=star_y,
            z=star_z,
            mode="markers",
            name="Starfield",
            hoverinfo="skip",
            marker=dict(
                size=rng.uniform(1, 2.2, star_count),
                color="#aebdff",
                opacity=0.35
            ),
            showlegend=False
        )
    )

    # ------------------------------------------------------
    # BASE WORD PARTICLES
    # ------------------------------------------------------

    # Outer glow
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            hoverinfo="skip",
            showlegend=False,
            marker=dict(
                size=17,
                color="#38cfff",
                opacity=0.055
            )
        )
    )

    # Middle glow
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            hoverinfo="skip",
            showlegend=False,
            marker=dict(
                size=10,
                color="#39cfff",
                opacity=0.16
            )
        )
    )

    # Main word particles
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name="Words",
            text=labels,
            hovertemplate=(
                "<b style='color:#8feaff'>%{text}</b>"
                "<br>Semantic particle"
                "<extra></extra>"
            ),
            marker=dict(
                size=4.5,
                color="#65dfff",
                opacity=0.95,
                line=dict(
                    color="#d4f8ff",
                    width=0.5
                )
            )
        )
    )

    # ------------------------------------------------------
    # SELECTED WORD AND NEIGHBOURS
    # ------------------------------------------------------

    if search_word != "None":

        neighbour_words = [
            word for word, score in selected_neighbours
        ]

        neighbour_records = [
            r for r in records
            if r["word"] in neighbour_words
        ]

        selected_record = next(
            r for r in records
            if r["word"] == search_word
        )

        # Connection lines between selected word and neighbours
        for neighbour in neighbour_records:

            fig.add_trace(
                go.Scatter3d(
                    x=[
                        selected_record["x"],
                        neighbour["x"]
                    ],
                    y=[
                        selected_record["y"],
                        neighbour["y"]
                    ],
                    z=[
                        selected_record["z"],
                        neighbour["z"]
                    ],
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                    line=dict(
                        color="rgba(180,120,255,0.45)",
                        width=3
                    )
                )
            )

        # Neighbour glow
        fig.add_trace(
            go.Scatter3d(
                x=[r["x"] for r in neighbour_records],
                y=[r["y"] for r in neighbour_records],
                z=[r["z"] for r in neighbour_records],
                mode="markers",
                hoverinfo="skip",
                showlegend=False,
                marker=dict(
                    size=19,
                    color="#ad72ff",
                    opacity=0.15
                )
            )
        )

        # Neighbour particles
        fig.add_trace(
            go.Scatter3d(
                x=[r["x"] for r in neighbour_records],
                y=[r["y"] for r in neighbour_records],
                z=[r["z"] for r in neighbour_records],
                mode="markers+text",
                name="Semantic neighbours",
                text=[r["word"] for r in neighbour_records],
                textposition="top center",
                textfont=dict(
                    color="#d9b9ff",
                    size=11
                ),
                marker=dict(
                    size=7,
                    color="#b27aff",
                    opacity=1
                ),
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<extra></extra>"
                )
            )
        )

        # Selected word outer glow
        fig.add_trace(
            go.Scatter3d(
                x=[selected_record["x"]],
                y=[selected_record["y"]],
                z=[selected_record["z"]],
                mode="markers",
                hoverinfo="skip",
                showlegend=False,
                marker=dict(
                    size=29,
                    color="#ff5fcf",
                    opacity=0.19
                )
            )
        )

        # Selected word core
        fig.add_trace(
            go.Scatter3d(
                x=[selected_record["x"]],
                y=[selected_record["y"]],
                z=[selected_record["z"]],
                mode="markers+text",
                name="Selected word",
                text=[search_word],
                textposition="top center",
                textfont=dict(
                    color="#ffffff",
                    size=15
                ),
                marker=dict(
                    size=12,
                    color="#ff66ce",
                    opacity=1,
                    line=dict(
                        color="#ffffff",
                        width=2
                    )
                ),
                hovertemplate=(
                    "<b>%{text}</b>"
                    "<extra></extra>"
                )
            )
        )

    # ------------------------------------------------------
    # COSMIC SCENE DESIGN
    # ------------------------------------------------------

    fig.update_layout(
        height=620,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#17152d",
            bordercolor="#9878ff",
            font=dict(
                color="#ffffff",
                size=13
            )
        ),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(
                eye=dict(
                    x=1.5,
                    y=1.5,
                    z=1.15
                )
            ),

            # Hide the mathematical box and grid.
            xaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            ),

            yaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            ),

            zaxis=dict(
                visible=False,
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=""
            )
        )
    )

    # ------------------------------------------------------
    # DISPLAY UNIVERSE
    # ------------------------------------------------------

    st.markdown("""
    <div style="
        border:1px solid rgba(135,120,255,0.22);
        border-radius:18px;
        padding:5px;
        background:rgba(10,12,30,0.45);
        box-shadow:0 0 45px rgba(100,70,210,0.10);">
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "displaylogo": False
        }
    )

    # ------------------------------------------------------
    # SELECTED WORD INFORMATION
    # ------------------------------------------------------

    if search_word != "None":

        st.markdown("### ✦ Your selected word")

        st.markdown(
            f"""
            <div style="
                background:linear-gradient(
                    120deg,
                    rgba(87,53,150,0.35),
                    rgba(23,30,66,0.75)
                );
                border:1px solid rgba(180,130,255,0.35);
                padding:20px;
                border-radius:16px;
                margin-bottom:15px;">
                <div style="
                    color:#bba2ff;
                    font-size:12px;
                    letter-spacing:2px;">
                    SELECTED PARTICLE
                </div>
                <div style="
                    color:white;
                    font-size:30px;
                    font-weight:700;">
                    {search_word}
                </div>
                <div style="
                    color:#b9c2e4;
                    font-size:13px;">
                    {len(selected_neighbours)}
                    nearest semantic neighbours highlighted
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Semantic connections")

        for word, score in selected_neighbours:

            st.write(
                f"✦ **{word}** "
                f"<span style='color:#a995ff'>"
                f"{score:.4f}</span>",
                unsafe_allow_html=True
            )

    else:

        st.markdown("""
        <div style="
            text-align:center;
            color:#8d9abf;
            padding:15px;">
            ✦ Select a word above to reveal its semantic connections.
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "The starfield is decorative. Word particles retain their "
        "original projected embedding coordinates. "
        "Cosine similarity is calculated in the original vector space."
    )

# ==========================================================
# TAB 2 — WORD EXPLORER
# ==========================================================

with tabs[1]:

    st.markdown('<div class="section-label">02 / Investigate</div>',
                unsafe_allow_html=True)

    st.header("Word Explorer")

    st.write(
        "Investigate a word's representation and discover "
        "which other words occupy nearby semantic space."
    )

    target_word = st.selectbox(
        "Choose a word",
        vocabulary,
        key="explorer_word"
    )

    top_n = st.slider(
        "Number of neighbours",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button("Explore word", key="explore_button"):

        try:

            result = explore_word(
                target_word,
                top_n=top_n
            )

            st.markdown(f"### Exploring: **{result['word']}**")

            st.metric(
                "Embedding dimensions",
                result["embedding_dimensions"]
            )

            st.markdown("### Nearest semantic neighbours")

            for word, score in result["similar_words"]:

                st.progress(
                    float(np.clip((score + 1) / 2, 0, 1)),
                    text=f"{word} — {score:.4f}"
                )

            st.markdown("### Cluster membership")

            if result["cluster"]:

                st.info(
                    f"Cluster {result['cluster']['cluster_number']}"
                )

                st.write(
                    ", ".join(result["cluster"]["words"])
                )

            else:

                st.write("No cluster found.")

        except Exception as error:

            st.error(str(error))

# ==========================================================
# TAB 3 — SIMILARITY
# ==========================================================

with tabs[2]:

    st.markdown('<div class="section-label">03 / Compare</div>',
                unsafe_allow_html=True)

    st.header("Similarity Lab")

    st.write(
        "Measure how closely two words are represented "
        "in the embedding space."
    )

    col1, col2 = st.columns(2)

    with col1:

        word1 = st.selectbox(
            "First word",
            vocabulary,
            key="similarity_word1"
        )

    with col2:

        word2 = st.selectbox(
            "Second word",
            vocabulary,
            index=min(1, len(vocabulary) - 1),
            key="similarity_word2"
        )

    if st.button("Calculate similarity"):

        try:

            score = calculate_similarity(word1, word2)

            st.markdown("### Result")

            col1, col2 = st.columns(2)

            col1.metric("Cosine similarity", f"{score:.4f}")
            col2.metric("Compared words", f"{word1} ↔ {word2}")

            st.progress(
                float(np.clip((score + 1) / 2, 0, 1))
            )

            st.caption(
                "Cosine similarity ranges from -1 to 1. "
                "Higher values indicate more similar vector directions."
            )

        except Exception as error:

            st.error(str(error))

# ==========================================================
# TAB 4 — ANALOGY LAB
# ==========================================================

with tabs[3]:

    st.markdown('<div class="section-label">04 / Experiment</div>',
                unsafe_allow_html=True)

    st.header("Analogy Lab")

    st.write(
        "Explore relationships between words using vector arithmetic."
    )

    st.latex(r"A - B + C = ?")

    col1, col2, col3 = st.columns(3)

    with col1:

        analogy_a = st.selectbox(
            "Word A",
            vocabulary,
            key="analogy_a"
        )

    with col2:

        analogy_b = st.selectbox(
            "Word B",
            vocabulary,
            key="analogy_b"
        )

    with col3:

        analogy_c = st.selectbox(
            "Word C",
            vocabulary,
            key="analogy_c"
        )

    if st.button("Solve analogy"):

        try:

            results = find_analogy(
                analogy_a,
                analogy_b,
                analogy_c,
                top_n=5
            )

            st.markdown("### Closest results")

            for index, (word, score) in enumerate(results, 1):

                st.write(
                    f"**{index}. {word}** — {score:.4f}"
                )

        except Exception as error:

            st.error(str(error))

    st.info(
        "These are experimental results from a small local corpus. "
        "They are not guaranteed to reproduce human-style analogies."
    )

# ==========================================================
# TAB 5 — CLUSTERS
# ==========================================================

with tabs[4]:

    st.markdown('<div class="section-label">05 / Discover</div>',
                unsafe_allow_html=True)

    st.header("Semantic Clusters")

    st.write(
        "Discover groups of words that share similar "
        "representations in the embedding space."
    )

    number_of_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=5
    )

    if st.button("Generate clusters"):

        try:

            clusters, model, normalized_vectors = cluster_words(
                number_of_clusters
            )

            st.markdown("### Discovered word groups")

            for cluster_number, cluster_words_list in clusters.items():

                with st.expander(
                    f"Cluster {cluster_number + 1} "
                    f"({len(cluster_words_list)} words)"
                ):

                    st.write(", ".join(cluster_words_list))

            word_to_cluster = {}

            for cluster_number, cluster_words_list in clusters.items():

                for word in cluster_words_list:

                    word_to_cluster[word] = cluster_number

            cluster_labels = [
                word_to_cluster[record["word"]]
                for record in records
            ]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter3d(
                    x=[r["x"] for r in records],
                    y=[r["y"] for r in records],
                    z=[r["z"] for r in records],
                    mode="markers",
                    text=[r["word"] for r in records],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Cluster: %{marker.color}"
                        "<extra></extra>"
                    ),
                    marker=dict(
                        size=6,
                        color=cluster_labels,
                        colorscale="Turbo",
                        opacity=0.9,
                        colorbar=dict(
                            title="Cluster"
                        )
                    )
                )
            )

            fig = style_3d_figure(fig, 650)

            st.markdown("### Cluster universe")

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Clustering happens in normalized embedding space. "
                "The 3D chart is a visual projection."
            )

        except Exception as error:

            st.error(str(error))

# ==========================================================
# TAB 6 — CONTEXTUAL LAB
# ==========================================================

with tabs[5]:

    st.markdown('<div class="section-label">06 / Understand context</div>',
                unsafe_allow_html=True)

    st.header("Contextual Embedding Lab")

    st.write(
        "Explore how the representation of a word changes "
        "depending on the sentence in which it appears."
    )

    st.info(
        "Powered by BERT. This feature requires PyTorch, "
        "Transformers and the downloaded BERT model."
    )

    target_context_word = st.text_input(
        "Target word",
        value="bank"
    )

    sentence_input = st.text_area(
        "Enter one sentence per line",
        value=(
            "I deposited my money in the bank.\n"
            "The river overflowed its bank after heavy rain.\n"
            "She works at a bank in the city.\n"
            "The boat reached the muddy river bank.\n"
            "He sat on the bank and watched the water."
        ),
        height=180
    )

    if st.button("Analyze contexts"):

        sentences = [
            sentence.strip()
            for sentence in sentence_input.splitlines()
            if sentence.strip()
        ]

        if len(sentences) < 2:

            st.error("Please enter at least two sentences.")

        elif not target_context_word.strip():

            st.error("Please enter a target word.")

        else:

            try:

                # Use the actual project filename:
                # context_visualization.py

                from context_visualization import (
                    build_contextual_dataset
                )

                from contextual_movement import (
                    analyze_contextual_movement
                )

                with st.spinner(
                    "Generating BERT embeddings..."
                ):

                    context_records, context_embeddings, pca = (
                        build_contextual_dataset(
                            sentences,
                            target_context_word.strip()
                        )
                    )

                    movement_results = analyze_contextual_movement(
                        context_records,
                        context_embeddings
                    )

                st.success("Contextual embeddings generated!")

                st.metric(
                    "BERT embedding dimensions",
                    context_embeddings.shape[1]
                )

                st.markdown("### Contextual universe")

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter3d(
                        x=[r["x"] for r in context_records],
                        y=[r["y"] for r in context_records],
                        z=[r["z"] for r in context_records],
                        mode="lines+markers+text",
                        text=[
                            f"Context {r['context_number']}"
                            for r in context_records
                        ],
                        textposition="top center",
                        customdata=[
                            [r["sentence"]]
                            for r in context_records
                        ],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "%{customdata[0]}"
                            "<extra></extra>"
                        ),
                        line=dict(
                            color="#8c72ff",
                            width=5
                        ),
                        marker=dict(
                            size=9,
                            color=list(
                                range(1, len(context_records) + 1)
                            ),
                            colorscale="Viridis",
                            opacity=1,
                            line=dict(
                                color="#ffffff",
                                width=1
                            ),
                            colorbar=dict(
                                title="Context"
                            )
                        )
                    )
                )

                fig = style_3d_figure(fig, 600)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown("### Contextual movement")

                for result in movement_results:

                    with st.expander(
                        f"Context {result['context_number']}: "
                        f"{result['sentence']}"
                    ):

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Similarity to first context",
                            f"{result['cosine_similarity']:.4f}"
                        )

                        col2.metric(
                            "Vector distance",
                            f"{result['vector_distance']:.4f}"
                        )

                st.caption(
                    "The first context is used as the reference "
                    "for the movement measurements. "
                    "The 3D positions are a shared PCA projection."
                )

            except ImportError as error:

                st.error(
                    f"A required module could not be imported: {error}"
                )

            except Exception as error:

                st.error(
                    f"Contextual analysis could not be completed: {error}"
                )



# ==========================================================
# LEXIS BRAND FOOTER
# ==========================================================

st.markdown("""
<div style="
    margin-top: 55px;
    padding: 35px 10px;
    border-top: 1px solid rgba(169,154,255,0.2);
    text-align: center;
    background: transparent;
">
    <div style="
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        color: #a99aff;
        letter-spacing: 2px;
    ">
        LEXIS ✦
    </div>

    <div style="
        color: #a0a8c8;
        font-size: 14px;
        margin-top: 9px;
    ">
        An Interactive Universe of Language
    </div>

    <div style="
        color: #68718f;
        font-size: 11px;
        margin-top: 14px;
    ">
        Built with Python · Streamlit · BERT · Plotly
    </div>
</div>
""", unsafe_allow_html=True)
