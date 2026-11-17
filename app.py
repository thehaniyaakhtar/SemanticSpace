
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


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="LEXIS | Language Universe",
    page_icon="🌌",
    layout="wide"
)


# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: #090d18;
    color: #edf2ff;
}

h1, h2, h3 {
    color: #edf2ff;
}

[data-testid="stSidebar"] {
    background: #10182a;
}

.stMetric {
    background: #111b2e;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD STATIC EMBEDDINGS
# --------------------------------------------------

@st.cache_data
def get_static_data():
    return load_static_embeddings()


@st.cache_data
def get_word_data():
    words, vectors, word_to_index = prepare_embedding_data()
    return words, vectors, word_to_index


records, vocabulary, ppmi, reducer = get_static_data()
words, vectors, word_to_index = get_word_data()


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.title("🌌 LEXIS")
st.markdown(
    "### An Interactive Universe of Language"
)
st.caption(
    "Explore how words are represented, compared, "
    "grouped, and understood through context."
)

st.divider()


# --------------------------------------------------
# 5. NAVIGATION
# --------------------------------------------------

tabs = st.tabs([
    "🌍 Language Universe",
    "🔎 Word Explorer",
    "↔️ Similarity",
    "🧮 Analogy Lab",
    "🧩 Clusters",
    "🧠 Contextual Lab"
])


# --------------------------------------------------
# TAB 1 — LANGUAGE UNIVERSE
# --------------------------------------------------

with tabs[0]:

    st.header("The Language Universe")

    st.write(
        "Every point represents a word. "
        "Words are positioned using a 3D projection "
        "of their learned static embeddings."
    )

    search_word = st.selectbox(
        "Highlight a word",
        options=["None"] + vocabulary
    )

    x = [record["x"] for record in records]
    y = [record["y"] for record in records]
    z = [record["z"] for record in records]

    labels = [record["word"] for record in records]

    colors = [
        "#ff4b91" if word == search_word else "#55b8ff"
        for word in labels
    ]

    sizes = [
        9 if word == search_word else 5
        for word in labels
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.85
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "X: %{x:.3f}<br>"
                    "Y: %{y:.3f}<br>"
                    "Z: %{z:.3f}"
                    "<extra></extra>"
                )
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#090d18",
        plot_bgcolor="#090d18",
        height=650,
        scene=dict(
            bgcolor="#090d18",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        margin=dict(l=0, r=0, b=0, t=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    col1.metric("Vocabulary size", len(vocabulary))
    col2.metric("Embedding dimensions", vectors.shape[1])
    col3.metric("Words visualized", len(records))

    st.info(
        "The 3D coordinates are a compressed visual representation. "
        "Similarity calculations use the original 20-dimensional vectors."
    )


# --------------------------------------------------
# TAB 2 — WORD EXPLORER
# --------------------------------------------------

with tabs[1]:

    st.header("Word Explorer")

    target_word = st.selectbox(
        "Choose a word to investigate",
        vocabulary,
        key="explorer_word"
    )

    top_n = st.slider(
        "Number of similar words",
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

            st.subheader(f"Exploring: {result['word']}")

            st.metric(
                "Embedding dimensions",
                result["embedding_dimensions"]
            )

            st.write("### Nearest semantic neighbours")

            for word, score in result["similar_words"]:
                st.write(f"**{word}** — {score:.4f}")

            st.write("### Cluster membership")

            if result["cluster"]:
                st.write(
                    f"Cluster {result['cluster']['cluster_number']}"
                )
                st.write(
                    ", ".join(result["cluster"]["words"])
                )
            else:
                st.write("No cluster found.")

        except Exception as error:
            st.error(str(error))


# --------------------------------------------------
# TAB 3 — SIMILARITY
# --------------------------------------------------

with tabs[2]:

    st.header("Word Similarity Lab")

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

            st.metric(
                "Cosine similarity",
                f"{score:.4f}"
            )

            st.progress(
                float(np.clip((score + 1) / 2, 0, 1))
            )

            st.caption(
                "Cosine similarity ranges from -1 to 1. "
                "A higher score means the vectors point in more similar directions."
            )

        except Exception as error:
            st.error(str(error))


# --------------------------------------------------
# TAB 4 — ANALOGY LAB
# --------------------------------------------------

with tabs[3]:

    st.header("Analogy Lab")

    st.write(
        "Explore vector arithmetic using the relationship:"
    )

    st.latex(r"A - B + C = ?")

    col1, col2, col3 = st.columns(3)

    with col1:
        analogy_a = st.selectbox(
            "A",
            vocabulary,
            key="analogy_a"
        )

    with col2:
        analogy_b = st.selectbox(
            "B",
            vocabulary,
            key="analogy_b"
        )

    with col3:
        analogy_c = st.selectbox(
            "C",
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

            st.subheader("Closest results")

            for index, (word, score) in enumerate(results, 1):
                st.write(
                    f"{index}. **{word}** — {score:.4f}"
                )

        except Exception as error:
            st.error(str(error))

    st.info(
        "These are experimental results from our small local corpus. "
        "They are not guaranteed to reproduce human-style analogies."
    )


# --------------------------------------------------
# TAB 5 — CLUSTERS
# --------------------------------------------------

with tabs[4]:

    st.header("Semantic Clusters")

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

            st.subheader("Discovered word groups")

            for cluster_number, cluster_words_list in clusters.items():

                with st.expander(
                    f"Cluster {cluster_number + 1} "
                    f"({len(cluster_words_list)} words)"
                ):
                    st.write(", ".join(cluster_words_list))

            # Map each word to its cluster.
            word_to_cluster = {}

            for cluster_number, cluster_words_list in clusters.items():
                for word in cluster_words_list:
                    word_to_cluster[word] = cluster_number

            cluster_labels = [
                word_to_cluster[record["word"]]
                for record in records
            ]

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=[record["x"] for record in records],
                        y=[record["y"] for record in records],
                        z=[record["z"] for record in records],
                        mode="markers+text",
                        text=[record["word"] for record in records],
                        textposition="top center",
                        marker=dict(
                            size=5,
                            color=cluster_labels,
                            colorscale="Turbo",
                            opacity=0.85
                        )
                    )
                ]
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#090d18",
                height=600,
                scene=dict(bgcolor="#090d18")
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Clustering happens in the normalized 20-dimensional "
                "embedding space. The 3D chart is only a visual projection."
            )

        except Exception as error:
            st.error(str(error))


# --------------------------------------------------
# TAB 6 — CONTEXTUAL LAB
# --------------------------------------------------

with tabs[5]:

    st.header("Contextual Embedding Lab")

    st.write(
        "Investigate how the representation of a word changes "
        "depending on the sentence in which it appears."
    )

    st.info(
        "This feature requires PyTorch, Transformers, and the "
        "BERT model. It can run separately from the static features."
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
                # Import only when this feature is used.
                from contextual_visualization import (
                    build_contextual_dataset
                )

                from contextual_movement import (
                    analyze_contextual_movement
                )

                with st.spinner(
                    "Loading BERT and analyzing sentences..."
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

                st.subheader("Context positions")

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
                        marker=dict(size=8)
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#090d18",
                    height=600,
                    scene=dict(bgcolor="#090d18")
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Contextual movement")

                for result in movement_results:

                    st.write(
                        f"**Context {result['context_number']}**: "
                        f"{result['sentence']}"
                    )

                    st.write(
                        f"Similarity to first context: "
                        f"{result['cosine_similarity']:.4f}"
                    )

                    st.write(
                        f"Vector distance from first context: "
                        f"{result['vector_distance']:.4f}"
                    )

                    st.divider()

            except ImportError:
                st.error(
                    "BERT dependencies are missing. "
                    "Install PyTorch and Transformers on your personal computer "
                    "to use this feature."
                )

            except Exception as error:
                st.error(
                    f"Contextual analysis could not be completed: {error}"
                )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "LEXIS | Experimental language representation explorer"
)
