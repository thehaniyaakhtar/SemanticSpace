
import plotly.graph_objects as go

from static_embeddings import load_static_embeddings
from clustering import cluster_words


def create_cluster_plot(n_clusters=5):

    records, vocabulary, ppmi, reducer = load_static_embeddings()

    clusters, model, vectors = cluster_words(n_clusters)

    # Create a word-to-cluster lookup
    word_to_cluster = {}

    for cluster_number, words in clusters.items():

        for word in words:
            word_to_cluster[word] = cluster_number

    # Extract coordinates and cluster labels
    x = [record["x"] for record in records]
    y = [record["y"] for record in records]
    z = [record["z"] for record in records]

    words = [record["word"] for record in records]

    labels = [
        word_to_cluster[word]
        for word in words
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,

            mode="markers+text",

            text=words,

            textposition="top center",

            marker=dict(
                size=5,
                color=labels,
                colorscale="Viridis",
                opacity=0.85,
                colorbar=dict(title="Cluster")
            ),

            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cluster: %{marker.color}"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        title="LEXIS | Semantic Word Clusters",

        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),

        template="plotly_dark",
        height=750
    )

    return figure
