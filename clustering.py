
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from similarity import prepare_embedding_data


def cluster_words(n_clusters=5):

    # Load our word vectors
    words, vectors, word_to_index = prepare_embedding_data()

    # Validate the requested number of clusters
    if n_clusters < 1 or n_clusters > len(words):
        raise ValueError(
            "Number of clusters must be between 1 and the vocabulary size."
        )

    # Normalize vectors so their lengths are comparable
    normalized_vectors = normalize(vectors)

    # Create the K-Means model
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    # Assign every word to a cluster
    cluster_labels = model.fit_predict(normalized_vectors)

    # Organize the words by cluster
    clusters = {}

    for word, label in zip(words, cluster_labels):

        label = int(label)

        if label not in clusters:
            clusters[label] = []

        clusters[label].append(word)

    return clusters, model, normalized_vectors
