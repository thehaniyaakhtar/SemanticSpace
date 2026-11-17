
from similarity import (
    prepare_embedding_data,
    find_similar_words,
    calculate_similarity
)

from clustering import cluster_words


def explore_word(target_word, top_n=5, n_clusters=5):

    target_word = target_word.lower().strip()

    words, vectors, word_to_index = prepare_embedding_data()

    if target_word not in word_to_index:
        raise ValueError(
            f"Word not found: {target_word}"
        )

    # Find similar words
    similar_words = find_similar_words(
        target_word,
        top_n
    )

    # Find the cluster containing the target word
    clusters, model, normalized_vectors = cluster_words(
        n_clusters
    )

    target_cluster = None

    for cluster_number, cluster_words_list in clusters.items():

        if target_word in cluster_words_list:
            target_cluster = {
                "cluster_number": cluster_number + 1,
                "words": cluster_words_list
            }
            break

    # Retrieve the target word vector
    target_vector = vectors[
        word_to_index[target_word]
    ]

    return {
        "word": target_word,
        "embedding_dimensions": len(target_vector),
        "embedding": target_vector,
        "similar_words": similar_words,
        "cluster": target_cluster
    }


def compare_words(word1, word2):

    score = calculate_similarity(
        word1,
        word2
    )

    return {
        "word_1": word1,
        "word_2": word2,
        "similarity": score
    }
