import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from similarity import prepare_embedding_data


def find_analogy(word_a, word_b, word_c, top_n=5):
    words, vectors, word_to_index = prepare_embedding_data()

    word_a = word_a.lower().strip()
    word_b = word_b.lower().strip()
    word_c = word_c.lower().strip()

    input_words = [word_a, word_b, word_c]

    for word in input_words:
        if word not in word_to_index:
            raise ValueError(f"Word not found in vocabulary: {word}")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    # Retrieve the three word vectors
    vector_a = vectors[word_to_index[word_a]]
    vector_b = vectors[word_to_index[word_b]]
    vector_c = vectors[word_to_index[word_c]]

    # Perform vector arithmetic: A - B + C
    result_vector = vector_a - vector_b + vector_c

    # Compare the resulting vector with every word
    similarities = cosine_similarity(
        result_vector.reshape(1, -1),
        vectors
    )[0]

    # Rank words from highest to lowest similarity
    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices:
        candidate = words[index]

        # Exclude the three input words
        if candidate in input_words:
            continue

        results.append((
            candidate,
            float(similarities[index])
        ))

        if len(results) == top_n:
            break

    return results
