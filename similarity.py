
import numpy as np
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

from static_embeddings import load_static_embeddings


# --------------------------------------------------
# 1. LOAD AND CACHE EMBEDDING DATA
# --------------------------------------------------

@lru_cache(maxsize=1)
def prepare_embedding_data():

    records, vocabulary, ppmi, reducer = load_static_embeddings()

    words = [record["word"] for record in records]

    vectors = np.array(
        [record["vector"] for record in records],
        dtype=np.float64
    )

    word_to_index = {
        word: index
        for index, word in enumerate(words)
    }

    return words, vectors, word_to_index


# --------------------------------------------------
# 2. CALCULATE SIMILARITY BETWEEN TWO WORDS
# --------------------------------------------------

def calculate_similarity(word1, word2):

    words, vectors, word_to_index = prepare_embedding_data()

    word1 = word1.lower().strip()
    word2 = word2.lower().strip()

    if word1 not in word_to_index:
        raise ValueError(f"Word not found: {word1}")

    if word2 not in word_to_index:
        raise ValueError(f"Word not found: {word2}")

    index1 = word_to_index[word1]
    index2 = word_to_index[word2]

    vector1 = vectors[index1].reshape(1, -1)
    vector2 = vectors[index2].reshape(1, -1)

    score = cosine_similarity(vector1, vector2)[0][0]

    return float(score)


# --------------------------------------------------
# 3. FIND THE MOST SIMILAR WORDS
# --------------------------------------------------

def find_similar_words(target_word, top_n=5):

    words, vectors, word_to_index = prepare_embedding_data()

    target_word = target_word.lower().strip()

    if target_word not in word_to_index:
        raise ValueError(f"Word not found: {target_word}")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    target_index = word_to_index[target_word]

    target_vector = vectors[target_index].reshape(1, -1)

    similarities = cosine_similarity(
        target_vector,
        vectors
    )[0]

    ranked_indices = np.argsort(similarities)[::-1]

    results = []

    for index in ranked_indices:

        if index == target_index:
            continue

        results.append(
            (words[index], float(similarities[index]))
        )

        if len(results) == top_n:
            break

    return results
