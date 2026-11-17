
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def analyze_contextual_movement(records, embeddings):

    embeddings = np.asarray(embeddings)

    if len(records) != len(embeddings):
        raise ValueError(
            "Number of records and embeddings must match."
        )

    if len(embeddings) < 2:
        raise ValueError(
            "At least two contextual embeddings are required."
        )

    results = []

    # Compare every context against the first context
    reference_vector = embeddings[0].reshape(1, -1)

    for index in range(len(embeddings)):

        current_vector = embeddings[index].reshape(1, -1)

        similarity = cosine_similarity(
            reference_vector,
            current_vector
        )[0][0]

        distance = np.linalg.norm(
            embeddings[0] - embeddings[index]
        )

        results.append({
            "context_number": index + 1,
            "sentence": records[index]["sentence"],
            "cosine_similarity": float(similarity),
            "vector_distance": float(distance)
        })

    return results


def calculate_consecutive_movement(embeddings):

    embeddings = np.asarray(embeddings)

    if len(embeddings) < 2:
        raise ValueError(
            "At least two embeddings are required."
        )

    movements = []

    for index in range(len(embeddings) - 1):

        vector1 = embeddings[index].reshape(1, -1)
        vector2 = embeddings[index + 1].reshape(1, -1)

        similarity = cosine_similarity(
            vector1,
            vector2
        )[0][0]

        distance = np.linalg.norm(
            embeddings[index] - embeddings[index + 1]
        )

        movements.append({
            "from_context": index + 1,
            "to_context": index + 2,
            "cosine_similarity": float(similarity),
            "vector_distance": float(distance)
        })

    return movements
