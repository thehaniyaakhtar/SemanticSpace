
import numpy as np

from sklearn.decomposition import PCA

from contextual_embeddings import get_contextual_embedding


def build_contextual_dataset(sentences, target_word):

    embeddings = []
    records = []

    for index, sentence in enumerate(sentences):

        # Generate the contextual representation
        vector = get_contextual_embedding(
            sentence,
            target_word
        )

        embeddings.append(vector)

        records.append({
            "sentence": sentence,
            "word": target_word,
            "context_number": index + 1
        })

    # Convert embeddings into a matrix
    embeddings = np.array(embeddings)

    # Determine the number of available PCA dimensions
    dimensions = min(
        3,
        embeddings.shape[0] - 1,
        embeddings.shape[1]
    )

    if dimensions < 1:
        raise ValueError(
            "At least two sentences are required."
        )

    # Fit ONE PCA model across all contextual embeddings
    reducer = PCA(n_components=dimensions)

    reduced = reducer.fit_transform(embeddings)

    # Ensure every record has three coordinates
    coordinates = np.zeros((len(sentences), 3))

    coordinates[:, :dimensions] = reduced

    # Attach coordinates to records
    for index, record in enumerate(records):

        record["x"] = float(coordinates[index, 0])
        record["y"] = float(coordinates[index, 1])
        record["z"] = float(coordinates[index, 2])

    return records, embeddings, reducer
