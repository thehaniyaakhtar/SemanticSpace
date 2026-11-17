
import numpy as np
from sklearn.decomposition import PCA
from contextual_embeddings import get_contextual_embedding


def build_contextual_dataset(sentences, target_word):
    if len(sentences) < 2:
        raise ValueError("At least two sentences are required.")

    embeddings = []
    records = []

    for index, sentence in enumerate(sentences):
        vector = get_contextual_embedding(sentence, target_word)
        embeddings.append(vector)

        records.append({
            "sentence": sentence,
            "word": target_word,
            "context_number": index + 1
        })

    embeddings = np.array(embeddings)

    dimensions = min(3, embeddings.shape[0] - 1, embeddings.shape[1])

    if dimensions < 1:
        raise ValueError("Not enough data for PCA.")

    reducer = PCA(n_components=dimensions)
    reduced = reducer.fit_transform(embeddings)

    coordinates = np.zeros((len(sentences), 3))
    coordinates[:, :dimensions] = reduced

    for index, record in enumerate(records):
        record["x"] = float(coordinates[index, 0])
        record["y"] = float(coordinates[index, 1])
        record["z"] = float(coordinates[index, 2])

    return records, embeddings, reducer
