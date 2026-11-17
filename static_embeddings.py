
import re
# to extract without punctuation
import numpy as np
from sklearn.decomposition import TruncatedSVD
# dimensionality reduction algorithm
# matrix with many columns, creates smaller represenation, preserve patterns


# --------------------------------------------------
# 1. LOCAL TRAINING CORPUS
# --------------------------------------------------

# raw sentences 
# examples of how words are used, in the context they are used
CORPUS = [
    "The king ruled the kingdom and governed the people.",
    "The queen ruled the kingdom beside the king.",
    "The prince was the son of the king.",
    "The princess was the daughter of the queen.",
    "The monarch lived inside the royal palace.",
    "The king wore a golden crown.",
    "The queen wore a beautiful crown.",

    "The man and woman walked through the city.",
    "The boy played with the girl in the garden.",
    "Every person is a human being.",
    "The child played with other children.",
    "A woman and man can become parents.",
    "The girl was a happy child.",

    "The computer runs different software programs.",
    "The programmer writes computer software.",
    "The internet connects computers across networks.",
    "Data travels through the computer network.",
    "The keyboard is used with a computer.",
    "Software processes information and data.",
    "The program connects to the internet.",

    "The river flows through the forest.",
    "The mountain stands above the valley.",
    "The ocean contains water and marine life.",
    "The tree grows inside the forest.",
    "Water flows from the mountain into the river.",
    "The earth contains forests, mountains and oceans.",
    "The river flows through the green valley.",

    "Love creates happiness and joy.",
    "Anger can produce sadness and fear.",
    "Happiness is associated with joy.",
    "Fear and anger are strong emotions.",
    "Love and happiness make people feel good.",
    "Sadness is an emotion experienced by humans.",
    "Joy and happiness are positive emotions.",
    "Fear is a natural human emotion.",

    "The king and queen lived in the palace.",
    "The prince wore a crown in the kingdom.",
    "The woman used a computer to write a program.",
    "The man worked with software and data.",
    "The child walked beside the river.",
    "The girl felt happiness and joy.",
    "The computer processed information from the internet."
]


# --------------------------------------------------
# 2. TEXT PREPROCESSING
# --------------------------------------------------

# removed during pre processing
STOP_WORDS = {
    "the", "a", "an", "and", "or", "is", "was",
    "were", "in", "on", "of", "to", "with", "from",
    "through", "beside", "inside", "into", "can",
    "be", "are", "used", "every", "other"
}

# accept collection of sentences
def preprocess_corpus(corpus):
    tokenized_sentences = []
    # store processed sentecnce
    for sentence in corpus:
        tokens = re.findall(
            r"[a-z]+",
            # tokenize
            sentence.lower()
        )

        tokens = [
            token for token in tokens
            if token not in STOP_WORDS
        ]
        # remove stop words

        tokenized_sentences.append(tokens)
    return tokenized_sentences


# --------------------------------------------------
# 3. BUILD WORD CO-OCCURRENCE MATRIX
# --------------------------------------------------

def build_cooccurrence_matrix(
    # coocurrance: 2 words appear near each other with specified context
    sentences,
    # preprocessed sentences
    window_size=3
    # how many words around the current word to examine
):

    vocabulary = sorted({
        word
        for sentence in sentences
        for word in sentence
    })
    # iterate through every sentence, every word within that sentence
    # create a set to remove duplicates
    # arrange the vocab aplhab

    word_to_index = {
        word: index
        for index, word in enumerate(vocabulary)
    }
    # assign an index to every word
    # a matrix works with numerical rows and cols indices not words

    # initilize the matrix
    # rows and cols represent words
    size = len(vocabulary)

    cooccurrence = np.zeros(
        (size, size),
        dtype=np.float64
    )

    # count neighboring words
    for sentence in sentences:
        for i, word in enumerate(sentence):
            word_index = word_to_index[word]
            start = max(0, i - window_size)
            end = min(len(sentence), i + window_size + 1)
            for j in range(start, end):
                if i == j:
                    continue
                context_word = sentence[j]
                context_index = word_to_index[context_word]
                distance = abs(i - j)
                cooccurrence[
                    word_index,
                    context_index
                ] += 1.0 / distance

    return vocabulary, cooccurrence


# --------------------------------------------------
# 4. CALCULATE PPMI
# --------------------------------------------------


# some words are repeated a lot because they are common words in the corpus
def calculate_ppmi(cooccurrence):
# positive pointwise mutual information
    total = cooccurrence.sum()
    # calculate total count
    # adds all values in the cooccurence matrix

    row_sums = cooccurrence.sum(axis=1)
    col_sums = cooccurrence.sum(axis=0)

    denominator = np.outer(
        row_sums,
        col_sums
    )

    # avoid division by 0
    denominator = np.maximum(
        denominator,
        1e-10
    )

    probabilities = (
        cooccurrence * total
    ) / denominator

    ppmi = np.maximum(
        np.log(np.maximum(probabilities, 1e-10)),
        0
    )

    return ppmi


# --------------------------------------------------
# 5. REDUCE DIMENSIONS FOR VISUALIZATION
# --------------------------------------------------
# dimensionality reduction using SVD

def create_latent_embeddings(ppmi):
    vocabulary_size = ppmi.shape[0]

    dimensions = min(20, vocabulary_size - 1)

    if dimensions < 1:
        raise ValueError("Not enough words to create embeddings.")

    embedding_reducer = TruncatedSVD(
        n_components=dimensions,
        random_state=42
    )

    embeddings = embedding_reducer.fit_transform(ppmi)

    return embeddings


def reduce_dimensions(embeddings):
    vocabulary_size = embeddings.shape[0]
    available_dimensions = embeddings.shape[1]

    dimensions = min(3, vocabulary_size - 1, available_dimensions)

    if dimensions < 1:
        raise ValueError("Not enough data to create a 3D projection.")

    reducer = TruncatedSVD(
        n_components=dimensions,
        random_state=42
    )

    reduced = reducer.fit_transform(embeddings)

    if dimensions < 3:
        coordinates = np.zeros((vocabulary_size, 3))
        coordinates[:, :dimensions] = reduced
    else:
        coordinates = reduced

    return coordinates, reducer


def load_static_embeddings():
    tokenized = preprocess_corpus(CORPUS)

    vocabulary, cooccurrence = build_cooccurrence_matrix(tokenized)

    ppmi = calculate_ppmi(cooccurrence)

    embeddings = create_latent_embeddings(ppmi)

    coordinates, reducer = reduce_dimensions(embeddings)

    records = []

    for index, word in enumerate(vocabulary):
        records.append({
            "word": word,
            "x": coordinates[index, 0],
            "y": coordinates[index, 1],
            "z": coordinates[index, 2],
            "vector": embeddings[index]
        })

    return records, vocabulary, ppmi, reducer