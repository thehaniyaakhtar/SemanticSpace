
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel


# --------------------------------
# 1. Load the pretrained BERT model
# --------------------------------

MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


# --------------------------------
# 2. Generate contextual embeddings
# --------------------------------

def get_contextual_embedding(sentence, target_word):

    target_word = target_word.lower()

    # Tokenize the sentence
    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    )

    # Extract token IDs
    input_ids = inputs["input_ids"][0]

    # Convert token IDs back into tokens
    tokens = tokenizer.convert_ids_to_tokens(
        input_ids
    )

    # Identify the target word's token positions
    target_positions = []

    for index, token in enumerate(tokens):

        cleaned_token = token.replace("##", "")

        if cleaned_token == target_word:
            target_positions.append(index)

    if not target_positions:
        raise ValueError(
            f"Target word '{target_word}' not found in sentence."
        )

    # Generate contextual representations
    with torch.no_grad():

        outputs = model(**inputs)

    # Last hidden layer
    hidden_states = outputs.last_hidden_state

    # Extract representations for the target word
    target_vectors = hidden_states[
        0, target_positions, :
    ]

    # Average the target word's token vectors
    word_embedding = target_vectors.mean(dim=0)

    return word_embedding.numpy()


# --------------------------------
# 3. Compare two contexts
# --------------------------------

def compare_contexts(sentence1, sentence2, target_word):

    vector1 = get_contextual_embedding(
        sentence1,
        target_word
    )

    vector2 = get_contextual_embedding(
        sentence2,
        target_word
    )

    return vector1, vector2


# --------------------------------
# 4. Calculate contextual similarity
# --------------------------------

def contextual_similarity(sentence1, sentence2, target_word):

    vector1, vector2 = compare_contexts(
        sentence1,
        sentence2,
        target_word
    )

    vector1 = vector1.reshape(1, -1)
    vector2 = vector2.reshape(1, -1)

    from sklearn.metrics.pairwise import cosine_similarity

    score = cosine_similarity(
        vector1,
        vector2
    )[0][0]

    return float(score)


# --------------------------------
# 5. Compare word representations
# --------------------------------

def analyze_contexts(sentence1, sentence2, target_word):

    vector1, vector2 = compare_contexts(
        sentence1,
        sentence2,
        target_word
    )

    similarity = contextual_similarity(
        sentence1,
        sentence2,
        target_word
    )

    movement = np.linalg.norm(
        vector1 - vector2
    )

    return {
        "word": target_word,
        "sentence_1": sentence1,
        "sentence_2": sentence2,
        "embedding_1": vector1,
        "embedding_2": vector2,
        "cosine_similarity": similarity,
        "vector_distance": float(movement)
    }
