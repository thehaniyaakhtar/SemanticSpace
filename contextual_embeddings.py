
import re
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. LOAD PRETRAINED BERT MODEL
# --------------------------------------------------

MODEL_NAME = "bert-base-uncased"

print("Loading BERT model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

print("BERT model loaded successfully!")


# --------------------------------------------------
# 2. EXTRACT CONTEXTUAL WORD EMBEDDING
# --------------------------------------------------

def get_contextual_embedding(sentence, target_word, occurrence=0):
    """
    Extracts the contextual embedding of a target word
    from a sentence using pretrained BERT.

    If the target word appears more than once, occurrence
    selects which appearance to extract (starting at 0).

    Returns:
        A NumPy array containing a 768-dimensional vector.
    """

    target_word = target_word.lower().strip()

    if not target_word:
        raise ValueError("Target word cannot be empty.")

    # Find exact occurrences of the target word.
    matches = list(
        re.finditer(
            rf"\b{re.escape(target_word)}\b",
            sentence,
            flags=re.IGNORECASE
        )
    )

    if not matches:
        raise ValueError(
            f"Target word '{target_word}' was not found in the sentence."
        )

    if occurrence < 0 or occurrence >= len(matches):
        raise ValueError(
            f"Occurrence {occurrence} does not exist. "
            f"Found {len(matches)} occurrence(s)."
        )

    # Locate the selected word in the original sentence.
    start, end = matches[occurrence].span()

    # Tokenize while preserving character offsets.
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512
    )

    offset_mapping = inputs.pop("offset_mapping")[0]

    # Check whether the target word was truncated.
    target_positions = []

    for index, (token_start, token_end) in enumerate(
        offset_mapping.tolist()
    ):

        # Ignore special tokens.
        if token_start == token_end:
            continue

        # Find tokens overlapping the target word.
        if token_start < end and token_end > start:
            target_positions.append(index)

    if not target_positions:
        raise ValueError(
            "The target word could not be located in BERT's tokens. "
            "It may have been truncated or tokenized unexpectedly."
        )

    # Generate contextual representations.
    with torch.no_grad():
        outputs = model(**inputs)

    # Last hidden layer:
    # Shape = (sequence length, 768)
    hidden_states = outputs.last_hidden_state[0]

    # Extract all subword vectors belonging to the target word.
    target_vectors = hidden_states[target_positions]

    # Average subword vectors to create one word representation.
    word_embedding = target_vectors.mean(dim=0)

    return word_embedding.cpu().numpy()


# --------------------------------------------------
# 3. COMPARE TWO CONTEXTS
# --------------------------------------------------

def compare_contexts(sentence1, sentence2, target_word):
    """
    Extracts the same target word from two sentences
    and compares their contextual embeddings.
    """

    embedding1 = get_contextual_embedding(
        sentence1,
        target_word
    )

    embedding2 = get_contextual_embedding(
        sentence2,
        target_word
    )

    similarity = cosine_similarity(
        embedding1.reshape(1, -1),
        embedding2.reshape(1, -1)
    )[0][0]

    distance = np.linalg.norm(
        embedding1 - embedding2
    )

    return {
        "sentence_1": sentence1,
        "sentence_2": sentence2,
        "target_word": target_word,
        "embedding_1": embedding1,
        "embedding_2": embedding2,
        "cosine_similarity": float(similarity),
        "vector_distance": float(distance)
    }


# --------------------------------------------------
# 4. CALCULATE CONTEXTUAL SIMILARITY
# --------------------------------------------------

def contextual_similarity(sentence1, sentence2, target_word):
    """
    Returns only the cosine similarity between
    the contextual representations of a word.
    """

    result = compare_contexts(
        sentence1,
        sentence2,
        target_word
    )

    return result["cosine_similarity"]


# --------------------------------------------------
# 5. ANALYZE MULTIPLE CONTEXTS
# --------------------------------------------------

def analyze_contexts(sentence1, sentence2, target_word):
    """
    Analyzes a word appearing in two different contexts.

    Returns both embeddings, cosine similarity,
    and Euclidean distance.
    """

    return compare_contexts(
        sentence1,
        sentence2,
        target_word
    )


# --------------------------------------------------
# 6. QUICK TEST
# --------------------------------------------------

if __name__ == "__main__":

    sentence1 = "I deposited my money in the bank."

    sentence2 = (
        "The river overflowed its bank after heavy rain."
    )

    target_word = "bank"

    print("\nAnalyzing contextual embeddings...\n")

    result = analyze_contexts(
        sentence1,
        sentence2,
        target_word
    )

    print("Target word:", target_word)

    print(
        "Cosine similarity:",
        result["cosine_similarity"]
    )

    print(
        "Vector distance:",
        result["vector_distance"]
    )

    print(
        "Embedding dimensions:",
        len(result["embedding_1"])
    )

    print("\nContext 1:", sentence1)
    print("Context 2:", sentence2)
