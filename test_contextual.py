
from contextual_embeddings import analyze_contexts

sentence1 = "I deposited my money in the bank."

sentence2 = (
    "The river overflowed its bank after heavy rain."
)

target_word = "bank"

result = analyze_contexts(
    sentence1,
    sentence2,
    target_word
)

print("\n========== LEXIS CONTEXTUAL ANALYSIS ==========")

print("\nTarget word:", result["target_word"])

print("\nContext 1:", result["sentence_1"])
print("Context 2:", result["sentence_2"])

print(
    "\nCosine similarity:",
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

print("\nFirst embedding:")
print(result["embedding_1"][:10])

print("\nSecond embedding:")
print(result["embedding_2"][:10])

print("\n========== TEST COMPLETED ==========")
