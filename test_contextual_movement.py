
from contextual_visualization import build_contextual_dataset

from contextual_movement import (
    analyze_contextual_movement,
    calculate_consecutive_movement
)


sentences = [
    "I deposited my salary in the bank.",
    "The bank approved my loan application.",
    "The river overflowed its bank after heavy rain.",
    "We sat on the grassy bank beside the river.",
    "The fishermen waited patiently along the bank."
]

records, embeddings, reducer = build_contextual_dataset(
    sentences,
    "bank"
)

print("\n========== CONTEXTUAL MOVEMENT ==========")

results = analyze_contextual_movement(
    records,
    embeddings
)

for result in results:

    print("\nContext:", result["context_number"])
    print("Sentence:", result["sentence"])

    print(
        "Similarity to first context:",
        round(result["cosine_similarity"], 4)
    )

    print(
        "Vector distance:",
        round(result["vector_distance"], 4)
    )


print("\n========== CONSECUTIVE MOVEMENT ==========")

movements = calculate_consecutive_movement(embeddings)

for movement in movements:
    print(movement)
