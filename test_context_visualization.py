
from contextual_visualization import build_contextual_dataset


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

print("\n========== LEXIS CONTEXTUAL DATASET ==========")

print("Number of sentences:", len(records))

print("Embedding matrix shape:", embeddings.shape)

print("\nContextual coordinates:\n")

for record in records:

    print("Sentence:", record["sentence"])

    print(
        "Coordinates:",
        record["x"],
        record["y"],
        record["z"]
    )

    print("----------------------------------------")

print(
    "\nExplained variance:",
    reducer.explained_variance_ratio_
)

print("\n==============================================")
