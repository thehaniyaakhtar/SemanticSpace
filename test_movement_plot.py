
from context_visualization import build_contextual_dataset
from context_movement_plot import create_movement_plot


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

figure = create_movement_plot(records, "bank")

figure.show()
