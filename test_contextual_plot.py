
from contextual_visualization import build_contextual_dataset
from contextual_plot import create_contextual_plot


sentences = [
    "I deposited my salary in the bank.",
    "The bank approved my loan application.",
    "The river overflowed its bank after heavy rain.",
    "We sat on the grassy bank beside the river.",
    "The fishermen waited patiently along the bank."
]

target_word = "bank"


# Generate contextual representations
records, embeddings, reducer = build_contextual_dataset(
    sentences,
    target_word
)


# Generate the interactive 3D plot
figure = create_contextual_plot(
    records,
    target_word
)


# Open the visualisation in a browser
figure.show()
