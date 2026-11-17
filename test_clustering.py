
from clustering import cluster_words


clusters, model, vectors = cluster_words(
    n_clusters=5
)

print("\n========== LEXIS WORD CLUSTERS ==========")

for cluster_number, words in sorted(clusters.items()):

    print(f"\nCLUSTER {cluster_number + 1}")

    print(", ".join(words))

print("\nTotal clusters:", len(clusters))

print("Total words:", sum(len(words) for words in clusters.values()))

print("\n=========================================")
