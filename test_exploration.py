
from exploration import explore_word, compare_words


result = explore_word("king", top_n=5)

print("\n========== LEXIS WORD EXPLORER ==========")

print("Word:", result["word"])

print(
    "Embedding dimensions:",
    result["embedding_dimensions"]
)

print("\nSimilar words:")

for word, score in result["similar_words"]:
    print(word, round(score, 4))

print("\nAssigned cluster:")
print(result["cluster"])

print("\nWord comparison:")

comparison = compare_words("king", "queen")

print(comparison)

print("\n=========================================")
