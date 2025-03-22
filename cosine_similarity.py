from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json

# Read all three abstracts
with open("text_files/abstract_1.txt", "r", encoding="UTF-8") as file:
    abstract_1_text = file.read()
with open("text_files/abstract_2.txt", "r", encoding="UTF-8") as file:
    abstract_2_text = file.read()
with open("text_files/abstract_3.txt", "r", encoding="UTF-8") as file:
    abstract_3_text = file.read()

# Create and configure the TF-IDF vectorizer for all documents together
vectorizer = TfidfVectorizer(stop_words="english")

# Transform all documents into TF-IDF features
documents = [abstract_1_text, abstract_2_text, abstract_3_text]
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity between documents
similarity_matrix = cosine_similarity(tfidf_matrix)

# Create individual DataFrames for each document's TF-IDF scores
for i, doc_name in enumerate(["D1", "D2", "D3"]):
    df = pd.DataFrame(
        tfidf_matrix[i].todense().T,
        index=vectorizer.get_feature_names_out(),
        columns=[doc_name],
    )
    df.to_json(f"output/tfidf_abstract{i+1}.json")

# Create similarity scores dictionary
similarity_scores = {
    "document_pairs": [
        {"pair": "Abstract 1 - Abstract 2", "similarity": similarity_matrix[0][1]},
        {"pair": "Abstract 1 - Abstract 3", "similarity": similarity_matrix[0][2]},
        {"pair": "Abstract 2 - Abstract 3", "similarity": similarity_matrix[1][2]},
    ]
}

# Save similarity scores to JSON
with open("output/document_similarities.json", "w") as f:
    json.dump(similarity_scores, f, indent=4)

# Print similarity matrix for verification
print("\nDocument Similarity Matrix:")
print(similarity_matrix)
