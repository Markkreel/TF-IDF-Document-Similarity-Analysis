from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Read the content of abstract_1.txt
with open("text_files/abstract_1.txt", "r", encoding="UTF-8") as file:
    abstract_1_text = file.read()

# Create and configure the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Transform the text into TF-IDF features
response_1 = vectorizer.fit_transform([abstract_1_text])
print(response_1)
print(response_1.todense())

# Create a DataFrame with the TF-IDF scores
df_1 = pd.DataFrame(
    response_1.todense().T,
    index=vectorizer.get_feature_names_out(),
    columns=["D1"],
)
print(df_1)
# Save DataFrame 1 to JSON
df_1.to_json("output/tfidf_abstract1.json")

# Abstract 2
with open("text_files/abstract_2.txt", "r", encoding="UTF-8") as file:
    abstract_2_text = file.read()

vectorizer = TfidfVectorizer(stop_words="english")

response_2 = vectorizer.fit_transform([abstract_2_text])
print(response_2)
print(response_2.todense())

df_2 = pd.DataFrame(
    response_2.todense().T,
    index=vectorizer.get_feature_names_out(),
    columns=["D2"],
)
print(df_2)
# Save DataFrame 2 to JSON
df_2.to_json("output/tfidf_abstract2.json")

# Abstract 3
with open("text_files/abstract_3.txt", "r", encoding="UTF-8") as file:
    abstract_3_text = file.read()

vectorizer = TfidfVectorizer(stop_words="english")

response_3 = vectorizer.fit_transform([abstract_3_text])
print(response_3)
print(response_3.todense())

df_3 = pd.DataFrame(
    response_3.todense().T,
    index=vectorizer.get_feature_names_out(),
    columns=["D3"],
)
print(df_3)
# Save DataFrame 3 to JSON
df_3.to_json("output/tfidf_abstract3.json")
