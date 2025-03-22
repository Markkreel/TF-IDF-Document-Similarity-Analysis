from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Read the content of abstract_1.txt
with open("text_files/abstract_1.txt", "r", encoding="UTF-8") as file:
    abstract_1_text = file.read()

# Create and configure the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Transform the text into TF-IDF features
response = vectorizer.fit_transform([abstract_1_text])
print(response)
print(response.todense())

# Create a DataFrame with the TF-IDF scores
df = pd.DataFrame(
    response.todense().T,
    index=vectorizer.get_feature_names_out(),
    columns=["D1"],
)
print(df)

vector1 = TfidfVectorizer(stop_words="english")
vector2 = TfidfVectorizer(stop_words="english")
vector3 = TfidfVectorizer(stop_words="english")
vector4 = TfidfVectorizer(stop_words="english")
