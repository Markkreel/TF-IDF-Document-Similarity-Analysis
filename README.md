# TF-IDF Document Similarity Analysis

This project implements document similarity analysis using TF-IDF (Term Frequency-Inverse Document Frequency) with both manual implementation and scikit-learn approach. It processes text documents to calculate similarity scores and generates detailed term frequency and TF-IDF matrices.

## Project Overview

The project includes:

- Manual implementation of TF-IDF calculation
- Scikit-learn based TF-IDF vectorization
- Document similarity comparison using cosine similarity
- Output generation in both JSON and CSV formats

## Features

- Text preprocessing and tokenization
- Term frequency calculation
- Inverse document frequency computation
- TF-IDF matrix generation
- Document similarity scoring
- Multiple output formats (JSON, CSV)

## Implementation Details

### Manual Implementation

- Custom implementation of TF-IDF calculation
- Step-by-step term frequency computation
- Inverse document frequency weighting
- Similarity score calculation

### Scikit-learn Implementation

- Utilizes TfidfVectorizer for efficient processing
- Automated text preprocessing
- Vectorized TF-IDF computation
- Streamlined document comparison

## Output Files

The analysis generates several output files:

- `term_frequencies.csv`: Contains term frequency matrices
- `tfidf_abstract1.json`: TF-IDF scores for first document
- `tfidf_abstract2.json`: TF-IDF scores for second document
- `tfidf_abstract3.json`: TF-IDF scores for third document

## Dependencies

- pandas
- scikit-learn
- numpy

## Usage

1. Place your text documents in the `text_files` directory
2. Run either implementation:
   - Jupyter notebook: `Document Similarity.ipynb`
   - Python script: `sklearn_implementation.py`
3. Check the `output` directory for results
