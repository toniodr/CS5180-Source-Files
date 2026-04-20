#-------------------------------------------------------------------------
# AUTHOR: Antonio Duran
# FILENAME: bm25_search.py
# SPECIFICATION: This program builds a BM25-based search engine 
#                   and evaluates its performance 
# FOR: CS 5180- Assignment #4
# TIME SPENT: ~1.5 hours
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd
import string

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------
# Helper function: tokenize text and remove stopwords only
# ---------------------------------------------------------
def preprocess(text):
    # --> add your Python code here
    # Suggested steps:
    # 1. convert text to lowercase
    # 2. split into tokens
    # 3. remove stopwords only
    # 4. return the filtered tokens
    text = text.lower()

    for p in string.punctuation:
        text = text.replace(p, ' ')
    
    tokens = text.split()

    return [word for word in tokens if word not in ENGLISH_STOP_WORDS]


# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# Files:
#   docs.csv
#   queries.csv
#   relevance_judgments.csv
# --> add your Python code here
docs_df = pd.read_csv('docs.csv')
queries_df = pd.read_csv('queries.csv')
relevance_df = pd.read_csv('relevance_judgments.csv')


# ---------------------------------------------------------
# 2. Build the BM25 index for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# Steps:
#   1. preprocess each document
#   2. store tokenized documents in a list
#   3. create the BM25 model
# --> add your Python code here
doc_texts = docs_df['text'].tolist()
doc_ids = docs_df['doc_id'].tolist()

tokenized_docs = [preprocess(doc) for doc in doc_texts]

bm25 = BM25Okapi(tokenized_docs)


# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# Suggested structure:
#   - for each query:
#       1. preprocess the query
#       2. compute BM25 scores for all documents
#       3. rank documents by score in descending order
#       4. retrieve the relevant documents for that query
#       5. compute AP
# --> add your Python code here
ap_scores = []

for idx, row in queries_df.iterrows():
    q_id = row['query_id']
    q_text = row['query_text']

    tokenized_q = preprocess(q_text)

    doc_scores = bm25.get_scores(tokenized_q)

    doc_score_pairs = list(zip(doc_ids, doc_scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    ranked_doc_ids = [pair[0] for pair in doc_score_pairs]

    rel_docs_df = relevance_df[(relevance_df['query_id'] == q_id) 
                              & (relevance_df['judgment'] == 'R')]
    rel_docs = set(rel_docs_df['doc_id'].tolist())

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------
    # Suggested steps:
    #   - initialize variables
    #   - go through the ranked documents
    #   - whenever a relevant document is found:
    #         precision = (# relevant found so far) / (current rank position)
    #         add precision to the running sum
    #   - AP = sum of precisions / total number of relevant documents
    #   - if there are no relevant documents, AP = 0

    # store the AP value for this query (use any data structure you prefer)
    count = 0 
    sum_precisions = 0.0

    for rank, d_id in enumerate(ranked_doc_ids, start=1):
        if d_id in rel_docs:
            count += 1
            precision = count / rank
            sum_precisions += precision

    if len(rel_docs) > 0:
        ap = sum_precisions / len(rel_docs)
    else:
        ap = 0.0

    ap_scores.append((q_id, ap))

# ---------------------------------------------------------
# 5. Sort queries by AP in descending order
# ---------------------------------------------------------
# --> add your Python code here
ap_scores.sort(key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------
# 6. Print the sorted queries and their AP scores
# ---------------------------------------------------------
print("====================================================")
print("Queries sorted by Average Precision (AP):")
# --> add your Python code here
for q_id, ap in ap_scores:
    print(f"{q_id}: {ap:.6f}")
