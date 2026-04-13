#-------------------------------------------------------------------------
# AUTHOR: Antonio Duran 
# FILENAME: vsm_search.py
# SPECIFICATION: This program builds a Vector Space Model search engine
#                   and evaluates its performance
# FOR: CS 5180- Assignment #3
# TIME SPENT: 
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Load the input files
# ---------------------------------------------------------
# --> add your Python code here
docs_df = pd.read_csv('docs.csv')
queries_df = pd.read_csv('queries.csv')
rlvnce_df = pd.read_csv('relevance_judgments.csv')

# ---------------------------------------------------------
# 2. Build the TF-IDF matrix for the documents
# ---------------------------------------------------------
# Requirement: remove stopwords only
# --> add your Python code here
vectorizer =TfidfVectorizer(stop_words='english')
doc_tfidf = vectorizer.fit_transform(docs_df['text'])

doc_ids = docs_df['doc_id'].tolist()

# ---------------------------------------------------------
# 3. Process each query and compute AP values
# ---------------------------------------------------------
# --> add your Python code here
ap_scores = []

for index, row in queries_df.iterrows():
    q_id = row['query_id']
    q_text = row['query_text']

    q_tfidf = vectorizer.transform([q_text])

    cos_scores = cosine_similarity(q_tfidf, doc_tfidf).flatten()

    ranked_indices = cos_scores.argsort()[::-1]

    q_judgements = rlvnce_df[rlvnce_df['query_id'] == q_id]

    judgment_dict = dict(zip(q_judgements['doc_id'], q_judgements['judgment']))

    # -----------------------------------------------------
    # 4. Compute Average Precision (AP)
    # -----------------------------------------------------

    # store the AP value for this query (use any data structure you prefer)
    total_rel = sum (1 for j in judgment_dict.values() if j == 'R')

    if total_rel == 0:
        ap_scores.append((q_id, 0.0))
        continue

    rel_found = 0
    sum_precision = 0

    for k, idx in enumerate(ranked_indices, start=1):
        doc_id = doc_ids[idx]

        is_rel = judgment_dict.get(doc_id, 'N') == 'R'

        if is_rel:
            rel_found += 1
            precision_at_k = rel_found / k 
            sum_precision += precision_at_k

    ap = sum_precision / total_rel
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
    print(f"{q_id}: {ap:.3f}")