#-------------------------------------------------------------
# AUTHOR: Antonio Duran
# FILENAME: SPIMI_index.py
# SPECIFICATION: This code implements a simplified SPIMI-based
# inverted index construction pipeline that follows strict 
# specifications 
# FOR: CS 5180- Assignment #2
# TIME SPENT: 3 hours 
#-----------------------------------------------------------*/

# importing required libraries
import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_PATH = "corpus/corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10

READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500


# ---------------------------------------------------------
# 1) READ FIRST BLOCK OF 100 DOCUMENTS USING PANDAS
# ---------------------------------------------------------
# Use pandas.read_csv with chunksize=100.
# Each chunk corresponds to one memory block.
# Convert docIDs like "D0001" to integers.
# ---------------------------------------------------------
# --> add your Python code here
chunks = pd.read_csv(
    INPUT_PATH,
    sep='\t',
    chunksize=BLOCK_SIZE,
    header=None,
    names=['DocID', 'Text']
)

first_block = next(chunks)
first_block['DocID'] = first_block['DocID'].str.replace(r'\D', '', regex=True).astype(int)

# ---------------------------------------------------------
# 2) BUILD PARTIAL INDEX (SPIMI STYLE) FOR CURRENT BLOCK
# ---------------------------------------------------------
# - Use CountVectorizer(stop_words='english')
# - Fit and transform the 100 documents
# - Reconstruct binary postings lists from the sparse matrix
# - Store postings in a dictionary: term -> set(docIDs)
# ---------------------------------------------------------
# --> add your Python code here
vectorizer = CountVectorizer(stop_words='english')
sparse_matrix = vectorizer.fit_transform(first_block['Text'])
terms = vectorizer.get_feature_names_out()

dict = {}
coord = sparse_matrix.tocoo()
doc_arrays = first_block['DocID'].values

for doc_idx, term_idx in zip(coord.row, coord.col):
    term = terms[term_idx]
    doc_id = doc_arrays[doc_idx]
    if term not in dict:
        dict[term] = set()
    dict[term].add(doc_id)

# ---------------------------------------------------------
# 3) FLUSH PARTIAL INDEX TO DISK
# ---------------------------------------------------------
# - Sort terms lexicographically
# - Sort postings lists (ascending docID)
# - Write to: block_1.txt, block_2.txt, ..., block_10.txt
# - Format: term:docID1,docID2,docID3
# ---------------------------------------------------------
# --> add your Python code here
sorted_terms = sorted(dict.keys())
with open("block_1.txt", "w", encoding='utf-8') as f:
    for term in sorted_terms:
        sorted_docs = sorted(list(dict[term]))
        doc_str = ",".join(map(str, sorted_docs))
        f.write(f"{term}:{doc_str}\n")

# ---------------------------------------------------------
# 4) REPEAT STEPS 1–3 FOR ALL 10 BLOCKS
# ---------------------------------------------------------
# - Continue reading next 100-doc chunks
# - After processing each block, flush to disk
# - Do NOT keep previous blocks in memory
# ---------------------------------------------------------
# --> add your Python code here
block_idx = 2

for chunk in chunks:
    if block_idx > NUM_BLOCKS:
        break
        
    chunk['DocID'] = chunk['DocID'].str.replace(r'\D', '', regex=True).astype(int)
    
    vec = CountVectorizer(stop_words='english')
    sparse_matrix_chunk = vec.fit_transform(chunk['Text'])
    chunk_terms = vec.get_feature_names_out()
    
    chunk_dict = {}
    coord_chunk = sparse_matrix_chunk.tocoo()
    chunk_doc_arrays = chunk['DocID'].values
    
    for doc_idx, term_idx in zip(coord_chunk.row, coord_chunk.col):
        term = chunk_terms[term_idx]
        doc_id = chunk_doc_arrays[doc_idx]
        if term not in chunk_dict:
            chunk_dict[term] = set()
        chunk_dict[term].add(doc_id)
        
    chunk_sorted_terms = sorted(chunk_dict.keys())
    with open(f"block_{block_idx}.txt", "w", encoding="utf-8") as f:
        for term in chunk_sorted_terms:
            sorted_docs = sorted(list(chunk_dict[term]))
            docs_str = ",".join(map(str, sorted_docs))
            f.write(f"{term}:{docs_str}\n")
            
    block_idx += 1

# ---------------------------------------------------------
# 5) FINAL MERGE PHASE
# ---------------------------------------------------------
# After all block files are created:
# - Open block_1.txt ... block_10.txt simultaneously
# ---------------------------------------------------------
# --> add your Python code here
block_files = []

for i in range(1, NUM_BLOCKS + 1):
    block_files.append(open(f"block_{i}.txt", "r", encoding="utf-8"))

# ---------------------------------------------------------
# 6) INITIALIZE READ BUFFERS
# ---------------------------------------------------------
# For each block file:
# - Read up to READ_BUFFER_LINES_PER_FILE lines
# - Parse each line into (term, postings_list)
# - Store in a per-file buffer
# ---------------------------------------------------------
# --> add your Python code here
read_buff = {}
buff_ptrs = {}

for i, f in enumerate(block_files):
    lines = []

    for _ in range(READ_BUFFER_LINES_PER_FILE):
        line = f.readline()
        if not line:
            break
        lines.append(line.strip())

    lines_parsed = []
    for line in lines:
        term, posting_str = line.split(":", 1)
        postings = [int(post) for post in posting_str.split(",")]
        lines_parsed.append((term, postings))

    read_buff[i] = lines_parsed
    buff_ptrs[i] = 0

# ---------------------------------------------------------
# 7) INITIALIZE MIN-HEAP (OR SORTED STRUCTURE)
# ---------------------------------------------------------
# - Push the first term from each buffer into a min-heap
# - Heap elements: (term, file_index)
# ---------------------------------------------------------
# --> add your Python code here
min_heap = []

for i in range(len(block_files)):
    if read_buff[i]:
        first_term = read_buff[i][0][0]
        heapq.heappush(min_heap, (first_term, i))

# ---------------------------------------------------------
# 8) MERGE LOOP
# ---------------------------------------------------------
# While heap is not empty:
#   1. Pop smallest term
#   2. Collect all buffers whose current term matches
#   3. Merge postings lists (sorted + deduplicated)
#   4. Advance corresponding buffer pointers
#   5. If a buffer is exhausted, read next 100 lines (if available)
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 9) WRITE BUFFER MANAGEMENT
# ---------------------------------------------------------
# - Append merged term-line to write buffer
# - If write buffer reaches WRITE_BUFFER_LINES:
#       flush (append) to final_index.txt
# - After merge loop ends:
#       flush remaining write buffer
# ---------------------------------------------------------
# --> add your Python code here


# ---------------------------------------------------------
# 10) CLEANUP
# ---------------------------------------------------------
# - Close all open block files
# - Ensure final_index.txt is properly written
# ---------------------------------------------------------
# --> add your Python code here
for f in block_files:
    f.close()
    
# final_index.close()
