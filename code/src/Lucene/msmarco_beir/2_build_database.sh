INPUT_DIR=code/data/local_index_search/msmarco_beir/jsonl_docs
INDEX_DIR=code/data/local_index_search/msmarco_beir/pyserini_index

python -m pyserini.index.lucene -collection JsonCollection \
 -input $INPUT_DIR \
 -index $INDEX_DIR \
 -generator DefaultLuceneDocumentGenerator \
 -threads 16 \
 -storePositions -storeDocvectors -storeRaw
