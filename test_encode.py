import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3')
with open('data/enriched_chunks_cache.json', encoding='utf-8') as f:
    chunks = json.load(f)

print(len(chunks))
for i, (orig, enriched) in enumerate(chunks.items()):
    text = enriched.get("context", orig)
    print(f"Chunk {i} len: {len(text)}", flush=True)
    model.encode(text)
    print(f"Chunk {i} done.", flush=True)

print("All done.")
