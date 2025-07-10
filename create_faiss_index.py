from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def create_faiss_index():
    try:
        print("Loading processed CUAD data...")
        with open("data/processed_cuad.json", 'r') as f:
            data = json.load(f)
        
        print("Initializing sentence transformer...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print("Extracting clause texts for embedding...")
        clause_texts = []
        clause_metadata = []
        
        for i, item in enumerate(data):
            for j, label in enumerate(item['labels']):
                if not label['is_impossible'] and label['answers']:
                    # Extract the answer text for embedding
                    answer_text = label['answers'][0]['text']
                    clause_texts.append(answer_text)
                    clause_metadata.append({
                        'item_index': i,
                        'label_index': j,
                        'question': label['question'],
                        'answer': answer_text
                    })
        
        print(f"Found {len(clause_texts)} clause texts to embed")
        
        if len(clause_texts) == 0:
            print("No valid clause texts found. Check the data structure.")
            return
        
        print("Generating embeddings...")
        embeddings = embedder.encode(clause_texts, show_progress_bar=True)
        
        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        print("Saving FAISS index and metadata...")
        faiss.write_index(index, "data/faiss_index")
        
        # Save metadata for retrieval
        with open("data/clause_metadata.json", 'w') as f:
            json.dump(clause_metadata, f, indent=2)
        
        print(f"FAISS index saved to data/faiss_index")
        print(f"Metadata saved to data/clause_metadata.json")
        print(f"Index contains {index.ntotal} vectors of dimension {dimension}")
        
    except Exception as e:
        print(f"Error creating FAISS index: {str(e)}")

if __name__ == "__main__":
    create_faiss_index()
