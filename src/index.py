import faiss,numpy as np ,pickle,json

def load_json(filename):

    """load the json file s"""

    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
    

def build_faiss(embs,metadatas, dim= 1536):

    """Takes the embedings generated and meta data of embeddings as arguments
    Inner product is calculated for the diminsion i.e length of embedding
    then embeddings are converted into numpy array and converted into L2 normalisation form
    then for each embeddings index is created
    index are stored in index.faiss file , metadata for each index is stored in pickle file"""

    index = faiss.IndexFlatIP(dim)
    x = np.array(embs).astype('float32')
    faiss.normalize_L2(x)
    index.add(x)
    with open("models/meta.pkl","wb") as f: pickle.dump(metadatas,f)
    faiss.write_index(index, "models/index.faiss")



def build_files():

    """load the json file that is created which has embeddings
    Meta data is extracted from the embeddings_list in json file 
    index,pickle files are created using build_faiss function"""

    embeds_loaded = load_json("/Users/radhakrishnabalusu/Downloads/rag-project/embed_response.json")
    all_embeds=[]
    metadata=[]

    for item in embeds_loaded:
        all_embeds.append(item["embedding"])
        metadata.append({
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "chunk_text": item["text"]
        })
    
    build_faiss(all_embeds,metadatas=metadata)