import faiss,numpy as np , pickle

def load_index():

    """ loads the data in index,metadata files into variables and return them
    """
    index = faiss.read_index("/Users/radhakrishnabalusu/Downloads/rag-project/models/index.faiss")
    metas = pickle.load(open("models/meta.pkl","rb"))

    return index,metas

def search(query_emb,k=5):

    """takes query_embedding that is generated and k value i.e number of close vectors
    caluculates the normalised array for query numpy embedding array
    D is similarities of k vectors in vector store,I is indices of those vectors
    then checks the meta data of those vecors and adds similarity score to those metadata
    """
    index,metas = load_index()
    q= np.array([query_emb]).astype('float32')
    faiss.normalize_L2(q)

    D,I = index.search(q,k)

    return [(metas[i], float(D[0][j])) for j,i in enumerate(I[0])]


