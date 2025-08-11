from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True)

def create_embeddings(chunk_list):

    """define client using correct paramenter values in .env file
    for each item in chunk_list, each chunk in item the embedding is created by using text-embedding-003-small model
    embedding,chink_id(generated here),source,text of chunk are stored in all_embeddings list and returned"""


    client= AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version = "2023-05-15"
    )

    deployment =  os.getenv('AZURE_API_DEPLOYMENT')
    all_embeddings = []  # store all embeddings
    chunk_id_count =1
    for item in chunk_list:  # each PDF
        for chunk in item["chunks"]:  # each chunk in the PDF
            embedding = client.embeddings.create(
                model=deployment,
                input=chunk
            )
            all_embeddings.append({
                "source": item["source"],
                "chunk_id": chunk_id_count,
                "text": chunk,
                "embedding": embedding.data[0].embedding
            })
            chunk_id_count+=1
    return all_embeddings

def get_query_embedding(text):
    
    """created the embeddings for the query given by user"""

    client= AzureOpenAI(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version = "2023-05-15"
    )

    deployment =  os.getenv('AZURE_API_DEPLOYMENT')

    response=client.embeddings.create(input=text,model=deployment)
    return response.data[0].embedding


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def store_embed():
    
    """in this files we run ingest file by sending pdf paths 
    source file of chunks genrated and chunks are stored in chunk_list
    pass the chunk_list into create_embeddings and store the response in json file
    """
    from ingest import load_pdf
    from chunk_1 import chunk_text
    import glob
    pdf_files = glob.glob("data/*.pdf")
    chunk_list = []  # store all PDFs' chunks

    for file_path in pdf_files:
        docs = load_pdf(path=file_path)  # [{'source': ..., 'text': ...}]
        chunks = chunk_text(docs[0]["text"])  # list of strings

        
        chunk_list.append({
            "source": docs[0]["source"],
            "chunks": chunks
        })

    response = create_embeddings(chunk_list)
    
    save_json(response, "embed_response.json")



