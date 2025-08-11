# generate.py
SYSTEM = """You are a helpful assistant. Answer ONLY from the context.
If unsure, say you don't know. Cite sources as [name of source]."""
USER_TMPL = """Question: {q}

Context:
{ctx}

Answer with citations."""



from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

def  answer(q,ctx_chunks):

    """define chat completions client and get ctx from the meta data that is returned ins search
    send role and messages and return the response"""

    client= AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version = "2023-05-15"
        )
    
    deployment = os.getenv('AZURE_CHAT_API_DEPLOYMENT')

    ctx = "\n\n".join([
        f"- ({c[0]['source']}) {c[0]['chunk_text'][:800]}"
        for c in ctx_chunks
    ])

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":USER_TMPL.format(q=q, ctx=ctx)}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content


def generate_output(query):
    """get the metadata of similar vectors from search and send them to generate answer
    """

    from embed import get_query_embedding
    from retrieve import search
    query = query
    query_emb = get_query_embedding(query)
    ctx_chunks = search(query_emb)

    final_output= answer(query,ctx_chunks=ctx_chunks)
    return final_output