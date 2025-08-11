
def chunk_text(text,chunk_size = 1200, overlap = 200):
    """Takes the text as input from the docs list in ingest.py ,predefine chunk_size and overlap
    Then we divide the text into chunks and store them in chunks list"""
    chunks=[]
    for i in range(0,len(text),chunk_size-overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks


