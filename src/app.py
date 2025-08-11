import os
import generate, embed, index

if __name__ == "__main__":
    # Run embedding + indexing only if files don't exist
    if not (os.path.exists("models/index.faiss") and os.path.exists("models/meta.pkl")):
        print("🔄 Building embeddings and index for the first time...")
        embed.store_embed()
        index.build_files()
        print("✅ Embeddings and index built.")
    else:
        print("✅ Found existing index. Skipping embedding & indexing.")

    # Query
    query = input("Enter the query: ")
    response = generate.generate_output(query)
    print(response)