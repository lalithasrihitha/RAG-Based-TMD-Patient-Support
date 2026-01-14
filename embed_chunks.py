import ollama
from src.chunk_text import load_text_from_file, token_chunker
from pathlib import Path
from src.config import CONFIG

"""
https://www.arsturn.com/blog/understanding-ollamas-embedding-models
https://ollama.com/library/mxbai-embed-large
"""

def embed_texts(chunks, model="mxbai-embed-large"):
    embeddings = []
    for chunk in chunks:
        resp = ollama.embeddings(model=model, prompt=chunk)
        embeddings.append(resp["embedding"])
    return embeddings


if __name__ == '__main__':
    data_dir = Path.cwd() / "data"
    file_name = f"{CONFIG.ARTICLE}.md"
    input_file_path = data_dir / file_name
    text_from_file = load_text_from_file(file_path=input_file_path)
    chunked_texts = token_chunker(text_from_file)
    embedded_texts = embed_texts(chunked_texts)
    print(embedded_texts)
