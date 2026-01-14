import chromadb
import ollama
from pathlib import Path
from textwrap import dedent
from src.config import CONFIG


chroma_dir = Path.cwd() / "chroma"
chroma_dir.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(chroma_dir / "chroma_store"))


selected_sources = [CONFIG.ARTICLE]


def embed_query(query: str, model: str = "mxbai-embed-large"):

    try:
        resp = ollama.embeddings(model=model, prompt=query)
        return resp["embedding"]
    except Exception as e:
        print(f"Embedding failed: {e}")
        return None


def retrieve_similar_chunks(query_embedding, collection, top_k: int = 10):

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
        where={"source": {"$in": selected_sources}},
    )
    return results["documents"][0], results["metadatas"][0]


def build_prompt(query: str, retrieved_chunks, metadatas, articles):

    article_chunks = []
    for chunk, meta in zip(retrieved_chunks, metadatas):
        if meta.get("source") in articles:
            article_chunks.append(chunk.strip())

    article_text = "\n\n".join(article_chunks) if article_chunks else "No relevant article content found."

    prompt = f"""
<EXCERPT>
{article_text}
</EXCERPT>

QUESTION:
{query}
"""
    return dedent(prompt).strip()


def answer_query(user_prompt: str, system_prompt: str, model: str):

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )
    return response["message"]["content"]


def build_system_prompt() -> str:

    prompt = dedent("""
prompt = 
    You are Patient care-RAG, a clinical Q&A assistant for patients.
    Your ONLY source of truth is the text between <EXCERPT>...</EXCERPT>.
    Do not use prior knowledge, outside sources, guesses, or assumptions.
    Do not cite or invent information that is not in the excerpt.

    If the excerpt does not contain the answer, reply exactly:
    "I don't know based on the given information, I will need some more information to give an accurate answer"

    If the excerpt does contain the answer, respond in this format:
    Answer: <concise, evidence-based reply, max 200 words>

    Write clearly for a general audience. You may reason internally but never reveal your reasoning.
    """) .strip()
    return prompt


if __name__ == '__main__':

    collection = client.get_collection(name=CONFIG.COLLECTION)


    _model = "phi3:latest"


    QUESTIONS = [
        "Why does my jaw click and hurt when I chew?",
        "What’s the most likely reason I’m having this jaw pain?",
        "Does putting ice or heat on the side of my jaw help? Which should I try first?",
        "What pain reliever can I buy without a prescription for quick relief right now?",
        "Should I try not to yawn wide or open my mouth big?",
        "What signs should I watch for that mean my jaw problem is getting worse?",
        "What can I do at home today to calm the pain (soft foods, gentle jaw moves, rest)?",
        "Could stress or clenching my teeth be causing this?",
        "Do night guards or mouth splints really help with jaw pain?",
        "Do people usually need surgery for this, or does it get better with simple treatments?"
    ]

    _system_prompt = build_system_prompt()

    for q in QUESTIONS:
        print("\n" + "=" * 80)
        print("QUESTION:", q)

        # Embed, retrieve, prompt, answer
        embedded_query = embed_query(q.strip())
        if not embedded_query:
            print("Embedding failed, skipping.")
            continue

        top_chunks, meta_data = retrieve_similar_chunks(
            query_embedding=embedded_query,
            collection=collection,
            top_k=10
        )

        prompt_text = build_prompt(
            query=q,
            retrieved_chunks=top_chunks,
            metadatas=meta_data,
            articles=selected_sources
        )

        answer = answer_query(
            user_prompt=prompt_text,
            system_prompt=_system_prompt,
            model=_model
        )

        print("ANSWER:\n", answer)
