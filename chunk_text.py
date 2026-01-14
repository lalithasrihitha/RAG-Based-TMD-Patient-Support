from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from src.config import CONFIG

"""
https://python.langchain.com/docs/concepts/text_splitters/
"""


def load_text_from_file(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
        return data


def token_chunker(text):
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300,
        chunk_overlap=30,
        encoding_name="cl100k_base"  # used by GPT-4, GPT-4o, GPT-3.5-turbo
    )
    chunks = splitter.split_text(text)
    return chunks


def recursive_token_chunker(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20
    )
    chunks = splitter.split_text(text)
    return chunks


if __name__ == '__main__':
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{CONFIG.ARTICLE}.md"
    input_file_path = data_dir / file_name
    text_from_file = load_text_from_file(file_path=input_file_path)
    _chunks = token_chunker(text=text_from_file)
    print(_chunks)
    for item in _chunks:
        print(item)
        print('-' * 50)
