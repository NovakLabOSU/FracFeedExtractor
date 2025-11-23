"""
PDF FracFeedExtractor Text Chunking and Indexing
-------------------------

Chunk and index text from documents that have bee classified as 'useful'.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.llms.base import LLM
import subprocess
import shutil
import os
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings


# Template for querying LLM about specific data
template = """
You are a precise information extraction system.

Using ONLY the provided context, answer the following questions:

- Does the statistic "{input}" appear in the document (it may not be explicitly stated)?
- If yes, what is its value?
- Provide the exact text snippet.

Output your answer in JSON format as follows (Format ends after the closing }}):

{{
  "exists": true/false,
  "value": "<value or null>",
  "snippet": "<snippet or null>"
}}

Context:
{context}
"""

# Create prompt template
prompt = PromptTemplate(
    template=template,
    input_variables=["input", "context"]
)


class OllamaLLM(LLM):
    """Wrapper for Ollama CLI"""

    model_name: str = "phi3:3.8b"

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        # Allow overriding the ollama executable via env var, or locate it via PATH
        ollama_path = os.environ.get("OLLAMA_PATH") or shutil.which("ollama")
        if not ollama_path:
            raise FileNotFoundError(
                "The 'ollama' CLI was not found.\n"
                "Install Ollama (https://ollama.com) and ensure 'ollama' is on your PATH,\n"
                "or set the environment variable OLLAMA_PATH to the executable path."
            )

        try:
            result = subprocess.run(
                [ollama_path, "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Failed to execute '{ollama_path}'. Ensure the path is correct and executable."
            )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                f"Ollama exited with code {result.returncode}. Stderr: {stderr}"
            )

        return result.stdout

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}
    
    @property
    def _llm_type(self):
        return "ollama"
    

# Chunk text document
def chunk_document(text_path: str) -> list[Document]:
    text_path = Path(text_path)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Wrap in Document object so metadata is preserved
    return [Document(page_content=chunk, metadata={"source": text_path.name})
            for chunk in splitter.split_text(text)]


# embed and build vector store from chunks
def build_vector_store(chunks: list[Document]):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore


# Extract data from text using LLM, and retriever
def extract_statistic(statistic: str, retriever, llm):
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    qa = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return qa.invoke({"input": statistic})


def main():
    text_path = "data/processed-text/Abe_1989.txt"
    chunks = chunk_document(text_path)
    vector_store = build_vector_store(chunks)
    retriever = vector_store.as_retriever(k=4)
    llm = OllamaLLM(model_name="phi3:3.8b")
    statistic_to_find = "fraction of predators that are feeding"

    result = extract_statistic(statistic_to_find, retriever, llm)
    print("Extraction Result:", result)

if __name__ == "__main__":
    main()
