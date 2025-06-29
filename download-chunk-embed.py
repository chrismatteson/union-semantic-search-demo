# ## Overview
#
# This is a workflow to download and chunk and embed 
# a the Paul Graham essays and expose it via a streamlit app.
#
# This particular workflow is based on the examples here:
# https://github.com/unionai/unionai-examples/blob/main/tutorials/agentic_rag/agentic_rag.py
# https://github.com/unionai/unionai-examples/blob/main/tutorials/vector_store_lance_db/vector_store_lance_db.py
# https://stackoverflow.com/questions/35371043/use-python-requests-to-download-csv
# https://notes.alexkehayias.com/turn-off-chroma-telemetry-in-langchain/
# 
# First, let's import the workflow dependencies:

import os
from dataclasses import dataclass, field
from typing import Annotated, Optional

import union
from flytekit.types.directory import FlyteDirectory
from flytekit import Resources
from flytekit.extras.accelerators import L4
from union.actor import ActorEnvironment


# maximum number of question rewrites
MAX_REWRITES = 10

# ## Creating Secrets for an OpenAI API key
#
# Go to the [OpenAI website](https://platform.openai.com/api-keys) to get an
# API key. Then, create a secret with the `union` CLI tool:
#
# ```shell
# $ union create secret openai_api_key
# ```
#
# then paste the client ID when prompted. We'll use the `openai_api_key` secret
# throughout this tutorial to authenticate with the OpenAI API and use GPT4 as
# the underlying LLM.

# ## Defining the container image
#
# Here we define the container image that the RAG workflow will run on, pinning
# dependencies to ensure reproducibility.

image = union.ImageSpec(
    builder="union",
    registry=os.environ.get("IMAGE_SPEC_REGISTRY"),
    apt_packages=["build-essential"],
    packages=[
        "opentelemetry-exporter-otlp-proto-grpc>=1.23.0",
        "chromadb",
        "pyarrow",
        "tqdm",
        "sentence-transformers",
        "requests",
    ],
)

# ## Creating an `ActorEnvironment`
#
# In order to run our RAG workflow quickly, we define an `ActorEnvironment` so
# that we can reuse the container to run the steps of our workflow. We can specify
# variables like `ttl_seconds`, which is how long to keep the actor alive while
# no tasks are being run.

actor = ActorEnvironment(
    name="agentic-rag-actor",
    ttl_seconds=30,
    container_image=image,
    requests=Resources(cpu="2", mem="4Gi", gpu="1"),
    limits=Resources(cpu="10", mem="10Gi", gpu="1"),
    #accelerator=L4,
)

EMBEDDING_MODEL = "nomic-ai/modernbert-embed-base"

# ## Creating a vector store `Artifact`
#

SemanticVectorStore = union.Artifact(name="semantic-search-vector-store")

@dataclass
class VectorStoreConfig:
    # Delimiters to split a document into chunks. The delimiters are used to
    # iteratively split the document into chunks. It will first try to split
    # the documents by the first delimiter. If the chunk is still larger than
    # the maximum chunk size, it will try to split the document by the second
    # delimiter, and so on.
    chunk_delimiters: list[str] = field(default_factory=lambda: ["_____", "___", "* * *", "**", "\n\n", ".\n"])

    # Approximate chunk size in characters
    approximate_chunk_size: int = 500



# ### Download the essays
@actor.task(container_image=image, cache=True, cache_version="0")
def download_csv() -> union.FlyteDirectory:
    """Download the essays from the csv file given a query."""
    
    import os
    import csv
    import requests

    CSV_URL = 'https://huggingface.co/datasets/sgoel9/paul_graham_essays/raw/main/pual_graham_essays.csv'


    with requests.Session() as s:
        download = s.get(CSV_URL)

        decoded_content = download.content.decode('utf-8')

        cr = csv.reader(decoded_content.splitlines(), delimiter=',')
        my_list = list(cr)

        documents_dir = f"{union.current_context().working_directory}/documents"
        os.makedirs(documents_dir, exist_ok=True)

        for row in my_list:
            print(row)
            essay_id = row[0]
            essay_title = row[1]
            essay_date = row[2]
            essay_text = row[3]
            text_fp = f"{documents_dir}/{essay_id}.txt"
            print(f"writing text to {text_fp}")
            with open(text_fp, "wb") as f:
                f.write(essay_text.encode("utf-8"))

    return union.FlyteDirectory(documents_dir)

def chunk_document(document: str, config: VectorStoreConfig) -> list[str]:
    """Helper function to chunk a document into smaller chunks."""
    
    if not document.strip():
        return []

    # Try to split the documents by iterating through the provided chunk
    # delimiters. If the largest chunk is smaller than the approximate_chunk_size,
    # we select that delimiter.
    _chunks = [document]  # fallback to whole document if no delimiter works
    for delimiter in config.chunk_delimiters:
        split_chunks = document.split(delimiter)
        if len(split_chunks) > 1 and max(len(x) for x in split_chunks) < config.approximate_chunk_size:
            _chunks = split_chunks
            break

    # Consolidate the chunks such that chunks are about the size specified by
    # the approximate_chunk_size.
    chunks: list[str] = []
    _new_chunk = ""
    for chunk in _chunks:
        if len(_new_chunk) > config.approximate_chunk_size:
            chunks.append(_new_chunk)
            _new_chunk = chunk
        else:
            _new_chunk += chunk

    # Don't forget to add the final chunk if it has content
    if _new_chunk.strip():
        chunks.append(_new_chunk)

    # If we still have no chunks, just return the whole document as one chunk
    if not chunks:
        chunks = [document.strip()]

    return chunks

@actor.task(
    container_image=image,
    cache=True,
    cache_version="3",
)
def create_vector_store(
    documents_dir: union.FlyteDirectory,
    config: VectorStoreConfig,
) -> Annotated[union.FlyteDirectory, SemanticVectorStore]:
    """Create a vector store from a directory of documents."""
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings
    import pyarrow as pa
    import tqdm
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,            # loads with SentenceTransformer under the hood
    )

    # Create a ChromaDB database and table.
    chromadb_dir = f"{union.current_context().working_directory}/chromadb"
    
    # Configure client settings to disable telemetry
    db = chromadb.PersistentClient(chromadb_dir, settings=Settings(anonymized_telemetry=False))
    collection = db.get_or_create_collection("essays", embedding_function=sentence_ef)


    documents_dir.download()
    documents_dir: Path = Path(documents_dir)

    document_paths = list(documents_dir.glob("**/*.txt"))

    # Iterate through the documents and add them to the vector store.
    for i, document_fp in tqdm.tqdm(
        enumerate(document_paths),
        total=len(document_paths),
        desc="chunking and embedding documents",
    ):

        with open(document_fp, "rb") as f:
            document = f.read().decode("utf-8")

        chunks = chunk_document(document, config)
        
        # Skip empty documents or failed chunking
        if not chunks:
            print(f"Warning: No chunks generated for {document_fp.name}, skipping...")
            continue
            
        collection.add(
            documents=chunks,
            ids=[f"{document_fp.stem}_{i}" for i in range(len(chunks))],
        )

    return union.FlyteDirectory(chromadb_dir)


@actor.task(
    container_image=image,
    cache=True,
    cache_version="0",
)
def retrieve_documents(
    vector_store: union.FlyteDirectory, 
    query: str, 
    max_results: int
) -> list[dict]:
    """Retrieve relevant documents from the vector store based on the query."""
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings
    from pathlib import Path

    # Download the vector store
    vector_store.download()
    chromadb_dir = Path(vector_store.path)

    # Create the same embedding function used during creation
    sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    # Load the ChromaDB database and collection with telemetry disabled
    db = chromadb.PersistentClient(chromadb_dir, settings=Settings(anonymized_telemetry=False))
    collection = db.get_collection("essays", embedding_function=sentence_ef)

    # Perform similarity search
    results = collection.query(
        query_texts=[query],
        n_results=min(max_results, collection.count()),  # Don't exceed available documents
        include=["documents", "distances"]
    )

    # Format the results into a more readable structure
    documents = []
    if results['documents'] and results['documents'][0]:  # Check if we have results
        for i, doc in enumerate(results['documents'][0]):
            documents.append({
                'document': doc,
                'distance': results['distances'][0][i] if results['distances'] else None,
                'id': results['ids'][0][i] if results['ids'] else None
            })

    return documents


@union.workflow
def query_essays(
    query: str,
    max_results: int = 5,
    vector_store: union.FlyteDirectory = SemanticVectorStore.query(),
) -> list[dict]:
    """Query an existing vector store for relevant Paul Graham essays."""
    return retrieve_documents(vector_store, query, max_results)


@union.workflow
def main(
    query: str,
    max_results: int,
    config: VectorStoreConfig = VectorStoreConfig(),
) -> tuple[union.FlyteDirectory, list[dict]]:
    """Main workflow to create a vector store from Paul Graham papers and retrieve relevant documents."""
    graham_papers = download_csv()
    vector_store = create_vector_store(graham_papers, config)
    relevant_documents = retrieve_documents(vector_store, query, max_results)
    return vector_store, relevant_documents