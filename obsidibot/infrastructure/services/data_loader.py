import logging
import pathlib

import anyio
import chromadb
import uvloop
from langchain import text_splitter

from obsidibot.domain.services.data_loader import AbstractDataLoader


logger = logging.getLogger(__name__)


class AsyncDataLoader(AbstractDataLoader):
    def __init__(self, collection: chromadb.api.models.AsyncCollection) -> None:
        self._collection = collection
        self._splitter = text_splitter.RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

    async def _load_file_to_collection(self, file_path: pathlib.Path) -> None:
        async with await anyio.open_file(file_path, mode='r') as file:
            file_content = await file.read()
            chunks = self._splitter.split_text(file_content)
            for i, chunk in enumerate(chunks):
                await self._collection.add(
                    documents=[chunk],
                    metadatas=[{'source': str(file_path)}],
                    ids=[f'{file_path}-chunk-{i}'],
                )

    def load_knowledgebase_to_collection(self, knowledgebase_path: pathlib.Path) -> None:
        uvloop.run(self._load_knoledgebase_to_collection(knowledgebase_path))

    async def _load_knoledgebase_to_collection(self, knoledgebase_path: pathlib.Path) -> None:
        for file_path in knoledgebase_path.glob('**/*.md'):
            await self._load_file_to_collection(file_path)


class DataLoader(AbstractDataLoader):
    def __init__(self, collection: chromadb.api.models.Collection) -> None:
        """
        Initialize the DataLoader with a chromadb collection.

        Args:
            collection: The chromadb collection to load data into
            max_workers: Maximum number of worker threads to use (None means auto-determine)

        """
        self._collection = collection
        self._splitter = text_splitter.RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self._max_workers = 10

    def _load_file_to_collection(self, file_path: pathlib.Path) -> None:
        """
        Load a single file into the collection.

        Args:
            file_path: Path to the file to load

        """
        with file_path.open('r', encoding='utf-8') as file:
            file_content = file.read()
            chunks = self._splitter.split_text(file_content)

            # Prepare batch data for adding to collection
            documents = []
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append({'source': str(file_path)})
                ids.append(f'{file_path}-chunk-{i}')

            # Add all chunks in a single batch operation if there are any
            if documents:
                self._collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )

    def load_knowledgebase_to_collection(self, knowledgebase_path: pathlib.Path) -> None:
        """
        Load all markdown files from the knowledgebase path into the collection using multithreading.

        Args:
            knowledgebase_path: Path to the directory containing markdown files

        """
        markdown_files = list(knowledgebase_path.glob('**/*.md'))
        for file in markdown_files:
            self._load_file_to_collection(file)
