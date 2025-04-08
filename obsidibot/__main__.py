import logging
import pathlib

import chromadb

from obsidibot.infrastructure.repositories.collection_repository import CollectionRepository
from obsidibot.infrastructure.services.data_loader import DataLoader


logger = logging.getLogger(__name__)


def main() -> None:
    client = chromadb.PersistentClient(path='data/persistant_storage')
    collection = client.get_or_create_collection('obsidibot')

    data_loader = DataLoader(collection)
    data_loader.load_knowledgebase_to_collection(pathlib.Path('data/personal_notes'))

    collection_repository = CollectionRepository(collection)
    result = collection_repository.query(
        'Отдыхать',
        n_results=5,
    )
    logger.info(result)


if __name__ == '__main__':
    main()
