import abc
import pathlib


class AbstractDataLoader(abc.ABC):
    @abc.abstractmethod
    def load_knowledgebase_to_collection(self, knowledgebase_path: pathlib.Path) -> None: ...
