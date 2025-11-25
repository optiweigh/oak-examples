import depthai as dai
from config.config_data_classes import ModelNames


class NNArchiveProvider:
    """Loads NNArchives for different DepthAI models."""

    def __init__(self, platform: str, models: ModelNames):
        self._platform = platform
        self._models = models

    def _load(self, model: str) -> dai.NNArchive:
        """Loads a model from the DepthAI model zoo"""
        description = dai.NNModelDescription(model=model, platform=self._platform)
        archive = dai.NNArchive(dai.getModelFromZoo(description))
        return archive

    def face(self) -> dai.NNArchive:
        return self._load(self._models.face)

    def emotions(self) -> dai.NNArchive:
        return self._load(self._models.emotions)

    def age_gender(self) -> dai.NNArchive:
        return self._load(self._models.age_gender)

    def people(self) -> dai.NNArchive:
        return self._load(self._models.people)

    def reid(self) -> dai.NNArchive:
        return self._load(self._models.reid)
