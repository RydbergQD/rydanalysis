from abc import abstractmethod


class ImageAnalysis:
    def __init__(self, preprocessor, processor, fitter):
        self.preprocessor = preprocessor
        self.processor = processor
        self.fitter = fitter


class Preprocessor:
    @abstractmethod
    @property
    def background(self):
        pass

    @abstractmethod
    @property
    def light(self):
        pass

    @abstractmethod
    @property
    def atoms(self):
        pass


class LivePreprocessor(Preprocessor):
    @property
    def background(self):
        pass

    @property
    def light(self):
        pass

    @property
    def atoms(self):
        pass
