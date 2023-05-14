from abc import ABC, abstractmethod


class InfillModel:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def predict(self, input_text: str) -> str:
        pass

    @abstractmethod
    def infill(self, input_text) -> str:
        pass
