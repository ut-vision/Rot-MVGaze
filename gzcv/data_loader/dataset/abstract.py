import abc


class AbstractIndex(metaclass=abc.ABCMeta):
    def __init__(self):
        self.indices = self.build_index()

    @abc.abstractmethod
    def build_index(self):
        pass

    def __getitem__(self, idx):
        return self.indices[idx]

    def __len__(self):
        return len(self.indices)
