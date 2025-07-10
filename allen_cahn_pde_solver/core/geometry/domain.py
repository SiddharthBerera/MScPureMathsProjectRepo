from abc import ABC, abstractmethod

class Domain(ABC):
    @abstractmethod
    def mesh(self, h: float):
        """Return a mesh object with max edge length h"""
    @abstractmethod
    def boundary_nodes(self, mesh):
        """Indices of boundary vertices for enforcing BCs"""


