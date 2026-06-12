import hashlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

import torch


class TensorCastGraphModulePass(ABC):
    """Use the same interface as Inductor's CustomGraphPass"""

    @abstractmethod
    def __call__(self, graph: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """
        Implementation of the custom pass.
        """

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used for code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        hasher = hashlib.sha256()
        src = inspect.getsource(self.__class__)  # pylint: disable=no-member
        hasher.update(src.encode("utf-8"))
        return hasher.hexdigest()
