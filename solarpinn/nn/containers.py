"""
Container classes for neural network modules.
"""

from __future__ import annotations

from collections.abc import Iterator

from .module import Module


class ModuleList(Module):
    """
    An ordered list of modules.

    Notes
    -----
    Unlike a standard Python list, every module added to a ModuleList is
    automatically registered as a child module.
    """

    def __init__(
        self,
        modules: list[Module] | None = None,
    ) -> None:
        super().__init__()

        self._list: list[Module] = []

        if modules is not None:
            self.extend(modules)

    def append(
        self,
        module: Module,
    ) -> None:
        """
        Append a module.
        """

        index = len(self._list)

        self._list.append(module)

        setattr(self, str(index), module)

    def extend(
        self,
        modules: list[Module],
    ) -> None:
        """
        Append multiple modules.
        """

        for module in modules:
            self.append(module)

    def __getitem__(
        self,
        index: int,
    ) -> Module:
        """
        Return the module at the given index.
        """

        return self._list[index]

    def __iter__(self) -> Iterator[Module]:
        """
        Iterate over modules.
        """

        return iter(self._list)

    def __len__(self) -> int:
        """
        Return the number of modules.
        """

        return len(self._list)

    def __contains__(
        self,
        module: Module,
    ) -> bool:
        """
        Return whether a module exists.
        """

        return module in self._list

    def forward(self, *args, **kwargs):
        """
        ModuleList cannot be called directly.
        """

        raise RuntimeError(
            "ModuleList does not implement a forward method."
        )


class ModuleDict(Module):
    """
    Dictionary of modules.
    """

    def __init__(
        self,
        modules: dict[str, Module] | None = None,
    ) -> None:
        super().__init__()

        self._dict: dict[str, Module] = {}

        if modules is not None:
            for name, module in modules.items():
                self.add(name, module)

    def add(
        self,
        name: str,
        module: Module,
    ) -> None:
        """
        Add a module.
        """

        self._dict[name] = module

        setattr(self, name, module)

    def keys(self):
        """
        Return dictionary keys.
        """

        return self._dict.keys()

    def values(self):
        """
        Return dictionary values.
        """

        return self._dict.values()

    def items(self):
        """
        Return dictionary items.
        """

        return self._dict.items()

    def __getitem__(
        self,
        key: str,
    ) -> Module:
        """
        Return a module.
        """

        return self._dict[key]

    def __iter__(self) -> Iterator[Module]:
        """
        Iterate over modules.
        """

        return iter(self._dict.values())

    def __len__(self) -> int:
        """
        Return the number of modules.
        """

        return len(self._dict)

    def __contains__(
        self,
        key: str,
    ) -> bool:
        """
        Return whether a key exists.
        """

        return key in self._dict

    def forward(self, *args, **kwargs):
        """
        ModuleDict cannot be called directly.
        """

        raise RuntimeError(
            "ModuleDict does not implement a forward method."
        )


class Sequential(ModuleList):
    """
    Sequential container.

    Modules are executed in the order they were added.
    """

    def __init__(
        self,
        *modules: Module,
    ) -> None:
        super().__init__(list(modules))

    def forward(self, x):
        """
        Execute all modules sequentially.
        """

        for module in self:
            x = module(x)

        return x