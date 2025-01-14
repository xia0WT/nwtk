import torch

th_int = torch.int32
th_float = torch.float32

class CaseInsensitiveDict(dict):
    """Case-insensitive dictionary implementation."""

    def __getitem__(self, key):
        return dict.__getitem__(self, key.casefold())

    def __setitem__(self, key, value):
        return dict.__setitem__(self, key.casefold(), value)

    # ---


    def __init__(self, seed=None, **kwargs):
        super().__init__()
        # Defer work to the method .update.
        self.update(seed)
        self.update(kwargs)

    def update(self, seed=None, **kwargs):
        if seed is None:
            seed = {}

        # Is the seed a mapping...
        if hasattr(seed, "items"):
            for key, value in seed.items():
                self[key] = value
        # or an iterable?
        else:
            for key, value in seed:
                self[key] = value

        for key, value in kwargs.items():
            self[key] = value


atom_list = CaseInsensitiveDict({"Fe": 1, "Co": 2, "Ni": 3, "Cu": 4, "Mn": 5, "Al": 6, "Mg": 7, "Zn":8})