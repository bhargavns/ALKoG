"""Scene object/relation vocabulary, symbol initialization, and checkpoint storage."""
import torch

SYMBOL_DIM = 4
RELATION_NAMES = ("inside", "near")


class SceneVocabulary:
    def __init__(self, bodies=(), device="cpu", seed=0):
        self.device = device
        self.bodies = tuple(bodies)
        self.relation_names = list(RELATION_NAMES)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        # Keep initialization ordering stable for existing Gumbel experiments.
        self.relation_symbols = torch.randn(len(self.relation_names), SYMBOL_DIM, generator=gen)
        self.symbols = (torch.stack([torch.randn(SYMBOL_DIM, generator=gen) for _ in self.bodies])
                        if self.bodies else torch.empty(0, SYMBOL_DIM))

    @property
    def num_objects(self):
        return len(self.symbols)

    def relation_index(self, name):
        return self.relation_names.index(name)

    def save(self, path):
        torch.save({"bodies": self.bodies, "symbols": self.symbols.cpu(),
                    "relation_names": self.relation_names,
                    "relation_symbols": self.relation_symbols.cpu()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        vocabulary = cls(blob.get("bodies", ()), device=device)
        vocabulary.symbols = blob["symbols"]
        vocabulary.relation_names = blob["relation_names"]
        vocabulary.relation_symbols = blob["relation_symbols"]
        return vocabulary


_BASE_BODIES = ("food", "lion", "cage", "receiver")
_RESOURCE_BODIES = ("water", "poison")


def scene_bodies(env):
    return _BASE_BODIES + (_RESOURCE_BODIES if env.resources else ())


def build_scene_vocabulary(bodies, device="cpu", seed=0):
    """Stable object IDs; randomly initialized symbols learned through PPO."""
    vocabulary = SceneVocabulary(bodies, device=device, seed=seed)
    return vocabulary, {name: i for i, name in enumerate(bodies)}


