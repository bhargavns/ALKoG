"""Stable semantic categories shared by rendering, perception, and PPO.

Object identity and category identity are intentionally separate.  A scene may
eventually contain several instances of any category; the current toy world has
one instance of each category so that the first soft-category experiment stays
easy to inspect.
"""

OBJECT_CATEGORY_NAMES = ("food", "lion", "cage", "water", "poison")
BACKGROUND_NAME = "background"
ALL_CATEGORY_NAMES = OBJECT_CATEGORY_NAMES + (BACKGROUND_NAME,)

NUM_OBJECT_CATEGORIES = len(OBJECT_CATEGORY_NAMES)
BACKGROUND_ID = NUM_OBJECT_CATEGORIES
NUM_CLASSES = len(ALL_CATEGORY_NAMES)

CATEGORY_TO_ID = {name: idx for idx, name in enumerate(ALL_CATEGORY_NAMES)}
ID_TO_CATEGORY = {idx: name for name, idx in CATEGORY_TO_ID.items()}

# MuJoCo body names are deliberately mapped here rather than inferred from
# colors.  This mapping is used only for simulator-oracle training labels and
# evaluation; inference receives RGB images only.
BODY_TO_CATEGORY = {name: CATEGORY_TO_ID[name] for name in OBJECT_CATEGORY_NAMES}

# RGB colors used in debug images.
CATEGORY_COLORS = {
    CATEGORY_TO_ID["food"]: (30, 220, 30),
    CATEGORY_TO_ID["lion"]: (245, 120, 20),
    CATEGORY_TO_ID["cage"]: (40, 90, 245),
    CATEGORY_TO_ID["water"]: (20, 220, 240),
    CATEGORY_TO_ID["poison"]: (225, 40, 225),
    BACKGROUND_ID: (150, 150, 150),
}


def validate_category_metadata(class_names):
    """Reject checkpoints made for a different semantic vocabulary."""
    if tuple(class_names) != ALL_CATEGORY_NAMES:
        raise ValueError(
            "Category checkpoint vocabulary mismatch: "
            f"expected {ALL_CATEGORY_NAMES}, got {tuple(class_names)}"
        )
