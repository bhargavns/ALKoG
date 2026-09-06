"""Stable semantic categories shared by rendering, perception, and PPO.

Object identity and category identity are intentionally separate.  A scene may
eventually contain several instances of any category; the current toy world has
one instance of each category so that the first soft-category experiment stays
easy to inspect.
"""

OBJECT_CATEGORY_NAMES = ("food", "lion", "cage", "water", "poison", "receiver")
BACKGROUND_NAME = "background"
ALL_CATEGORY_NAMES = OBJECT_CATEGORY_NAMES + (BACKGROUND_NAME,)

# The base world: the three objects every prior KG result was produced on, plus
# the receiving agent, which is present in BOTH worlds because identifying it is
# the point of the communication phase. The vocabulary itself never shrinks --
# the head always has NUM_CLASSES outputs, so one checkpoint format serves both
# worlds -- but only these categories can occur when the environment runs with
# resources=False. Metrics ignore zero-support classes (see
# CategoryData.classification_metrics), so absent water/poison do not distort
# macro-F1; a prediction of one of them is a real false positive and still shows
# up in the confusion matrix.
#
# `receiver` is appended last in OBJECT_CATEGORY_NAMES so food/lion/cage/water/
# poison keep ids 0-4; only BACKGROUND_ID and NUM_CLASSES shift. Checkpoints
# built against the old 6-class vocabulary are rejected by
# validate_category_metadata rather than silently misread.
BASE_OBJECT_CATEGORY_NAMES = ("food", "lion", "cage", "receiver")

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
    CATEGORY_TO_ID["receiver"]: (245, 215, 25),
    BACKGROUND_ID: (150, 150, 150),
}


def active_category_names(resources):
    """Object categories that can actually appear for a given env setting."""
    return OBJECT_CATEGORY_NAMES if resources else BASE_OBJECT_CATEGORY_NAMES


def validate_category_metadata(class_names):
    """Reject checkpoints made for a different semantic vocabulary."""
    if tuple(class_names) != ALL_CATEGORY_NAMES:
        raise ValueError(
            "Category checkpoint vocabulary mismatch: "
            f"expected {ALL_CATEGORY_NAMES}, got {tuple(class_names)}"
        )
