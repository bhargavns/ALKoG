"""
Agent architecture configuration.

Covers three sub-systems:
  1. Visual encoder  — maps a rendered observation to a feature vector
  2. Token vocabulary — the agent's finite "alphabet" for communication
  3. Policy network   — combines visual features + KG embedding + utterance
                        history into action logits and a value estimate

Design note on the visual encoder:
  We start with a pretrained ResNet backbone.  The early convolutional
  layers capture low-level edge/texture features that transfer well to
  the procedurally generated simulation scenes.  Only the later layers
  and the projection head are trained from scratch, keeping initial
  training stable and fast.
"""

from typing import Literal

from pydantic import BaseModel, Field


class VisualEncoderConfig(BaseModel):
    backbone: Literal["resnet18", "resnet50"] = Field(
        default="resnet18",
        description=(
            "Torchvision backbone.  resnet18 is faster and sufficient for "
            "the small-scene early curriculum phases; upgrade to resnet50 "
            "for Phase 3/4 once the basic pipeline is validated."
        ),
    )
    pretrained: bool = Field(
        default=True,
        description="Load ImageNet-pretrained weights.",
    )
    freeze_early_layers: bool = Field(
        default=True,
        description=(
            "Freeze the first two ResNet layer groups (layer1, layer2). "
            "These learn generic low-level features that transfer well "
            "and don't need retraining for this domain."
        ),
    )
    output_dim: int = Field(
        default=256,
        gt=0,
        description=(
            "Dimensionality of the projected visual feature vector output "
            "by the encoder (after the linear projection head)."
        ),
    )
    bounding_box_pool: Literal["roi_align", "crop_resize"] = Field(
        default="roi_align",
        description=(
            "Method used to extract features for a proposed bounding box region. "
            "roi_align is differentiable and standard in object detection; "
            "crop_resize is simpler if roi_align causes dependency issues."
        ),
    )


class TokenVocabConfig(BaseModel):
    vocab_size: int = Field(
        default=64,
        gt=0,
        description=(
            "Number of distinct tokens in the agent's vocabulary.  "
            "Initially all are meaningless; meaning emerges through "
            "communicative success (Stage 3 symbol binding)."
        ),
    )
    max_utterance_length: int = Field(
        default=8,
        gt=0,
        description=(
            "Maximum number of tokens per utterance.  Shorter cap forces "
            "the agent to be concise; increase for Phase 4 compositional tasks."
        ),
    )
    utterance_history_length: int = Field(
        default=16,
        gt=0,
        description=(
            "Number of past (token_sequence, outcome) pairs kept in the "
            "state's utterance history buffer."
        ),
    )
    token_embedding_dim: int = Field(
        default=64,
        gt=0,
        description="Dimensionality of each token's learned embedding.",
    )
    binding_learning_rate: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Rate at which token-to-meaning binding strengths are updated "
            "after a communication success or failure.  Separate from the "
            "neural network learning rate."
        ),
    )


class PolicyNetworkConfig(BaseModel):
    hidden_dim: int = Field(
        default=512,
        gt=0,
        description="Hidden dimension in the shared trunk of the actor-critic network.",
    )
    num_trunk_layers: int = Field(
        default=2,
        gt=0,
        description="Number of MLP layers in the shared trunk.",
    )
    use_layer_norm: bool = Field(
        default=True,
        description="Apply LayerNorm after each trunk layer.  Stabilises PPO training.",
    )
    action_head_hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dim in the actor (action logits) head.",
    )
    value_head_hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dim in the critic (state value) head.",
    )


class AgentConfig(BaseModel):
    visual_encoder: VisualEncoderConfig = Field(default_factory=VisualEncoderConfig)
    token_vocab: TokenVocabConfig = Field(default_factory=TokenVocabConfig)
    policy_network: PolicyNetworkConfig = Field(default_factory=PolicyNetworkConfig)

    num_agents: int = Field(
        default=1,
        gt=0,
        description=(
            "Number of agents in the environment.  1 for Phases 1-2 "
            "(solo exploration); 2 for Phases 3-4 (partner communication). "
            "Overridden per curriculum phase at runtime."
        ),
    )
    share_policy: bool = Field(
        default=True,
        description=(
            "Whether both agents share a single policy network (parameter sharing). "
            "Recommended True: cuts memory in half and has been shown to speed "
            "up emergence of shared communication protocols."
        ),
    )
