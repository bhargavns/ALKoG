import torch
import torch.nn as nn

from representation.Grounding import TRIPLE_PAD
from representation.SceneVocabulary import SYMBOL_DIM

N_TRIPLE_SLOTS = 3  # src / relation / dst positions within a triple


class SymbolTable(nn.Module):
    """Trainable symbol machinery for the scene triple encoder.

    Holds the 4-dim node and relation symbols (initialized from a fixed
    SceneVocabulary), a 3x4 positional embedding table (src/rel/dst slot within a
    triple), and the shared 2->4 distance projection that gates node tokens
    by their agent-frame (forward, left) ground-plane offset. All of it
    receives PPO gradients; scene vocabulary identities and visual features stay
    frozen.
    """

    # Parameters:
    #   vocabulary: SceneVocabulary containing object/relation IDs and initial embedding tensors.
    # pos_embeddings/dist_proj retain serialized names for existing checkpoints.
    def __init__(self, vocabulary):
        super().__init__()
        self.node_symbols = nn.Parameter(vocabulary.symbols.clone())
        self.relation_symbols = nn.Parameter(vocabulary.relation_symbols.clone())
        self.pos_embeddings = nn.Parameter(torch.randn(N_TRIPLE_SLOTS, SYMBOL_DIM))
        self.dist_proj = nn.Linear(2, SYMBOL_DIM)  # one set of weights for all nodes
        # identity (FiLM-style) init: the gate starts as a pass-through so
        # symbols receive full-strength gradient from step one, and distance
        # modulation is learned as a perturbation on top
        nn.init.zeros_(self.dist_proj.weight)
        nn.init.ones_(self.dist_proj.bias)

    # Parameters (B = batch size, K = scene rows, V = message categories):
    #   triples: Integer scene rows [B,K,3]: source/relation/destination IDs, with -1 for padding.
    #   object_offsets: Normalized forward/left offsets [B,K,2,2] for each row's source and destination.
    def forward(self, triples, object_offsets):
        """triples: LongTensor [B, K, 3] with TRIPLE_PAD for empty slots.
        object_offsets: FloatTensor [B, K, 2, 2] -- agent->src and agent->dst offsets.

        Per token: symbol + positional embedding, then (node tokens only)
        elementwise-multiplied by the projected offset. PAD tokens are zeroed.
        Returns (per-triple encodings [B, K, 3 * SYMBOL_DIM] laid out
        [src || rel || dst], validity mask [B, K]).
        """

        # Parameters:
        #   embedding_table: Trainable object or relation embeddings indexed by the supplied IDs.
        #   indices: Object/relation ID tensor [B,K]; -1 entries are masked after safe lookup.
        #   triple_position: Token position inside a triple: 0 source, 1 relation, 2 destination.
        #   object_offset: Optional forward/left offsets [B,K,2]; modulate object embeddings, not relation embeddings.
        def token(embedding_table, indices, triple_position, object_offset=None):
            valid_token_mask = (indices != TRIPLE_PAD).unsqueeze(-1).float()
            token_embedding = embedding_table[indices.clamp(min=0)] + self.pos_embeddings[triple_position]
            if object_offset is not None:
                token_embedding = token_embedding * self.dist_proj(object_offset)
            return token_embedding * valid_token_mask

        token_encodings = [
            token(self.node_symbols, triples[..., 0], 0, object_offsets[..., 0, :]),
            token(self.relation_symbols, triples[..., 1], 1),
            token(self.node_symbols, triples[..., 2], 2, object_offsets[..., 1, :]),
        ]
        return torch.cat(token_encodings, dim=-1), triples[..., 0] != TRIPLE_PAD


class SlotAttention(nn.Module):
    """Learned slot queries cross-attending over the per-triple encodings.

    Each of n_slots learned queries fills its slot with a softmax-weighted
    mixture of triple encodings, so the slot a triple occupies in the
    perception output is irrelevant -- what a slot holds is decided by its
    query. This makes the flat concatenated input permutation-invariant to
    perception's triple ordering (which shifts pass to pass as the detected
    entry mix changes) while keeping a fixed-size, slot-structured vector,
    independent of the number of object categories.

    PAD triples are masked to a large negative logit; their encodings are
    all-zero anyway (and value has no bias), so a row with no valid triples
    yields all-zero slots rather than NaNs.
    """

    # Parameters:
    #   num_slots: Number of learned attention queries and output slots.
    #   encoding_dimension: Width of each encoded triple and attention slot, normally 12.
    def __init__(self, num_slots, encoding_dimension):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_slots, encoding_dimension))
        self.key = nn.Linear(encoding_dimension, encoding_dimension, bias=False)
        self.value = nn.Linear(encoding_dimension, encoding_dimension, bias=False)
        self.scale = encoding_dimension**0.5

    # Parameters (B = batch size, K = scene rows, V = message categories):
    #   triple_encodings: Encoded scene rows [B,K,D] from SymbolTable.forward().
    #   valid_triples: Boolean mask [B,K] marking scene rows with a real source object.
    def forward(self, triple_encodings, valid_triples):
        """triple_encodings [B,K,D], valid_triples [B,K] -> flat slots [B, num_slots * D]."""
        attention_logits = torch.einsum("sd,bkd->bsk", self.queries, self.key(triple_encodings)) / self.scale
        attention_logits = attention_logits.masked_fill(~valid_triples.unsqueeze(1), -1e9)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        slots = torch.einsum("bsk,bkd->bsd", attention_weights, self.value(triple_encodings))
        return slots.reshape(triple_encodings.shape[0], -1)


# Parameters:
#   input_dimension: Number of input features to the Linear -> Tanh -> Linear network.
#   output_dimension: Number of output logits or scalar predictions produced by that network.
#   hidden_dimension: Width of the hidden layer in the actor/critic or transmitter message-head MLP.
def _mlp(input_dimension, output_dimension, hidden_dimension):
    return nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.Tanh(),
        nn.Linear(hidden_dimension, output_dimension),
    )


class GumbelTransmitter(nn.Module):
    """Perceptual transmitter with categorical message logits and no private critic."""

    # Parameters:
    #   vocabulary: SceneVocabulary containing object/relation IDs and initial embedding tensors.
    #   observation_dimension: Number of transmitter pose inputs; the arena supplies four.
    #   k_triples: Number of input rows/attention slots, INCLUDING the reserved receiver row in this model.
    #   hidden_dimension: Width of the hidden layer in the actor/critic or transmitter message-head MLP.
    #   n_symbols: Number of categories in the communication vocabulary (V), at least two.
    def __init__(self, vocabulary, observation_dimension=4, k_triples=7, hidden_dimension=38, n_symbols=16):
        super().__init__()
        self.symbol_table = SymbolTable(vocabulary)
        # slot_attn is the established checkpoint key for the attention module.
        self.slot_attn = SlotAttention(k_triples, N_TRIPLE_SLOTS * SYMBOL_DIM)
        self.message_head = _mlp(observation_dimension + k_triples * 12, n_symbols, hidden_dimension)
        nn.init.normal_(self.message_head[-1].weight, std=0.01)
        nn.init.zeros_(self.message_head[-1].bias)

    # Parameters (B = batch size, K = scene rows, V = message categories):
    #   observation: Transmitter pose [B,4]: normalized x/y and cosine/sine of yaw.
    #   triples: Integer scene rows [B,K,3]: source/relation/destination IDs, with -1 for padding.
    #   object_offsets: Normalized forward/left offsets [B,K,2,2] for each row's source and destination.
    def forward(self, observation, triples, object_offsets):
        triple_encodings, valid_triples = self.symbol_table(triples, object_offsets)
        scene_features = torch.cat((observation, self.slot_attn(triple_encodings, valid_triples)), dim=-1)
        return self.message_head(scene_features)


class OracleSequenceTransmitter(nn.Module):
    """Private oracle evidence -> unrestricted ordered logits [B,L,V].

    Concept features and public tokens have independent dimensions. No message
    position is assigned an entity, cage, or safety meaning.
    """

    def __init__(self, input_dimension=5, message_length=2, n_symbols=6, hidden_dimension=64):
        super().__init__()
        if message_length < 1 or n_symbols < 2:
            raise ValueError('Positive message length and vocabulary >= 2 required')
        self.message_length, self.n_symbols = message_length, n_symbols
        self.message_head = _mlp(input_dimension, message_length * n_symbols, hidden_dimension)

    def forward(self, evidence):
        return self.message_head(evidence).reshape(-1, self.message_length, self.n_symbols)
