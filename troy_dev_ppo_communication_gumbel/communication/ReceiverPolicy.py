"""Blind current-message receiver with separate actor and critic MLPs."""
import torch
from torch import nn
from torch.distributions import Categorical
from representation.SceneVocabulary import SYMBOL_DIM


class CurrentMessageReceiverActorCritic(nn.Module):
    """One message per batch item -> separate actor/critic MLPs, without memory.

    Embedding width and head hidden width default to four.
    """

    # Parameters:
    #   n_actions: Number of receiver movements; the arena uses four.
    #   embedding_dimension: Width of the trainable receiver message embedding.
    #   hidden_dimension: Width of the hidden layer in the actor/critic or transmitter message-head MLP.
    #   n_symbols: Number of categories in the communication vocabulary (V), at least two.
    def __init__(self, n_actions=4, embedding_dimension=SYMBOL_DIM, hidden_dimension=SYMBOL_DIM, n_symbols=4):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.token_embeddings = nn.Embedding(n_symbols, embedding_dimension)
        nn.init.normal_(self.token_embeddings.weight, std=0.5)
        self.actor = _mlp(embedding_dimension, n_actions, hidden_dimension)
        self.critic = _mlp(embedding_dimension, 1, hidden_dimension)

    # Parameters (B = batch size, K = scene rows, V = message categories):
    #   messages: One categorical message per batch item [B,V]; one-hot forward values retain channel gradients.
    def distribution_messages(self, messages):
        """Map categorical messages [B,V] to action distributions and values.

        Matrix multiplication preserves straight-through gradients to every
        category. There is no time axis, history, or message-validity mask.
        """
        if messages.ndim != 2 or messages.shape[1] != self.token_embeddings.num_embeddings:
            raise ValueError('Expected one categorical message per batch item [B,V]')
        message_embeddings = messages @ self.token_embeddings.weight
        return Categorical(logits=self.actor(message_embeddings)), self.critic(message_embeddings).squeeze(-1)


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
