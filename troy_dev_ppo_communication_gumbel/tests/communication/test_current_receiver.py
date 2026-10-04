"""Single-message receiver interface, batch independence, and stateless inference."""
import pytest
import torch
from communication.ReceiverPolicy import CurrentMessageReceiverActorCritic


@pytest.mark.parametrize('batch_size', [1, 4, 10])
def test_message_outputs_do_not_depend_on_batch_peers_or_previous_calls(batch_size):
    torch.manual_seed(17)
    rx = CurrentMessageReceiverActorCritic(n_symbols=16)
    messages = torch.rand(batch_size, 16, requires_grad=True)
    dist, value = rx.distribution_messages(messages)
    # Calling with unrelated inputs cannot change the result for a message.
    rx.distribution_messages(torch.rand(5, 16))
    single_dist, single_value = rx.distribution_messages(messages[:1])
    torch.testing.assert_close(dist.probs[:1], single_dist.probs)
    torch.testing.assert_close(value[:1], single_value)
    (dist.log_prob(torch.zeros(batch_size, dtype=torch.long))[0] + value[0]).backward()
    assert messages.grad[0].abs().sum() > 0
    assert torch.count_nonzero(messages.grad[1:]) == 0


@pytest.mark.parametrize('shape', [(16,), (1, 1, 16), (1, 10, 16), (2, 15)])
def test_invalid_message_shape_rejected(shape):
    with pytest.raises(ValueError, match='one categorical message'):
        CurrentMessageReceiverActorCritic(n_symbols=16).distribution_messages(torch.zeros(shape))
