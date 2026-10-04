"""Conditional categorical sender entropy and iteration schedule."""
import math

# Parameters:
#   iteration: One-based outer training iteration (or direct-loss diagnostic update).
#   total_iterations: Full iteration budget used to locate the current point in the entropy schedule.
#   entropy_bonus_coefficient: Nonnegative initial/constant sender entropy weight, conventionally beta.
#   schedule: constant or annealed; annealed holds to 25% and reaches zero at 75% of the budget.
def entropy_coefficient(iteration, total_iterations, entropy_bonus_coefficient, schedule):
    """One-based updates: hold first quarter, decay to zero at three quarters."""
    if not math.isfinite(entropy_bonus_coefficient) or entropy_bonus_coefficient < 0:
        raise ValueError('entropy beta must be finite and nonnegative')
    if schedule not in ('constant', 'annealed'):
        raise ValueError('Unknown entropy schedule')
    if total_iterations < 1 or not 1 <= iteration <= total_iterations:
        raise ValueError('Update must be within training budget')
    if schedule == 'constant':
        return entropy_bonus_coefficient
    return entropy_bonus_coefficient * max(0., min(1., (0.75 * total_iterations - iteration) / (0.5 * total_iterations)))


# Parameters (B = batch size, K = scene rows, V = message categories):
#   symbol_logits: Unnormalized sender category scores [...,V]; entropy averages over the leading axes.
def sender_entropy(symbol_logits):
    symbol_log_probabilities = symbol_logits.log_softmax(dim=-1)
    return -(symbol_log_probabilities.exp() * symbol_log_probabilities).sum(dim=-1).mean()
