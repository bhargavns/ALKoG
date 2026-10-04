"""Episode-local perception cadence shared by training and inference."""


def perception_due(env, episode_step, perceive_passes=1):
    """Called after a nonterminal step; reset perception is handled by callers.

    Co-located mode refreshes scene perception: reset, then every 75 steps.
    Stationary mode uses its opening perception passes.
    """
    if getattr(env, "colocated", False):
        return episode_step > 0 and episode_step % 75 == 0
    return episode_step < perceive_passes
