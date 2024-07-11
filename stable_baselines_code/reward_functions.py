# all reward functions should have same signature:
# - losses: list of losses throughout run, losses[0] loss for no tactile points
# - occurrences: list of strings informing reward function about different occurrences at a given step
#       'missed': ray did not strike outline
#       'same': points creating ray are the same
# - avg reward: average reward over runs


def dummy_reward(losses, occurrences, avg_reward):
    return 0


def basic_reward(losses, occurrences, avg_reward):
    return -losses[-1]


def reward_1(losses, occurrences, avg_reward):
    alph = -1
    bet = -1

    missed = ('missed' in occurrences) * avg_reward
    same = ('same' in occurrences) * avg_reward

    return (losses[-2] - losses[-1]) + alph * missed + bet * same
