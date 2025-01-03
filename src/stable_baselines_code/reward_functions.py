# all reward functions should have same signature:
# - losses: list of losses throughout run, losses[0] loss for no tactile points
# - occurrences: list of strings informing reward function about different occurrences at a given step
#       'missed': ray did not strike outline
#       'same': points creating ray are the same
# - avg reward: average reward over runs


def dummy_reward(losses, metrics, occurrences):
    return 0


def basic_reward(losses, metrics, occurrences):
    return metrics[-1]

def complex_reward(losses, metrics, occurences):
    if 'missed' in occurences:
        return -10
    if 'double' in occurences:
        return -10
    
    return metrics[-1]

# also referrd to as complex_reward_diff
def improve_reward(losses, metrics, occurences):
    if 'missed' in occurences:
        return -10
    if 'double' in occurences:
        return -10
    
    if len(metrics) < 2:
        return metrics[0]
    else:
        return metrics[-1] - metrics[-2]

def reward_1(losses, metrics, occurrences):
    alph = -1
    bet = -1

    missed = ('missed' in occurrences)
    same = ('same' in occurrences)

    return (losses[-2] - losses[-1]) + alph * missed + bet * same


def reward_2(losses, metrics, occurrences):
    alph = -1
    bet = -1

    missed = ('missed' in occurrences)
    same = ('same' in occurrences)

    return metrics[-1] + alph * missed + bet * same
