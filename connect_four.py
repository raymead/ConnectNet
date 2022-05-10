from typing import Optional, List, Tuple

import torch


def get_pairs() -> List[Tuple[List[int], List[int]]]:
    possible_pairs = []
    for i in range(6):
        for j in range(7):
            possible_pairs.append([(i, j), (i+1, j), (i+2, j), (i+3, j)])
            possible_pairs.append([(i, j), (i+1, j+1), (i+2, j+2), (i+3, j+3)])
            possible_pairs.append([(i, j), (i, j+1), (i, j+2), (i, j+3)])
            possible_pairs.append([(i, j), (i-1, j+1), (i-2, j+2), (i-3, j+3)])
    pairs_filtered = []
    for plist in possible_pairs:
        plist = [(x, y) for x, y in plist if 0 <= x < 6 and 0 <= y < 7]
        if len(plist) == 4:
            pairs_filtered.append(([p[0] for p in plist], [p[1] for p in plist]))
    return pairs_filtered


pairs = get_pairs()
pairs_0 = torch.LongTensor([p[0] for p in pairs])
pairs_1 = torch.LongTensor([p[1] for p in pairs])


def start_state() -> torch.Tensor:
    return torch.zeros([6, 7])


def game_ended(state: torch.Tensor) -> Optional[int]:
    state_sum = state[pairs_0, pairs_1].sum(axis=1)
    if state_sum.max() == 4:
        return 1
    elif state_sum.min() == -4:
        return -1
    elif state.count_nonzero() == 42:
        return 0
    else:
        return None


# State is always oriented to it being red's turn
def next_state(state: torch.Tensor, action: int) -> torch.Tensor:
    state = state.clone()
    position = 5 - state[:, action].count_nonzero()
    if position == -1:
        raise ValueError(f"Invalid action {action} for state: \n{state}")
    else:
        state[position, action] = 1
        return state


def to_rep(state: torch.Tensor) -> str:
    return ''.join(f"{c:.0f}" for r in (state + 1).tolist() for c in r)
