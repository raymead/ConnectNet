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
    return torch.zeros([6, 7]).view(1, 1, 6, 7)


def game_ended(state: torch.Tensor) -> Optional[int]:
    state = state.view(6, 7)
    state_sum = state[pairs_0, pairs_1].sum(axis=1)
    if state_sum.max() == 4:
        return 1
    elif state_sum.min() == -4:
        return -1
    elif state.count_nonzero() == 42:
        return 0
    else:
        return None


def get_valid_actions(state: torch.Tensor) -> List[int]:
    val = (state.view(6, 7)[0, :] == 0).nonzero()[:, 0].tolist()
    if isinstance(val, int):
        return [val]
    else:
        return val


# State is always oriented to it being red's turn
def next_state(state: torch.Tensor, action: int) -> torch.Tensor:
    state = state.detach().clone().view(6, 7)
    if state[0, action] != 0:
        print(state)
        print(action)
        raise ValueError(f"Invalid action {action} for state: \n{state}")
    elif state[5, action] == 0:
        state[5, action] = 1
        return state.view(1, 1, 6, 7)
    else:
        position = torch.nonzero(state[:, action]).min() - 1
        state[position, action] = 1
        return state.view(1, 1, 6, 7)


def to_rep(state: torch.Tensor) -> str:
    state = state + 1
    return ''.join(state.to(torch.int).numpy().flatten().astype(str))
