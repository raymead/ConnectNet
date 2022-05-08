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
pairs_0 = [p[0] for p in pairs]
pairs_1 = [p[1] for p in pairs]


def start_state() -> torch.Tensor:
    return torch.zeros([1, 1, 6, 7])


def game_ended(state: torch.Tensor) -> Optional[int]:
    state = state.squeeze()
    state_sum = state[pairs_0, pairs_1].sum(axis=1)
    if (state_sum == 4).any():
        return 1
    elif (state_sum == -4).any():
        return -1
    # # Old Code
    # for plist in pairs:
    #     val = state[plist[0], plist[1]].sum()
    #     if val == 4:
    #         return 1
    #     elif val == -4:
    #         return -1
    if len(get_valid_actions(state)) == 0:
        return 0
    return None


def get_valid_actions(state: torch.Tensor) -> List[int]:
    val = torch.nonzero(((state.squeeze() != 0).sum(axis=0) < 6)).squeeze().tolist()
    if isinstance(val, int):
        return [val]
    else:
        return val


# State is always oriented to it being red's turn
def next_state(state: torch.Tensor, action: int) -> torch.Tensor:
    state = state.squeeze().clone()
    position = torch.argwhere(state[:, action] != 0)

    if len(position) == 0:
        state[5, action] = 1
        return state.reshape(1, 1, 6, 7)

    position = position.min()
    if position == 0:
        print(state)
        print(action)
        raise ValueError(f"Invalid action {action} for state: \n{state}")
    else:
        state[position - 1, action] = 1
        return state.reshape(1, 1, 6, 7)


def to_rep(state: torch.Tensor) -> str:
    state = state + 1
    return ''.join(state.to(torch.int).numpy().flatten().astype(str))
