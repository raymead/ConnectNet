import random
from typing import Optional, Union

import torch

import connect_four
import connect_net
import mcts
import rmcts
import utils


def competition_game(
        player1: Optional[Union[int, torch.nn.Module]] = None, player2: Optional[Union[int, torch.nn.Module]] = None,
        p1_mcts_sims: Optional[int] = None, p2_mcts_sims: Optional[int] = None, random_move: int = 5):
    gc = connect_four.GameCache()
    p1_mcts = get_mcts(player=player1, gc=gc)
    p2_mcts = get_mcts(player=player2, gc=gc)

    offset = random.randint(0, 1)
    game_info = []
    move = 1
    state = connect_four.start_state()
    rep = connect_four.to_rep(state=state)
    while True:
        if (move + offset) % 2 == 1:
            a = get_move(
                player=player1, move=move, state=state, rep=rep,
                mcts_search=p1_mcts, num_mcts_sims=p1_mcts_sims,
                random_move=random_move,
            )
        else:
            a = get_move(
                player=player2, move=move, state=state, rep=rep,
                mcts_search=p2_mcts, num_mcts_sims=p2_mcts_sims,
                random_move=random_move,
            )

        next_state, next_rep = gc.next_state_rep(state=state, rep=rep, action=a)
        ge = gc.game_ended(state=next_state, rep=next_rep)

        if ge is not None:
            mi = utils.MoveInfo(state=state, rep=rep, move=move, action=a, ge=-ge)
            game_info.append(mi)
            winner = -ge
            break
        else:
            mi = utils.MoveInfo(state=state, rep=rep, move=move, action=a)
            game_info.append(mi)

        move += 1
        state, rep = next_state, next_rep

    winner = "Draw" if winner == 0 else ("player1" if (move + offset) % 2 == 1 else "player2")
    return winner, move, state, game_info


def get_mcts(
        player: Union[int, None, torch.nn.Module],
        gc: connect_four.GameCache,
        ) -> Union[mcts.MCTS, random_mcts.RandomMCTS, None]:
    if isinstance(player, torch.nn.Module):
        p1_nc = connect_net.NetworkCache(nnet=player)
        p1_mcts = mcts.MCTS(c=1, game_cache=gc, network_cache=p1_nc)
    elif isinstance(player, int):
        p1_mcts = random_mcts.RandomMCTS(c=1, game_cache=gc)
    else:
        p1_mcts = None
    return p1_mcts


def get_move(player: Union[int, None, torch.nn.Module], move: int, state: torch.Tensor, rep: str,
             mcts_search: Union[mcts.MCTS, random_mcts.RandomMCTS, None], num_mcts_sims: Optional[int],
             random_move: int = 5) -> int:
    if isinstance(player, torch.nn.Module):
        mcts_search.simulate_moves(num_mcts_sims=num_mcts_sims, state=state, rep=rep, nnet=player)
        if move <= random_move * 2:
            a, _ = mcts_search.get_random_action(rep=rep)
        else:
            a = mcts_search.get_best_action(rep=rep)
    elif isinstance(player, int):
        mcts_search.simulate_games(num_mcts_sims=num_mcts_sims, state=state, rep=rep)
        if move <= random_move * 2:
            a, _ = mcts_search.get_random_action(rep=rep)
        else:
            a = mcts_search.get_best_action(rep=rep)
    else:
        a = int(random.choice(connect_four.possible_moves(state=state).tolist()))
    return a
