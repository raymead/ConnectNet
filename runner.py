import glob
import os
import pickle
import time
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

import connect_net
import mcts
import utils

T_GAMES = List[List]


def start_simulation(
        model: Optional[str], trial: str, iteration: int,
        num_processes: int, num_episodes: int, num_mcts_sims: int) -> None:
    # Setup save folder
    folder = utils.get_simulation_folder(trial=trial, iteration=iteration)

    # Load model
    if model is None:
        nnet = connect_net.ConnectNet()
    else:
        nnet = connect_net.load_model(model)
    nnet.share_memory()
    connect_net.save_model(nnet, get_model_path(folder=folder, iteration=iteration))

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=mcts.find_policy_examples, args=(nnet, folder, iteration, i, num_episodes, num_mcts_sims))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def train_model(
        trial: str, iteration: int,
        learning_rate: float = 0.001, epochs: int = 250, train_frac: float = 0.85):
    sim_folder = utils.get_simulation_folder(trial=trial, iteration=iteration)
    sim_model_path = get_model_path(folder=sim_folder, iteration=iteration)
    sim_files = glob.glob(os.path.join(sim_folder, f"examples-{iteration}-*.pickle"))
    nnet = connect_net.load_model(sim_model_path)

    examples = []
    for fname in sim_files:
        with open(fname, "rb") as f:
            examples += pickle.load(f)
    train_games, val_games = split_games(examples=examples, train_fraction=train_frac)

    train_boards, train_scores, train_moves = split_items(games=train_games)
    print(train_boards.shape, train_scores.shape, train_moves.shape)
    val_boards, val_scores, val_moves = split_items(games=val_games)
    print(val_boards.shape, val_scores.shape, val_moves.shape)

    optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate)
    losses_train = []
    losses_val = []
    for i in range(epochs):
        start_time = time.time()
        v, proba = nnet(train_boards.view(-1, 1, 6, 7))
        train_loss = full_loss_fn(v, proba, train_scores, train_moves)
        losses_train.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        v, proba = nnet(val_boards.view(-1, 1, 6, 7))
        val_loss = full_loss_fn(v, proba, val_scores, val_moves).item()
        losses_val.append(val_loss)
        if i % 10 == 0:
            print(f"EPOCH::{i}  LOSS::{val_loss:.4f}  TIME::{time.time() - start_time:.4f}")

    training_folder = utils.get_training_folder(trial=trial, iteration=iteration)
    training_path = get_model_path(folder=training_folder, iteration=iteration)
    losses_train_path = get_losses_train_path(folder=training_folder, iteration=iteration)
    losses_val_path = get_losses_val_path(folder=training_folder, iteration=iteration)

    connect_net.save_model(nnet=nnet, path=training_path)
    with open(losses_train_path, "wb") as f:
        pickle.dump(obj=losses_train, file=f)
    with open(losses_val_path, "wb") as f:
        pickle.dump(obj=losses_val, file=f)


def full_loss_fn(est_scores, est_probs, target_scores, target_probs):
    loss_scores = (est_scores - target_scores).square().sum()
    loss_probs = (target_probs * est_probs.log()).sum()
    loss_total = loss_scores - loss_probs
    return loss_total


def create_pretrained_model():
    boards = np.load(os.path.join(utils.get_data_folder(), "move8_boards.npy"))
    winners = np.load(os.path.join(utils.get_data_folder(), "move8_winner.npy"))

    total_num = len(boards)
    train_num, val_num = utils.split_train_val_values(total_count=total_num, train_fraction=0.85)
    shuffled_indexes = utils.get_shuffled_list(size=total_num)

    train_vals = torch.Tensor(boards[shuffled_indexes[:train_num]]).unsqueeze(1)
    valid_vals = torch.Tensor(boards[shuffled_indexes[train_num:train_num + val_num]]).unsqueeze(1)

    train_targ = torch.Tensor(winners[shuffled_indexes[:train_num]])
    valid_targ = torch.Tensor(winners[shuffled_indexes[train_num:train_num + val_num]])
    print(len(train_vals), len(valid_vals))

    nnet = connect_net.ConnectNet()
    optimizer = torch.optim.Adam(nnet.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss(reduction="sum")
    l2_lambda = 1
    losses_train = []
    losses_validation = []
    for i in range(250):
        start_time = time.time()
        train_v, proba = nnet(train_vals)
        if torch.isnan(proba).any():
            print("Found nan")
            break
        reg_loss = l2_lambda * sum(p.square().sum() for p in nnet.parameters())

        train_loss = loss_function(train_v, train_targ)
        train_loss = train_loss + reg_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        losses_train.append(train_loss.item())

        valid_v, _ = nnet(valid_vals)
        val_loss = loss_function(valid_v, valid_targ).item()
        losses_validation.append(val_loss)
        if i % 10 == 0:
            print(f"EPOCH::{i}  LOSS::{val_loss:.4f}  TIME::{time.time() - start_time:.4f}")

    # TODO: make nicer
    connect_net.save_model(nnet, "models/pretrain01.model")


def split_games(examples: List[List[utils.MoveInfo]], train_fraction: float):
    total_num = len(examples)
    train_num, val_num = utils.split_train_val_values(total_count=total_num, train_fraction=train_fraction)
    shuffled_indexes = utils.get_shuffled_list(size=total_num)

    train_games = [examples[i] for i in shuffled_indexes[:train_num]]
    val_games = [examples[i] for i in shuffled_indexes[train_num:]]
    return train_games, val_games


def split_items(games: List[List[utils.MoveInfo]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boards = torch.stack([m.state for g in games for m in g])
    scores = torch.Tensor([m.ge for g in games for m in g])
    moves = torch.stack([m.pi_val for g in games for m in g])
    return boards, scores, moves


# File Paths
def get_model_path(folder: str, iteration: int) -> str:
    return os.path.join(folder, f"model-{iteration}.model")


def get_losses_val_path(folder: str, iteration: int) -> str:
    return os.path.join(folder, f"validation-losses-{iteration}.pickle")


def get_losses_train_path(folder: str, iteration: int) -> str:
    return os.path.join(folder, f"training-losses-{iteration}.pickle")


if __name__ == '__main__':
    trial_name = "test01"
    for j in range(8, 26):
        if j == 1:
            prev_path = "models/pretrain01.model"
        else:
            prev_folder = utils.get_training_folder(trial=trial_name, iteration=j-1)
            prev_path = get_model_path(folder=prev_folder, iteration=j-1)
        start_simulation(
            model=prev_path, trial=trial_name, iteration=j,
            num_processes=6, num_episodes=256, num_mcts_sims=64,
        )
        train_model(trial=trial_name, iteration=j, epochs=50)
