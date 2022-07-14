import glob
import os
import pickle
import time
from typing import Optional, List, Tuple, Type

import torch
import torch.multiprocessing as mp

import connect_net
import mcts
import utils


def start_simulation(
        model: Optional[str], klass: Type[torch.nn.Module], trial: str, iteration: int, c: float, random_moves: int,
        num_processes: int, num_episodes: int, num_mcts_sims: int, batch_size: int, prob_move8: float) -> None:
    # Setup save folder
    folder = utils.get_simulation_folder(trial=trial, iteration=iteration)
    simulation_model_path = utils.get_simulation_model_path(trial=trial, iteration=iteration)

    # Load model
    nnet = klass() if model is None else connect_net.load_model(path=model, klass=klass)
    nnet.eval()
    nnet.share_memory()
    connect_net.save_model(nnet=nnet, path=simulation_model_path)

    processes = []
    for i in range(num_processes):
        p = mp.Process(
            target=mcts.generate_examples,
            kwargs=dict(
                nnet=nnet, folder=folder, iteration=iteration, process=i,
                c=c, num_episodes=num_episodes, num_mcts_sims=num_mcts_sims, random_moves=random_moves,
                batch_size=batch_size, prob_move8=prob_move8,
            ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def train_model(
        klass: Type[torch.nn.Module], trial: str, iteration: int,
        learning_rate: float, epochs: int, train_frac: float, l2_lambda: float) -> None:
    sim_model_path = utils.get_simulation_model_path(trial=trial, iteration=iteration)
    sim_folder = utils.get_simulation_folder(trial=trial, iteration=iteration)
    sim_files = glob.glob(os.path.join(sim_folder, f"examples-{iteration}-*.pickle"))
    nnet = connect_net.load_model(path=sim_model_path, klass=klass)
    nnet.train()

    # Load Games
    examples = []
    for fname in sim_files:
        with open(fname, "rb") as f:
            examples += pickle.load(f)
    # Split games
    train_games, val_games = split_games(examples=examples, train_fraction=train_frac)
    # Make samples
    train_boards, train_scores, train_moves = split_items(games=train_games)
    print(train_boards.shape, train_scores.shape, train_moves.shape)
    val_boards, val_scores, val_moves = split_items(games=val_games)
    print(val_boards.shape, val_scores.shape, val_moves.shape)

    optimizer = torch.optim.Adam(nnet.parameters(), lr=learning_rate)
    losses_train = []
    losses_val = []
    losses_val_avg = []
    for i in range(epochs):
        start_time = time.time()
        nnet.train()
        v, proba = nnet(train_boards.view(-1, 1, 6, 7))
        train_loss = full_loss_fn(v, proba, train_scores, train_moves, l2_lambda, nnet)

        losses_train.append(train_loss.item())

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            nnet.eval()
            v, proba = nnet(val_boards.view(-1, 1, 6, 7))
            val_loss = full_loss_fn(v, proba, val_scores, val_moves, l2_lambda, nnet)
            val_loss_avg = val_loss / len(val_boards)

            val_loss = val_loss.item()
            val_loss_avg = val_loss_avg.item()

            losses_val.append(val_loss)
            losses_val_avg.append(val_loss_avg)
            total_time = time.time() - start_time
            print(f"EPOCH::{i}  LOSS::{val_loss:.4f}  AVG_LOSS::{val_loss_avg:.4f}  TIME::{total_time:.4f}")

    training_path = utils.get_training_model_path(trial=trial, iteration=iteration)
    losses_train_path = get_losses_train_path(trial=trial, iteration=iteration)
    losses_val_path = get_losses_val_path(trial=trial, iteration=iteration)
    losses_val_avg_path = get_losses_val_avg_path(trial=trial, iteration=iteration)
    # Save information
    connect_net.save_model(nnet=nnet, path=training_path)
    with open(losses_train_path, "wb") as f:
        pickle.dump(obj=losses_train, file=f)
    with open(losses_val_path, "wb") as f:
        pickle.dump(obj=losses_val, file=f)
    with open(losses_val_avg_path, "wb") as f:
        pickle.dump(obj=losses_val_avg, file=f)


# Helper functions
def full_loss_fn(est_scores: torch.Tensor, est_probs: torch.Tensor,
                 target_scores: torch.Tensor, target_probs: torch.Tensor,
                 l2_lambda: float, nnet: torch.nn.Module) -> torch.Tensor:
    loss_scores = (est_scores - target_scores).square().sum()
    loss_probs = (target_probs * est_probs.log()).sum()
    loss_reg = l2_lambda * sum(p.square().sum() for p in nnet.parameters())
    loss_total = loss_scores - loss_probs + loss_reg
    return loss_total


def split_games(examples: List[List[utils.MoveInfo]],
                train_fraction: float) -> Tuple[List[List[utils.MoveInfo]], List[List[utils.MoveInfo]]]:
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


def get_losses_val_path(trial: str, iteration: int) -> str:
    training_folder = utils.get_training_folder(trial=trial, iteration=iteration)
    return os.path.join(training_folder, f"validation-losses-{iteration}.pickle")


def get_losses_val_avg_path(trial: str, iteration: int) -> str:
    training_folder = utils.get_training_folder(trial=trial, iteration=iteration)
    return os.path.join(training_folder, f"validation-avg-losses-{iteration}.pickle")


def get_losses_train_path(trial: str, iteration: int) -> str:
    training_folder = utils.get_training_folder(trial=trial, iteration=iteration)
    return os.path.join(training_folder, f"training-losses-{iteration}.pickle")


if __name__ == '__main__':
    trial_name = "conv402"
    train_class = connect_net.ConnectNet4
    for j in range(48, 101):
        if j == 1:
            # prev_path = "models/pretrain01.model"
            prev_path = None
        else:
            prev_path = utils.get_training_model_path(trial=trial_name, iteration=j-1)
        start_simulation(
            model=prev_path, klass=train_class, trial=trial_name, iteration=j,
            c=1, random_moves=21, num_mcts_sims=64, prob_move8=0.75,
            num_processes=6, num_episodes=512, batch_size=16,
        )
        train_model(
            trial=trial_name, klass=train_class, iteration=j,
            epochs=25, learning_rate=0.001, train_frac=0.85, l2_lambda=1e-4,
        )
