import glob
import os
import random
from typing import Tuple, Optional, List

import torch


class MoveInfo:
    def __init__(self, state: torch.Tensor, rep: str, move: int,
                 action: int,  pi_val: Optional[torch.Tensor] = None,
                 ge: Optional[int] = None):
        self.state = state
        self.rep = rep
        self.move = move

        self.action = action
        self.pi_val = pi_val

        self.ge = ge

    def set_game_ended(self, ge: int):
        self.ge = ge


def split_train_val_test_values(total_count: int, train_fraction: float, val_fraction: float) -> Tuple[int, int, int]:
    train_num = int(total_count * train_fraction)
    val_num = int(total_count * val_fraction)
    test_num = total_count - train_num - val_num
    return train_num, val_num, test_num


def split_train_val_values(total_count: int, train_fraction: float) -> Tuple[int, int]:
    train_num = int(total_count * train_fraction)
    val_num = total_count - train_num
    return train_num, val_num


def get_shuffled_list(size: int):
    shuffled_indexes = list(range(size))
    random.shuffle(shuffled_indexes)
    return shuffled_indexes


def get_data_folder() -> str:
    return "data"


def get_trial_folder(trial: str) -> str:
    folder = os.path.join(get_data_folder(), f"trial{trial}")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def get_simulation_folder(trial: str, iteration: int) -> str:
    folder = os.path.join(get_trial_folder(trial=trial), f"simulation{iteration}")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def get_training_folder(trial: str, iteration: int) -> str:
    folder = os.path.join(get_trial_folder(trial=trial), f"training{iteration}")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def get_results_folder(trial: str) -> str:
    folder = os.path.join(get_trial_folder(trial=trial), f"results")
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return folder


def get_training_folders(trial: str) -> List[str]:
    return glob.glob(os.path.join(get_trial_folder(trial=trial), f"training*"))


def get_model_path(folder: str, iteration: int) -> str:
    return os.path.join(folder, f"model-{iteration}.model")


def get_simulation_model_path(trial: str, iteration: int) -> str:
    return get_model_path(folder=get_simulation_folder(trial=trial, iteration=iteration), iteration=iteration)


def get_training_model_path(trial: str, iteration: int) -> str:
    return get_model_path(folder=get_training_folder(trial=trial, iteration=iteration), iteration=iteration)


def get_move8_boards_path():
    return os.path.join(get_data_folder(), "move8_boards.npy")
