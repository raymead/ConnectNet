import os
from typing import Optional

import torch.multiprocessing as mp

import connect_net
import mcts


def start_mcts(
        model: Optional[str], trial: str, iteration: int,
        num_processes: int, num_episodes: int, num_mcts_sims: int) -> None:
    # Setup save folder
    folder = f"data/trial{trial}/iteration{iteration}"
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Load model
    if model is None:
        nnet = connect_net.ConnectNet()
    else:
        nnet = connect_net.load_model(model)
    nnet.share_memory()
    connect_net.save_model(nnet, os.path.join(folder, f"model-{iteration}.model"))

    processes = []
    for i in range(num_processes):
        p = mp.Process(target=mcts.find_policy_examples, args=(nnet, folder, iteration, i, num_episodes, num_mcts_sims))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    start_mcts(
        model="models/pretrain01-iter02.model", trial="pretrain01", iteration=3,
        num_processes=6, num_episodes=256, num_mcts_sims=256,
    )
