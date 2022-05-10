import sys

import torch.multiprocessing as mp

sys.path.append("/Users/raymond/code/FinalProject563")
import connect_net
import mcts


def start_mcts(num_processes: int, num_episodes: int, num_mcts_sims: int) -> None:
    nnet = connect_net.ConnectNet()
    nnet.share_memory()
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=mcts.find_policy_examples, args=(nnet, 1, i, num_episodes, num_mcts_sims))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    start_mcts(num_processes=4, num_episodes=8, num_mcts_sims=128)
