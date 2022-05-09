import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# import multiprocessing

import numpy as np
import torch
import torch.multiprocessing

import connect_net


def a_process(a_nnet):
    test_state = np.zeros([6, 7], dtype=np.float32)
    test_state[5, 3] = 1
    test_state[5, 2] = -1
    test_state[4, 3] = 1
    test_state[3, 3] = 1
    test_state[0, 0] = 1
    test_state[1, 0] = -1
    test_state[2, 0] = 1
    test_state[3, 0] = -1
    test_state[4, 0] = 1
    test_state[5, 0] = -1

    test_state_tensor = torch.Tensor(test_state).view(1, 1, 6, 7)
    print("here2.5")
    print(test_state_tensor)
    try:
        v, a_prob = a_nnet(test_state_tensor)
        print(v)
    except Exception as e:
        print(e)
        raise e
    print("here3")
    print("here4")
    return a_prob


if __name__ == '__main__':
    nnet = connect_net.ConnectNet()
    nnet.share_memory()
    with torch.multiprocessing.Pool(processes=4) as pool:
        pool.starmap(a_process, [(nnet,), (nnet,), (nnet,), (nnet,)])