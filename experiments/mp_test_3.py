import torch
import torch.multiprocessing
import numpy as np

def a_process(num: int):
    print(a_process)

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

    test_state_tensor = torch.Tensor(test_state)
    print(test_state_tensor)


if __name__ == '__main__':
    with torch.multiprocessing.Pool(processes=4) as pool:
        pool.map(a_process, [1, 2, 3, 4])
