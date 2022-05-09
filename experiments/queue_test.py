import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.multiprocessing
import numpy as np

import connect_net


def a_process(a_nnet, a_queue):
    a_queue.put("atest")
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
    a_queue.put("test")
    a_queue.put("test2")
    a_queue.put([test_state_tensor, a_prob, v])
    print("here4")


if __name__ == '__main__':
    print(torch.multiprocessing.cpu_count())
    print(torch.multiprocessing.get_all_sharing_strategies())
    print(torch.multiprocessing.get_sharing_strategy())
    print(torch.multiprocessing.get_all_start_methods())
    print(torch.multiprocessing.get_start_method())
    torch.multiprocessing.set_start_method("spawn", force=True)

    with torch.no_grad():
        # ctx = torch.multiprocessing.get_context("ctx")
        nnet = connect_net.ConnectNet()
        nnet.share_memory()
        q = torch.multiprocessing.Queue()
        num_processes = 4
        processes = []
        for i in range(num_processes):
            p = torch.multiprocessing.Process(target=a_process, args=(nnet, q,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while not q.empty():
            print(q.get())

        q.close()


