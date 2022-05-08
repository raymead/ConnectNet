import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch.multiprocessing
import numpy as np

import connect_net

# print(torch.multiprocessing.cpu_count())
print(torch.multiprocessing.get_all_sharing_strategies())
print(torch.multiprocessing.get_sharing_strategy())
print(torch.multiprocessing.get_all_start_methods())
print(torch.multiprocessing.get_start_method())
torch.multiprocessing.set_start_method('spawn', force=True)


def a_process(a_nnet, a_queue):
    print("here")
    test_state = np.zeros([6, 7], dtype=np.float32)
    test_state[5, 3] = 1
    test_state[5, 2] = -1
    test_state[4, 3] = 1
    test_state[3, 3] = 1
    # test_state[2,3] = 1
    test_state[0, 0] = 1
    test_state[1, 0] = -1
    test_state[2, 0] = 1
    test_state[3, 0] = -1
    test_state[4, 0] = 1
    test_state[5, 0] = -1
    print("here2")
    print(test_state)
    test_state_tensor = torch.Tensor(test_state).reshape(1, 1, 6, 7)
    test_state_tensor.share_memory()
    print(test_state_tensor)
    print("here2.5")
    # print(a_nnet)
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
    with torch.no_grad():
        ctx = torch.multiprocessing.get_context("spawn")
        nnet = connect_net.ConnectNet()
        nnet.share_memory()
        q = ctx.Queue()
        num_processes = 4
        processes = []
        for i in range(num_processes):
            p = ctx.Process(target=a_process, args=(nnet, q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()

        while not q.empty():
            print(q.get())

        q.close()

