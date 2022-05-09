import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import time
import multiprocessing as mp
import torch

import connect_net

done = mp.Event()


def extractor_worker(nnet, done_queue):
    a_tensor = torch.zeros((6, 7)).view(1, 1, 6, 7)
    a_tensor[0,0,5, 3] = 1
    print("here")
    done_queue.put(a_tensor)
    print("here2")
    done_queue.put(None)
    print("here3")
    print(a_tensor)
    v, proba = nnet(a_tensor)
    print(v, proba)
    print("here4")
    print(v)
    time.sleep(3)



if __name__ == '__main__':
    producers = []
    nnet = connect_net.ConnectNet()
    # nnet.share_memory()
    done_queue = mp.Queue()
    for i in range(0, 1):
        process = mp.Process(target=extractor_worker,
                             args=(nnet, done_queue,))
        process.start()
        producers.append(process)

    result_arrays = []
    nb_ended_workers = 0
    while nb_ended_workers != 1:
        worker_result = done_queue.get()
        if worker_result is None:
            nb_ended_workers += 1
        else:
            result_arrays.append(worker_result)

    done.set()
