import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import torch
import torch.multiprocessing as mp

import connect_net


def aproc(rank, path):
    print(os.getcwd())
    print(rank, path)
    print("test")
    if rank == 1:
        a_tensor = torch.zeros((6, 7)).view(1, 1, 6, 7)
        print("load model")
        try:
             nnet = connect_net.load_model(path)
        except Exception as e:
            print(e)
            raise e
        print("done model")
        print(a_tensor)
        try:
            v, proba = nnet(a_tensor)
            # v, proba =
        except Exception as e:
            print(e)
            raise e
        print(v)
        print(proba)
    else:
        print("test")


if __name__ == '__main__':
    # nnet = connect_net.ConnectNet()
    # nnet.share_memory()
    mp.set_start_method("spawn", force=True)
    mp.spawn(aproc, args=("/Users/raymond/code/FinalProject563/model_save_01.model",), nprocs=2, join=True)
