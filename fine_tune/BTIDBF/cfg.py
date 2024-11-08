import argparse
import os

def get_root():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(root_dir, "checkpoints")
    results = os.path.join(root_dir, "results")
    if not os.path.exists(ckpt):
        os.mkdir(ckpt)
    if not os.path.exists(results):
        os.mkdir(results)
    return root_dir


def get_arguments():
    parser = argparse.ArgumentParser()

    #load box
    parser.add_argument("--dataset", type=str, default="cifar")
    parser.add_argument("--tlabel", type=int, default=5)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--attack", type=str, default="wanet")
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--size", type=int, default=32) 
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--attack_type", type=str, default="all2one")
    parser.add_argument("--root", type=str, default=get_root())
    parser.add_argument("--model_path", type=str)

    #pretrain
    parser.add_argument("--preround", type=int, default=50)
    parser.add_argument("--gen_lr", type=float, default=1e-3)

    #bti-dbf
    parser.add_argument("--mround", type=int, default=20)
    parser.add_argument("--uround", type=int, default=30)
    parser.add_argument("--norm_bound", type=float, default=0.3)
    parser.add_argument("--feat_bound", type=float, default=3)

    #defense
    parser.add_argument("--nround", type=int, default=5)
    parser.add_argument("--ul_round", type=int, default=30, help="iteration of dpi-dbf (u)")
    parser.add_argument("--pur_round", type=int, default=30, help="iteration of dpi-dbf (p)")
    parser.add_argument("--pur_norm_bound", type=float, default=0.05)
    parser.add_argument("--earlystop", type=bool, default=False)
    parser.add_argument("--use_sam", action='store_true', default=False)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--attack_name", type=str, default='badnet')
    parser.add_argument("--pratio", type=float, default=0.05)
    parser.add_argument("--pre", action='store_true')


    return parser
