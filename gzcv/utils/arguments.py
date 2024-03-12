import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_test", action="store_true")
    parser.add_argument("--is_train", action="store_true")
    parser.add_argument("--cfgs", nargs="*")
    parser.add_argument("--model", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--resume", type=str, help="path/to/your/*.pth")
    parser.add_argument("--level", type=str, default="info")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("-m", "--message", type=str, help="note for experiments")
    parser.add_argument("--run-k-fold", action="store_true")
    parser.add_argument("--mpii-k-fold", action="store_true")
    parser.add_argument("--k-fold-index", type=int, default=-1)
    parser.add_argument("--only-first-fold", action="store_true")

    args, unknown = parser.parse_known_args()
    print("Unrecognized args = ", unknown)
    return args
