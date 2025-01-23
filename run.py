
import argparse
import os


def get_experiment_functions():
    return {
        i: globals()[f"exp{i}"]
        for i in range(1, 101)  # 최대 100개의 실험 함수 등록 가능
        if f"exp{i}" in globals()
    }

from experiments import *
def main():
    # Argument parser 설정
    root = './dataset'
    parser = argparse.ArgumentParser(description="Experiment Runner with Training Epochs")
    parser.add_argument("--exp", type=int, required=True, help="Experiment number to run (e.g., 1, 2, 3)")
    parser.add_argument("--train_epoch", type=int, default=10, help="Number of training epochs (default: 10)")

    # 인자 파싱
    args = parser.parse_args()

    # Experiment 실행
    exp_functions = get_experiment_functions()

    if args.exp in exp_functions:
        print(f"Running experiment {args.exp} with {args.train_epoch} training epochs...")
        exp_functions[args.exp](root,args.train_epoch)  # 해당 실험 함수 실행
    else:
        print(f"Experiment {args.exp} is not defined. Please choose from {list(exp_functions.keys())}.")

if __name__ == '__main__':
    root = './dataset'

    # plt.plot(array1[7])
    # print(array2[7])
    # print(array3[7])
    # plt.show()q
    # generate_datas(root='./dataset')
    # main()
    # exp2(root,epoch=3000)
    exp4(root, epoch=500)
    # exp5(root, epoch=2000)