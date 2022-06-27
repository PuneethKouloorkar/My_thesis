from argparse import ArgumentParser

args = ArgumentParser()
args.add_argument('-a', '--archi', type=str, default='ResNet18')
final_args = args.parse_args()

def junk_guru(archi):
    for val in range(1, 5000):
        i = 0

    print("0")


junk_guru(final_args.archi)