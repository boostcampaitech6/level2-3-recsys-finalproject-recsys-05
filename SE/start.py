import subprocess

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_size", type=int, help="")
    parser.add_argument("--hidden_size", type=int, help="")
    parser.add_argument("--lr", type=float, help="")
    parser.add_argument("--dropout", type=float, help="")

    args = parser.parse_args()

    return args

args = parse_args()

result = subprocess.run(['./your-script.sh', f'--emb_size={args.emb_size}', f'--hidden_size={args.hidden_size}', f'--lr={args.lr}',  f'--dropout={args.dropout}'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)