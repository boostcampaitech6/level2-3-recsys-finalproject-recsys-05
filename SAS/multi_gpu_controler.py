import argparse
from doctest import master
import time
import paramiko
from scp import SCPClient
import subprocess


def parse_args():
    # 입력 인자 처리
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action='store_true', default=False)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--emb_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()

# 원격 서버 접속 정보
workers = ["ngo-server", "lsg-server", "jsj-server"]
# workers = []


def connect_and_execute(server, commands, copy=False):
    # 새로운 SSHClient 인스턴스 생성
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(server)
            with SCPClient(ssh.get_transport()) as scp:
                # 필요한 작업 수행...
                if copy:
                    scp.put('../../shared_data', recursive=True, remote_path='~/SAS/shared_data/')

                stdin, stdout, stderr = ssh.exec_command(commands)
        except paramiko.ssh_exception.NoValidConnectionsError as e:
            print(f"Connection failed for {server}: {e}")
        
        return stdout.read().decode('utf-8')


def start():
    for i, worker in enumerate(workers, start=1):
        commands = f"""
        cd SAS;
        source venv/SAS/bin/activate;
        cd level2-3-recsys-finalproject-recsys-05;
        git checkout SAS;
        git pull;
        cd SAS;
        nohup torchrun --nnodes={len(workers) + 1} --nproc_per_node=1 --node_rank={i} --master_addr=10.0.2.7 --master_port=20000 \
            run.py --hidden_size={args.hidden_size} --emb_size={args.emb_size} --dropout={args.dropout} --lr={args.lr} --batch_size={args.batch_size} \
            --n_head={args.n_head} > nohup.out 2>&1 &
        """
        output = connect_and_execute(worker, commands, copy=True)
        # print(f"Output for {worker}: {output}")


    command = f"torchrun --nnodes={len(workers) + 1} --nproc_per_node=1 --node_rank=0 --master_addr=10.0.2.7 --master_port=20000 \
            run.py --hidden_size={args.hidden_size} --emb_size={args.emb_size} --dropout={args.dropout} --lr={args.lr} --batch_size={args.batch_size} \
            --n_head={args.n_head} > nohup.out 2>&1 &"

    output = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(f"Output for bgw-server: {output}")

def stop():
    time.sleep(10)
    for worker in ["bgw-server"] + workers:
        commands = f"""
        pkill -15 -u bgw torchrun;
        """
        connect_and_execute(worker, commands)


    time.sleep(10)
    for server in ["bgw-server"] + workers:
        commands = f"""
        pkill -9 -u bgw torchrun;
        """
        connect_and_execute(worker, commands)


if __name__ == "__main__":
    args = parse_args()

    if not args.stop:
        print("start")
        start()
        print("stop")
        stop()
    else:
        print("stop")
        stop()
