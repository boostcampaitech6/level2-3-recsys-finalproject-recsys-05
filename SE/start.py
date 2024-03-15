import argparse
from doctest import master
from time import sleep
import paramiko
from scp import SCPClient


# 입력 인자 처리
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", type=str, required=True)
parser.add_argument("--emb_size", type=str, required=True)
parser.add_argument("--dropout", type=str, required=True)
parser.add_argument("--lr", type=str, required=True)
args = parser.parse_args()

# 원격 서버 접속 정보
master = "bgw-server"
worker = ["ngo-server", "lsg-server", "jsj-server"]

# SSH 접속 설정
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

for i, server in enumerate(worker, start=1):
    try:
        # SSH 접속
        ssh.connect(server)
        scp = SCPClient(ssh.get_transport())

        # 데이터 디렉토리 전송
        scp.put('../../../data/', recursive=True, remote_path='gpu-cluster/')

        # 원격 명령어 실행
        commands = f"""
        cd gpu-cluster;
        source multi-gpu/bin/activate;
        cd level2-3-recsys-finalproject-recsys-05;
        git checkout cossim;
        git pull;
        cd SE;
        nohup torchrun --nnodes={len(worker) + 1} --nproc_per_node=1 --node_rank={i} --master_addr=10.0.2.7 --master_port=20000 \
        run.py --hidden_size={args.hidden_size} --emb_size={args.emb_size} --dropout={args.dropout} --lr={args.lr} > nohup.out 2>&1 &
        """
        stdin, stdout, stderr = ssh.exec_command(commands)

    finally:
        if ssh:
            ssh.close()

try:
    # SSH 접속
    ssh.connect(master)
    scp = SCPClient(ssh.get_transport())

    # 데이터 디렉토리 전송
    scp.put('../../../data/', recursive=True, remote_path='gpu-cluster/data')

    # 원격 명령어 실행
    commands = f"""
    cd gpu-cluster;
    source multi-gpu/bin/activate;
    cd level2-3-recsys-finalproject-recsys-05;
    git checkout cossim;
    git pull;
    cd SE;
    torchrun --nnodes={len(worker) + 1} --nproc_per_node=1 --node_rank=0 --master_addr=10.0.2.7 --master_port=20000 \
    run.py --hidden_size={args.hidden_size} --emb_size={args.emb_size} --dropout={args.dropout} --lr={args.lr} &
    """
    stdin, stdout, stderr = ssh.exec_command(commands)
    print(stdout.read().decode())
    print(stderr.read().decode())

finally:
    if ssh:
        ssh.close()


sleep(10)
for server in [master] + worker:
    try:
        # SSH 접속
        ssh.connect(server)

        # 원격 명령어 실행
        commands = f"""
        pkill -15 -u bgw torchrun;
        """
        stdin, stdout, stderr = ssh.exec_command(commands)

    finally:
        if ssh:
            ssh.close()

sleep(10)
for server in [master] + worker:
    try:
        # SSH 접속
        ssh.connect(server)

        # 원격 명령어 실행
        commands = f"""
        pkill -9 -u bgw torchrun;
        """
        stdin, stdout, stderr = ssh.exec_command(commands)

    finally:
        if ssh:
            ssh.close()