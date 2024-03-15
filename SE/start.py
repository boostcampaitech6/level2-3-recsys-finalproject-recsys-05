import argparse
from doctest import master
import time
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
servers = ["bgw-server", "ngo-server", "lsg-server", "jsj-server"]


def connect_and_execute(server, commands, copy=False):
    # 새로운 SSHClient 인스턴스 생성
    with paramiko.SSHClient() as ssh:
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(server)
            with SCPClient(ssh.get_transport()) as scp:
                # 필요한 작업 수행...
                if copy:
                    scp.put('../../../data/', recursive=True, remote_path='gpu-cluster/')

                stdin, stdout, stderr = ssh.exec_command(commands)
                # 결과 처리...
        except paramiko.ssh_exception.NoValidConnectionsError as e:
            print(f"Connection failed for {server}: {e}")


# 사용 예
for i, server in enumerate(servers):
    commands = f"""
    cd gpu-cluster;
    source multi-gpu/bin/activate;
    cd level2-3-recsys-finalproject-recsys-05;
    git checkout cossim;
    git pull;
    cd SE;
    nohup torchrun --nnodes={len(servers)} --nproc_per_node=1 --node_rank={i} --master_addr=10.0.2.7 --master_port=20000 \
    run.py --hidden_size={args.hidden_size} --emb_size={args.emb_size} --dropout={args.dropout} --lr={args.lr} > nohup.out 2>&1 &
    """
    connect_and_execute(server, commands, copy=True)



time.sleep(10)
for server in servers:
    commands = f"""
    pkill -15 -u bgw torchrun;
    """
    connect_and_execute(server, commands)


time.sleep(10)
for server in servers:
    commands = f"""
    pkill -9 -u bgw torchrun;
    """
    connect_and_execute(server, commands)