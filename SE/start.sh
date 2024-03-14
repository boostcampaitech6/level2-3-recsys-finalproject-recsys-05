#!/bin/bash

# 원격 서버 접속 정보
SERVERS=("bgw-server" "ngo-server" "lsg-server" "jsj-server")

# SSH를 사용하여 원격 서버에 접속하고 명령어 실행
for i in $(seq 0 $((${#SERVERS[@]} - 1))); do
	COMMANDS='cd gpu-cluster;
	source multi-gpu/bin/activate;
	cd level2-3-recsys-finalproject-recsys-05;
	git checkout cossim;
	git pull;
	cd SE;
	nohup torchrun --nnodes=${#SERVERS[@]} --nproc_per_node=1 --node_rank=$i --master_addr=10.0.2.7 --master_port=20000 run.py &;'
	
	ssh ${SERVERS[i]} "$(echo -e $COMMANDS)"
done
