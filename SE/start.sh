#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hidden_size) hidden_size="$2"; shift ;;
        --emb_size) emb_size="$2"; shift ;;
        --dropout) dropout="$2"; shift ;;
        --lr) lr="$2"; shift ;;
        *) echo "잘못된 옵션: $1"; exit 1 ;;
    esac
    shift
done

# 원격 서버 접속 정보
SERVERS=("bgw-server" "ngo-server" "lsg-server" "jsj-server")
# SERVERS=("bgw-server" "ngo-server" "lsg-server")

# SSH를 사용하여 원격 서버에 접속하고 명령어 실행
for i in $(seq 0 $((${#SERVERS[@]} - 1))); do
	COMMANDS="cd gpu-cluster;
	source multi-gpu/bin/activate;
	cd level2-3-recsys-finalproject-recsys-05;
	git checkout cossim;
	git pull;
	cd SE;
	pkill -9 -u bgw torchrun;
	nohup torchrun --nnodes=${#SERVERS[@]} --nproc_per_node=1 --node_rank=$i --master_addr=10.0.2.7 --master_port=20000 \
	run.py --hidden_size=$hidden_size --emb_size=$emb_size --dropout=$dropout --lr=$lr > nohup.out 2>&1 &
	echo done;
	"
	# --hidden_size=$hidden_size --emb_size=$emb_size --dropout=$dropout --lr=$lr
	
	scp -r ../../data/ ${SERVERS[i]}:gpu-cluster/
	ssh -n ${SERVERS[i]} "$(echo -e $COMMANDS)"
done
