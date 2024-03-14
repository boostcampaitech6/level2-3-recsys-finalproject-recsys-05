#!/bin/bash

# 원격 서버 접속 정보
SERVERS=("bgw-server" "ngo-server" "lsg-server" "jsj-server")
# SERVERS=("bgw-server" "ngo-server" "lsg-server")

# SSH를 사용하여 원격 서버에 접속하고 명령어 실행
for i in $(seq 0 $((${#SERVERS[@]} - 1))); do
	COMMANDS="
	pkill -15 -u bgw torchrun;
    "

	ssh -n ${SERVERS[i]} "$(echo -e $COMMANDS)"
done

sleep 10;

for i in $(seq 0 $((${#SERVERS[@]} - 1))); do
	COMMANDS="
	pkill -9 -u bgw torchrun;
    "

	ssh -n ${SERVERS[i]} "$(echo -e $COMMANDS)"
done