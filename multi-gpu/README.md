
## 실행 방법
```
$ torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr=123.456.789.123 --master_port=23456 multi_gpu.py
```

* nnodes : 총 노드 개수  
* nproc_per_node : 노드당 프로세스(GPU) 개수  
* node_rank : 노드 번호. 0 번으로 실행된 노드가 마스터 노드  
* master_addr : 마스터 노드 주소  
* master_port : 마스터 노드 포트  

## 참조  
https://csm-kr.tistory.com/47  
https://csm-kr.tistory.com/89