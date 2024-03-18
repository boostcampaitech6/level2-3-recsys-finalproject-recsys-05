import torch
import torch.nn as nn
import torch.nn.functional as F



def batch_cosine_similarity(embeddings):
    """
    embedding_matrix와 vectors 간의 코사인 유사도를 배치로 계산합니다.
    
    Args:
    - embedding_matrix (torch.Tensor): 임베딩 테이블 행렬, 크기는 (num_embeddings, embedding_dim)
    - vectors (torch.Tensor): 비교할 벡터들의 배치, 크기는 (num_vectors, embedding_dim)
    
    Returns:
    - torch.Tensor: 코사인 유사도 행렬, 크기는 (num_vectors, num_embeddings)
    """
    batch_size, seq_len, _ = embeddings.size()

    # 각 벡터를 정규화합니다.
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)

    # 각 배치에 대한 코사인 유사도 계산
    cos_sim = torch.zeros((batch_size, seq_len, seq_len), device=embeddings.device)
    for i in range(batch_size):
        cos_sim[i] = torch.matmul(embeddings_norm[i], embeddings_norm[i].transpose(0, 1))
        
    return cos_sim


class CosLoss(nn.Module):
    def __init__(self):
        super(CosLoss, self).__init__()

    def forward(self, input):
        cos_sim = batch_cosine_similarity(input)

        # 자기 자신과의 유사도 제외
        eye = torch.eye(cos_sim.size(1), device=cos_sim.device).unsqueeze(0)
        cos_sim = cos_sim * (1. - eye)

        # 평균 손실 계산
        loss = (1 - cos_sim).sum() / (cos_sim.size(0) * cos_sim.size(1) * (cos_sim.size(2) - 1))
        return loss
    

### 위험!!!! loss가 0이 됩니다.
class StdLoss(nn.Module):
    def __init__(self):
        super(StdLoss, self).__init__()

    def forward(self, input):
        ### 위험! dim=0일 때 표준편차가 0이 되면 loss가 0이 됩니다.
        loss = torch.sum(torch.std(input + 1e-6, dim=0))
        
        return loss