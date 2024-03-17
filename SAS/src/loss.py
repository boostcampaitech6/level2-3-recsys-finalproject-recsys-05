import torch
import torch.nn as nn



def batch_cosine_similarity(embedding_matrix, vectors):
    """
    embedding_matrix와 vectors 간의 코사인 유사도를 배치로 계산합니다.
    
    Args:
    - embedding_matrix (torch.Tensor): 임베딩 테이블 행렬, 크기는 (num_embeddings, embedding_dim)
    - vectors (torch.Tensor): 비교할 벡터들의 배치, 크기는 (num_vectors, embedding_dim)
    
    Returns:
    - torch.Tensor: 코사인 유사도 행렬, 크기는 (num_vectors, num_embeddings)
    """
    # 내적을 계산하기 전에 두 행렬을 정규화합니다.
    embedding_matrix_norm = embedding_matrix / embedding_matrix.norm(dim=-1, keepdim=True)
    vectors_norm = vectors / vectors.norm(dim=-1, keepdim=True)
    
    # 정규화된 행렬의 내적을 계산하여 코사인 유사도를 얻습니다.
    cosine_similarity = torch.matmul(vectors_norm, embedding_matrix_norm.t())
    
    return cosine_similarity


class cos_loss(nn.Module):
    def __init__(self):
        super(cos_loss, self).__init__()

    def forward(self, input):
        loss = batch_cosine_similarity(input, input[])
        
        return loss
    