import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
from models.SAS.model import SASModel
from models.SAS.cfg import CFG


cfg = CFG('models/SAS/config.yaml')
model = SASModel(cfg).to(cfg['device'])
model = torch.load


def cont_norm(df):
    ### 게임 시간으로 continuous column들을 나눠줌
    cont_col_mod = [item for item in cfg['cont_cols'] if item != "gameDuration" and item != "visionScore" and item != "summonerLevel"]
    for col in cont_col_mod:
        df[col] /= df['gameDuration']

    scaler = MinMaxScaler()
    df[cfg['cont_cols']] = pd.DataFrame(scaler.fit_transform(df[cfg['cont_cols']]))

    return df


def cate_change_unique(df):
    ### 각 값들을 고유 값으로 변경
    for cate in cfg['indexing_cols'].keys():
        if cate == 'other':
            for col in cfg['indexing_cols'][cate]:
                df[col] = df[col].apply(lambda x: col + '.' + str(x))

        else:
            for col in cfg['indexing_cols'][cate]:
                df[col] = df[col].apply(lambda x: cate + '.' + str(x))

    return df


def cate_mapping(df):
    cate_to_index = pickle.load(open('models/SAS/cate_to_index.pkl', 'rb'))

    for col in cfg['cate_cols']:
        df[col] = df[col].map(cate_to_index)

    return df


def cosine_similarity(vector_a, vectors_b):
    # 벡터의 크기(놈)를 계산
    norm_a = vector_a.norm(p=2)
    norms_b = vectors_b.norm(p=2, dim=1)

    # 벡터 a와 벡터 b들의 내적을 계산
    dot_product = torch.matmul(vectors_b, vector_a)

    # 코사인 유사도 계산
    cossim = dot_product / (norm_a * norms_b)

    return cossim


def inference(matches):
    df = pd.DataFrame(matches)

    df = cont_norm(df)
    df = cate_change_unique(df)
    df = cate_mapping(df)
    
    ### 100개의 후보 유저, 각 유저의 10개의 매치데이터, 각 매치에 10개의 참여자, feature 차원으로 tensor를 변경
    cate = torch.tensor(df[cfg['cate_cols']].values, dtype=torch.int64).view(100, 10, 10, -1)
    cont = torch.tensor(df[cfg['cont_cols']].values, dtype=torch.float32).view(100, 10, 10, -1)
    pos = torch.tensor(df['position'].values, dtype=torch.int64).view(100, 10)

    model = SASModel(cfg).to(cfg['device'])
    output = model(cate, cont, pos)

    cossim = cosine_similarity(output[0], output[1:])

    df = pd.DataFrame(cossim.cpu().detach().numpy(), columns=['similarity'])
    sorted_index = df.sort_values(by='similarity').index - 1

    return sorted_index