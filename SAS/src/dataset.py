import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SASDataset(Dataset):
    def __init__(self, cfg, df, output_cate_col=None):
        self.cate_col = cfg['cate_cols']
        if output_cate_col is None:
            self.output_cate_col = [x for x in self.cate_col if x != 'summonerId' and x != 'matchId' and x != 'position']
        else:
            self.output_cate_col = output_cate_col
        
        self.cont_col = cfg['cont_cols']

        self.df_by_match, self.df_by_summoner, self.compressed_index_table_by_summoner = self.prepare_df(df)
        self.len = len(self.compressed_index_table_by_summoner)


    def prepare_df(self, df):
        df_by_match = self.prepare_match(df)
        df_by_summoner, compressed_index_table_by_summoner = self.prepare_summoner(df)        

        return df_by_match, df_by_summoner, compressed_index_table_by_summoner
    

    def prepare_summoner(self, df):
        df = df[['summonerId', 'matchId', 'position']].sort_values(by=['summonerId']).reset_index(drop=True)

        temp = df.groupby('summonerId').size()
        compressed_index_table_by_summoner= []
        start = 0
        for end, bool in zip(temp.values, temp >= 10):
            if bool:
                ### 처음 10개 경기만 사용하기 위해 start + 10
                compressed_index_table_by_summoner.append((start, start + 10))
            start += end

        compressed_index_table_by_summoner = np.array(compressed_index_table_by_summoner)

        return df, compressed_index_table_by_summoner
    

    def prepare_match(self, df):
        df = df.sort_values(by=['matchId', 'teamId', 'individualPosition']).reset_index(drop=True)
        
        ### 게임 시간으로 continuous column들을 나눠줌
        cont_col_mod = [item for item in self.cont_col if item != "gameDuration" and item != "visionScore" and item != "summonerLevel"]
        for col in cont_col_mod:
            df[col] /= df['gameDuration']

        ### continuous column들을 정규화
        scaler = MinMaxScaler()
        df[self.cont_col] = pd.DataFrame(scaler.fit_transform(df[self.cont_col]))
        
        return df


    def __getitem__(self, idx):
        summoner_start, summoner_end = self.compressed_index_table_by_summoner[idx]
        summoner_df = self.df_by_summoner.iloc[summoner_start: summoner_end]

        match_ids = summoner_df['matchId'].values - 1
        posision = summoner_df['position'].values

        cate, cont = [], []
        for i in match_ids:
            ### 각 매치 match_id는 총 10개의 행을 가지므로 인덱스를 유추하기 위해 i * 10
            i = i * 10
            ### 각 경기의 참여자는 항상 10명 임으로 i+10
            cate.append(self.df_by_match[i: i+10][self.output_cate_col].values)
            cont.append(self.df_by_match[i: i+10][self.cont_col].values)

        cate = np.array(cate)
        cont = np.array(cont)

        cate = torch.tensor(cate, dtype=torch.int)
        cont = torch.tensor(cont, dtype=torch.float)
        posision = torch.tensor(posision, dtype=torch.int64)

        return cate, cont, posision


    def __len__(self):
        return self.len
