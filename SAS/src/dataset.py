import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SASDataset(Dataset):
    def __init__(self, cfg, summoner_df, match_df):
        self.cate_col = cfg['cate_cols']
        self.cont_col = cfg['cont_cols']
        self.max_seq_len = cfg['max_seq_len']    ### 10
        self.device = cfg['device']
        
        self.summoner_df, self.len, self.summoner_compression_table = self.prepare_summoner(summoner_df)
        self.match_df = self.prepare_match(match_df)


    def prepare_summoner(self, df):
        df = df.sort_values(by=['summoner_id']).reset_index(drop=True)

        temp = df.groupby(['summoner_id']).size()
        len = temp.__len__()

        summoner_compression_table = []

        ### summoner_compression_table
        start = 0
        for end in temp.values:
            summoner_compression_table.append((start, start+end))
            start += end

        summoner_compression_table = np.array(summoner_compression_table)

        return df, len, summoner_compression_table
    

    def prepare_match(self, df, k=2):
        df = df.sort_values(by=['match_id', 'team_key', 'position']).reset_index(drop=True)
        
        ### 게임 시간으로 continuous column들을 나눠줌
        cont_col_mod = [item for item in self.cont_col if item != "game_length_second"]
        for col in cont_col_mod:
            df[col] /= df['game_length_second']

        ### continuous column들을 정규화
        scaler = MinMaxScaler()
        df[self.cont_col] = pd.DataFrame(scaler.fit_transform(df[self.cont_col]))
        
        return df


    def __getitem__(self, idx):
        match_start, match_end = self.summoner_compression_table[idx]
        summoner_df = self.summoner_df.iloc[match_start: match_end]

        match_ids = summoner_df['match_id'].values - 1
        print(summoner_df['match_id'])
        posision = summoner_df['position_index'].values

        cate, cont = [], []
        for i in match_ids:
            ### 각 경기의 참여자는 항상 10명
            i = i * 10
            cate.append(self.match_df[i: i+10][self.cate_col].values)
            cont.append(self.match_df[i: i+10][self.cont_col].values)

        cate = torch.tensor(cate, dtype=torch.int)
        cont = torch.tensor(cont, dtype=torch.float)
        posision = torch.tensor(posision, dtype=torch.int)

        return cate, cont, posision


    def __len__(self):
        return len(self.summoner_compression_table)
