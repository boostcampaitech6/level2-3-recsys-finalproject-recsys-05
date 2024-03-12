import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler



class SimilarityDataset(Dataset):
    def __init__(self, cfg, summoner_df, match_df):
        self.cate_col = cfg['cate_col']
        self.cont_col = cfg['cont_col']
        self.max_seq_len = cfg['max_seq_len']    ### 10
        self.device = cfg['device']
        
        self.summoner_df, self.summoner_compression_table = self.prepare_summoner(summoner_df)
        self.match_df, self.len, self.combinations, self.match_compression_table = self.prepare_match(match_df)

        self.correction_summoner_idx = self.summoner_df['summoner_id'].min()


    def prepare_summoner(self, df):
        df = df.sort_values(by='summoner_id')
        summoner_compression_table = []

        ### summoner_compression_table
        start = 0
        for end in df['summoner_id'].value_counts().sort_index().values:
            summoner_compression_table.append((start, start+end))
            start += end

        summoner_compression_table = np.array(summoner_compression_table)

        ### 게임 시간을 기준으로 나누어줌
        cont_col_mod = [item for item in self.cont_col if item != "game_length_second"]
        for col in cont_col_mod:
            df[col] /= df['game_length_second']

        ### 정규화
        scaler = MinMaxScaler()
        df[self.cont_col] = pd.DataFrame(scaler.fit_transform(df[self.cont_col]))

        return df, summoner_compression_table
    

    def prepare_match(self, df, k=2):
        df = df.sort_values(by='match_id')
        temp = df.groupby(['match_id', 'team_key']).size()

        len = temp.__len__()

        combinations = [[list(element) for element in itertools.combinations(range(count), 2)] for count in temp.values]

        match_compression_table = []
        start = 0
        for end in temp.values:
            match_compression_table.append((start, start+end))
            start += end

        match_compression_table = np.array(match_compression_table)
        
        return df, len, combinations, match_compression_table


    '''
    idx를 받으면 먼저 match 데이터에서 해당하는 match_id와 team_key에 맞는 데이터를 가져온다.
    해당하는 매치 데이터는 아군 최대 5개 최소 2개의 summoner_id와 1개의 동일한 result를 가지고 있다.
    매치데이터에서 summoner_id 2개를 뽑는 경우의 수 A, B를 구하고, 각 경우의 수에 맞게 summoner 데이터에서 해당하는 데이터를 가져온다.
    가져온 데이터는 cate, cont로 나누어져 있으며, cate는 embedding을 위해 int로, cont는 float로 변환한다.
    변환이 끝난 데이터는 A, B로 나누어 출력한다.
    '''
    def __getitem__(self, idx):
        ### 1개의 매치에 1개의 블루 or 레드팀의 5~2명의 팀원이 존재
        ### 팀원들 존재하는 iloc 위치를 가져옴
        match_start, match_end = self.match_compression_table[idx]
        match_df = self.match_df.iloc[match_start: match_end]

        summoner_compression_idx = match_df['summoner_id'].values - self.correction_summoner_idx
        summoner_idx = self.summoner_compression_table[summoner_compression_idx]

        summoner_table_cate = []
        summoner_table_cont = []
        for summoner_start, summoner_end in summoner_idx:
            summoner_table_cate.append(self.summoner_df[summoner_start:summoner_end][self.cate_col].values)
            summoner_table_cont.append(self.summoner_df[summoner_start:summoner_end][self.cont_col].values)

        A_cate_data, A_cont_data, B_cate_data, B_cont_data = [], [], [], []
        combination = self.combinations[idx]
        for a, b in combination:
            A_cate = torch.zeros(self.max_seq_len, len(self.cate_col), dtype=torch.int)
            A_cate[:len(summoner_table_cate[a])] = torch.tensor(summoner_table_cate[a], dtype=torch.int)
            A_cate_data.append(A_cate)

            A_cont = torch.zeros(self.max_seq_len, len(self.cont_col), dtype=torch.float)
            A_cont[:len(summoner_table_cont[a])] = torch.tensor(summoner_table_cont[a], dtype=torch.float)
            A_cont_data.append(A_cont)

            B_cate = torch.zeros(self.max_seq_len, len(self.cate_col), dtype=torch.int)
            B_cate[:len(summoner_table_cate[b])] = torch.tensor(summoner_table_cate[b], dtype=torch.int)
            B_cate_data.append(B_cate)

            B_cont = torch.zeros(self.max_seq_len, len(self.cont_col), dtype=torch.float)
            B_cont[:len(summoner_table_cont[b])] = torch.tensor(summoner_table_cont[b], dtype=torch.float)
            B_cont_data.append(B_cont)
        
        axis0 = len(combination)
        A_cate_data = torch.stack(A_cate_data).view(axis0, -1)
        A_cont_data = torch.stack(A_cont_data).view(axis0, -1)
        B_cate_data = torch.stack(B_cate_data).view(axis0, -1)
        B_cont_data = torch.stack(B_cont_data).view(axis0, -1)
        
        result = match_df['result'].values[0]
        result = torch.tensor(result, dtype=torch.float).expand(axis0)

        return A_cate_data.to(self.device), A_cont_data.to(self.device), B_cate_data.to(self.device), B_cont_data.to(self.device), result.to(self.device)
        

    def __len__(self):
        return len(self.match_compression_table)



def custom_collate_fn(batch):
    A, B, C, D, E = zip(*batch)

    A = torch.cat(A, dim=0)
    B = torch.cat(B, dim=0)
    C = torch.cat(C, dim=0)
    D = torch.cat(D, dim=0)
    E = torch.cat(E, dim=0)

    return A, B, C, D, E