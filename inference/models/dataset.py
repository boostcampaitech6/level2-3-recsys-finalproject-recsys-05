import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from io import StringIO

class CustomDataset(Dataset):
    def __init__(self, matches_file, transform=None):
        self.transform = transform
        self.data = self.preprocess(matches_file)

    def preprocess(self, matches_file):
        # load data
        df = pd.read_json(matches_file, lines=True)

        team_result = []
        # 각 팀별로 데이터를 수집합니다.
        for i in range(0, len(df), 5):  # 여기서는 예시 데이터가 2개밖에 없으므로, 실제 데이터에 맞게 조정해야 합니다.
            team_data = df.iloc[i:i+5]  # 5개의 row를 선택 (실제 데이터가 5개 이상일 때 유효)
            if not team_data.empty:
                champion_ids = team_data['champion_id'].tolist()
                result = team_data['result'].iloc[0]  # 가정: 한 팀 내 모든 row의 승리 여부가 동일합니다.
                team_result.append(champion_ids + [result])

        # 새로운 DataFrame 생성
        columns = [f'champion_{i+1}' for i in range(5)] + ['result']
        match_df = pd.DataFrame(team_result, columns=columns)

        return match_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_data(self):
        return self.data

