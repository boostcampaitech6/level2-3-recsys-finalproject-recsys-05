import os
import numpy as np
import pandas as pd
from dataset import CustomDataset

# dataset에서 data 불러오기
# data = CustomDataset('/Users/seoku/workspace/naver_boost/duofinder/inference/models/matches.jsonl')

def save_matrix(play_matrix, win_rate, self_winrate):
    # save matrix to matrix directory
    # 나중에 config
    if not os.path.exists('matrix'):
        os.makedirs('matrix')
    np.save('matrix/play_matrix.npy', play_matrix)
    np.save('matrix/win_rate.npy', win_rate)
    np.save('matrix/self_winrate.npy', self_winrate)

def synergy_matrix(data):
    # number of champion(가변)
    n_champion = 167
    index2name, name2idx = champion_id_indexing()

    play_matrix = get_champion_combination(data, n_champion, index2name)
    win_rate = get_winrate(data, play_matrix, n_champion, index2name)
    self_winrate = get_self_winrate(data, n_champion, name2idx)

    return play_matrix, win_rate, self_winrate

# champion 조합 win_rate matrix 만들기
def get_winrate(data: pd.DataFrame, play_matrix, n_champion, index2name):
    eps = 1e-10
    win_count = np.zeros((n_champion, n_champion), dtype=int)

    for d in data.values:
        if d[5] == 1:
            for i in range(5):
                for j in range(i+1, 5):
                    win_count[index2name[str(d[i])], index2name[str(d[j])]] += 1
                    win_count[index2name[str(d[j])], index2name[str(d[i])]] += 1

    win_rate = win_count / (play_matrix + eps)
    win_rate[np.isnan(win_rate)] = 0
    win_rate[np.isinf(win_rate)] = 0

    return win_rate


# champion 조합 matrix 만들기
def get_champion_combination(data: pd.DataFrame, n_champion, index2name):
    # number of champion(가변)
    n_champion = 167

    play_matrix = np.zeros((n_champion, n_champion), dtype=int)

    for d in data.values:
        for i in range(5):
            for j in range(i+1, 5):
                play_matrix[index2name[str(d[i])], index2name[str(d[j])]] += 1
                play_matrix[index2name[str(d[j])], index2name[str(d[i])]] += 1

    return play_matrix

# champion_id indexing
def champion_id_indexing():
    import json

    with open('/Users/seoku/workspace/naver_boost/duofinder/inference/models/champion_name_key.json', 'r') as f:
        champion_name_key = json.load(f)

    index2name = {k: i for i, k in enumerate(champion_name_key.keys())}
    name2idx = {i: k for i, k in enumerate(champion_name_key.keys())}

    # save index2name, name2idx
    with open('index2name.json', 'w') as f:
        json.dump(index2name, f)
    with open('name2idx.json', 'w') as f:
        json.dump(name2idx, f)

    return index2name, name2idx


# champion 개별 승률
def get_self_winrate(data, n_champion, name2idx):
    champion_cross1 = pd.crosstab(data['champion_1'], data['result'])
    champion_cross1['win_rate'] = champion_cross1[1] / (champion_cross1[0] + champion_cross1[1])
    champion_cross1 = champion_cross1.sort_values(by='win_rate', ascending=False)

    champion_cross2 = pd.crosstab(data['champion_2'], data['result'])
    champion_cross2['win_rate'] = champion_cross2[1] / (champion_cross2[0] + champion_cross2[1])
    champion_cross2 = champion_cross2.sort_values(by='win_rate', ascending=False)

    champion_cross3 = pd.crosstab(data['champion_3'], data['result'])
    champion_cross3['win_rate'] = champion_cross3[1] / (champion_cross3[0] + champion_cross3[1])
    champion_cross3 = champion_cross3.sort_values(by='win_rate', ascending=False)

    champion_cross4 = pd.crosstab(data['champion_4'], data['result'])
    champion_cross4['win_rate'] = champion_cross4[1] / (champion_cross4[0] + champion_cross4[1])
    champion_cross4 = champion_cross4.sort_values(by='win_rate', ascending=False)

    champion_cross5 = pd.crosstab(data['champion_5'], data['result'])
    champion_cross5['win_rate'] = champion_cross5[1] / (champion_cross5[0] + champion_cross5[1])
    champion_cross5 = champion_cross5.sort_values(by='win_rate', ascending=False)

    # champion cross table 합하기
    champion_cross = champion_cross1.add(champion_cross2, fill_value=0).add(champion_cross3, fill_value=0).add(champion_cross4, fill_value=0).add(champion_cross5, fill_value=0)
    champion_cross['win_rate'] = champion_cross[1] / (champion_cross[0] + champion_cross[1])
    champion_cross = champion_cross.sort_values(by='win_rate', ascending=False)

    self_win_rate = np.zeros(n_champion)
    for i in range(n_champion):
        self_win_rate[i] = champion_cross.loc[int(name2idx[i]), 'win_rate']

    return self_win_rate
