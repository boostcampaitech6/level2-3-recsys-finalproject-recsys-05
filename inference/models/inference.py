import os
import json
import numpy as np

def inference(anchor_most3_champions, candidate_most3_champions):
    # load numpy matrix(.npy) from matrix directory
    if not os.path.exists('matrix'):
        raise ValueError("Matrix directory is not exists")

    # todo: n_champion should be loaded from config file
    n_champion = 167

    play_matrix = np.load('matrix/play_matrix.npy')
    win_rate = np.load('matrix/win_rate.npy')
    self_winrate = np.load('matrix/self_winrate.npy')

    # read json file to get index2name, name2idx
    with open('index2name.json', 'r') as f:
        index2name = json.load(f)
    with open('name2idx.json', 'r') as f:
        name2idx = json.load(f)

    # get synergy set from anchor_most3_champions, and each candidate's candidate_most3_champions
    # candidate_most3_champions -> {summonor_id_idx: most3_champions}
    anchor_synergy_set = get_synergy_set(anchor_most3_champions, n_champion, win_rate, self_winrate, play_matrix)

    # get intersection of anchor_synergy_set and candidate_synergy_set
    final_score = dict()
    for idx, most3_champions in candidate_most3_champions:
        intersection_score = get_intersection_score(anchor_synergy_set, most3_champions)
        final_score[idx] = intersection_score

    # return the final_score index
    final_score = sorted(final_score.items(), key=lambda x: x[1], reverse=True)
    return final_score

def get_intersection_score(anchor_synergy_set, most3_champions):
    score = 0
    for chapion_idx in most3_champions:
        if anchor_synergy_set.get(chapion_idx) is not None:
            score += anchor_synergy_set[chapion_idx]

    return score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_score(comb_winrate, self_winrate, play_count, mu_play_matrix, sigma_play_matrix ,alpha=1):
    x = alpha * ((play_count - mu_play_matrix) / sigma_play_matrix)

    weight = sigmoid(x)

    score = comb_winrate * weight + self_winrate * (1 - weight)

    return score

# synergy_set with score
def get_synergy_set(most3_champions, n_champion, win_rate, self_winrate, play_matrix):
    mu_play_matrix = np.mean(play_matrix)
    sigma_play_matrix = np.std(play_matrix)

    synergy_set = dict()
    for i in most3_champions:
        for j in range(n_champion):
            if i != j:
                score = calculate_score(win_rate[i, j], self_winrate[i], play_matrix[i, j], mu_play_matrix, sigma_play_matrix)
                if synergy_set.get(j) is None:
                    synergy_set[j] = score
                else:
                    synergy_set[j] += score

    synergy_set = sorted(synergy_set.items(), key=lambda x: x[1], reverse=True)[:10]

    return synergy_set
