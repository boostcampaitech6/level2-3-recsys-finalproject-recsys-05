from matrix import synergy_matrix, save_matrix
from dataset import CustomDataset


# dataset에서 data 불러오기\
data = CustomDataset('/Users/seoku/workspace/naver_boost/duofinder/inference/models/matches.jsonl').get_data()

play_matrix, win_rate, self_winrate = synergy_matrix(data)
save_matrix(play_matrix, win_rate, self_winrate)

if __name__ == "__main__":
    print("Matrix saved successfully")
