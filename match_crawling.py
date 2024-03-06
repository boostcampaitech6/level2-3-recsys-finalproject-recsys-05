import json
import orjson
import os
import requests
import time
import argparse
from tqdm import tqdm



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tier", type=str, help="", default="gold")
    parser.add_argument("--summoner_id_path", type=str, help="", default="../boostcamp_notebooks/summorer_ids")
    parser.add_argument("--output_dir", type=str, help="", default="/home/piddle/hdd/matches")
    
    args = parser.parse_args()

    return args


### 각종 json에 필요없는 요소를 제거하는 함수입니다.
def data_cleaning(data):
    data = data['data']
    
    ### myData 요소를 제거합니다.
    for d in data:
        del d['myData']

    return data


def write_json(data, output_dir):
    for summoner_id, match in data:
        with open(os.path.join(output_dir, f"{summoner_id}.json"), 'wb') as file:
            file.write(orjson.dumps(match))
            

def summoner_id_generator(summoner_id_path, tier, output_dir):
    num_of_file = len(os.listdir(output_dir))

    with open(os.path.join(summoner_id_path, tier + '.json')) as json_file:
        json_data = json.load(json_file)
        for i in json_data[num_of_file:]:
            summoner_id = i['summoner_id']
            yield summoner_id


def left_summoner_id_generator(summoner_id_path, tier, target_dir):
    import pandas as pd

    original = pd.read_json(os.path.join(summoner_id_path, tier + '.json'))
    original = original['summoner_id'].sort_values().drop_duplicates()
    file = pd.Series(os.listdir(target_dir)).sort_values().apply(lambda x: x[:-5])

    bmax = len(file)
    need = []
    j = 0
    for i in range(len(original)):
        if original.iloc[i] == file.iloc[j]:
            j += 1
            if j == bmax:
                break

        else:
            need.append(original.iloc[i])

    print(f'left ids : {len(need)}')

    for i in tqdm(need, mininterval=2):
        yield i


def crawling_match_generator(headers, summoner_id_generator):
    for summoner_id in summoner_id_generator:
        url = f"https://www.op.gg/api/v1.0/internal/bypass/games/kr/summoners/{summoner_id}?&limit=10&hl=ko_KR&game_type=soloranked"
        sleep_time = 1
        
        while True:
            try:
                resp = requests.get(url, headers=headers)
            except ConnectionError as e:
                now = time.strftime('%Y.%m.%d - %H:%M:%S')
                tqdm.write(f"{now} : ConnectionError : {e}")
                time.sleep(5)
                continue

            if resp.status_code == 200:
                user_json = orjson.loads(resp.text)
                user_json = data_cleaning(user_json)
                break

            ### 응답코드가 429일 때, 대기 시간을 1초씩 늘려가며 다시 요청합니다.
            elif resp.status_code == 429 or resp.status_code // 100 == 5:
                time.sleep(sleep_time)

                now = time.strftime('%Y.%m.%d - %H:%M:%S')
                tqdm.write(f'{now} : sleep {sleep_time}')   
                sleep_time += 1
                continue

            else:
                tqdm.write(f"unexpected response : {resp.status_code}, summoncer_id : {summoner_id}")
                user_json = orjson.loads('[{"resp.status_code": ' + f'{resp.status_code}' + ' }]')
                break

        yield summoner_id, user_json


if __name__ == "__main__":
    argu = parse_args()

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0',}
    
    target_dir = os.path.join(argu.output_dir, argu.tier)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        summoner_id_gen = summoner_id_generator(argu.summoner_id_path, argu.tier, target_dir)
    else:
        summoner_id_gen = left_summoner_id_generator(argu.summoner_id_path, argu.tier, target_dir)
    
    match_gen = crawling_match_generator(headers, summoner_id_gen)
    write_json(match_gen, target_dir)
