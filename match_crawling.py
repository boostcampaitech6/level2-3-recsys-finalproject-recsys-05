import json
import orjson
import os
import requests
import time
import argparse


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
            

def summoner_id_generator(summoner_id_path, output_dir):
    num_of_file = len(os.listdir(output_dir))

    with open(summoner_id_path) as json_file:
        json_data = json.load(json_file)
        for i in json_data[num_of_file:]:
            summoner_id = i['summoner_id']
            yield summoner_id


def crawling_match_generator(headers, summoner_id_generator):
    for summoner_id in summoner_id_generator:
        # print(f"start: {summoner_id}")
        url = f"https://www.op.gg/api/v1.0/internal/bypass/games/kr/summoners/{summoner_id}?&limit=10&hl=ko_KR&game_type=soloranked"
        resp = requests.get(url, headers=headers)

        if resp.status_code == 200:
            user_json = orjson.loads(resp.text)
            user_json = data_cleaning(user_json)
            yield summoner_id, user_json

        ### 응답코드가 429일 때, 대기 시간을 1초씩 늘려가며 다시 요청합니다.
        elif resp.status_code == 429:
            sleep_time = 1
            while resp.status_code == 429:
                time.sleep(sleep_time)
                sleep_time += 1

                now = time.strftime('%Y.%m.%d - %H:%M:%S')
                print(f'{now} : sleep {sleep_time}')      
                resp = requests.get(url, headers=headers)
            now = time.strftime('%Y.%m.%d - %H:%M:%S')
            print(f'{now}: sleep done')      
        
        else:
            print(f"unexpected response : {resp.status_code}, summoncer_id : {summoner_id}")
            user_json = orjson.loads('[{"resp.status_code": ' + f'{resp.status_code}' + ' }]')
            yield summoner_id, user_json

        

if __name__ == "__main__":
    argu = parse_args()

    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0',}
    
    target_dir = os.path.join(argu.output_dir, argu.tier)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    summoner_id_gen = summoner_id_generator(os.path.join(argu.summoner_id_path, argu.tier + '.json'), target_dir)
    match_gen = crawling_match_generator(headers, summoner_id_gen)
    write_json(match_gen, target_dir)