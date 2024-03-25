from pymongo import MongoClient
from tqdm import tqdm
import orjson
import os


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--tier", type=int, help="tier", required=True)
    parser.add_argument("--data_dir", type=int, help="data_dir", required=True)

    args = parser.parse_args()

    return args


def put_match_data(collection_match, dump):
    collection_match.insert_many(dump)


def collect_data(collection_match, json_data_gen):
    dump = []
    for data in json_data_gen:
        if verify_json(data):
            dump += data 

        if len(dump) >= 2000:  # 수정: 정확한 비교 연산자 사용
            put_match_data(collection_match, dump)
            dump = []  # dump 리스트를 초기화


def verify_json(json_data):
    return 'participants' in json_data[0]


def gen_json(file_gen):
    for file in tqdm(file_gen, mininterval=2):
        with open(file.path, 'rb') as f:
            try:
                json_data = orjson.loads(f.read())
            except Exception as e:
                print('Error reading {}: {}'.format(file.path, e))
                continue
        yield json_data


def migrate(stored_dir, collection_match):
    file_gen = os.scandir(stored_dir)
    json_data_gen = gen_json(file_gen)

    collect_data(collection_match, json_data_gen)


if __name__ == '__main__':
    argu = parse_args()
    tier = argu.tier
    data_dir = argu.data_dir

    client = MongoClient("mongodb://power16one5.iptime.org:27017/")
    db = client["loldb"]
    collection_match = db[f'{tier}_match']

    migrate(data_dir, collection_match)
