from tqdm import tqdm
import asyncio
from pymongo import MongoClient
from src.MongoDB_Controller import MongoDBController


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument("--tier", type=str, help="tier", required=True)
    parser.add_argument("--num_of_semaphore", type=int, help="num_of_semaphore")

    args = parser.parse_args()

    return args


async def main(args):
    cate_to_index_dict = {}
    cate_to_index_dict['other'] = ['summonerId', 'teamId', 'individualPosition', 'role', 'championId', 'win', 'defense', 'flex', 'offense', 'matchId' ]
    cate_to_index_dict['item'] = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6']
    cate_to_index_dict['summonerSpell'] = ['summoner1Id', 'summoner2Id']

    except_cols = {'_id': 0, 'riotIdGameName': 0, 'riotIdTagline': 0, 'gameVersion': 0, 'gameCreation': 0}
    cate_cols = ['summonerId', 'teamId', 'individualPosition', 'role', 'championId', 'win', 'defense', 'flex', 'offense', 'matchId',
                 'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'summoner1Id', 'summoner2Id']

    batch_size = 100000
    client = MongoClient("mongodb://localhost:27017/")
    mongo = MongoDBController(client, batch_size)

    cate_to_index = mongo.get_cate_to_index('SAS_data_v2', cate_to_index_dict)

    dump = []
    cursor = mongo['SAS_data_v2'].find({}, except_cols, batch_size=batch_size)
    for doc in tqdm(cursor):
        for col in cate_cols:
            doc[col] = cate_to_index[col][doc[col]]

        dump.append(doc)

        if len(dump) >= batch_size:
            await mongo.save_to_mongo('SAS_data_v3', dump)
            dump = []

    await mongo.save_to_mongo('SAS_data_v3', dump)
        



if __name__ == "__main__":
    args = parse_args()

    asyncio.run(main(args))
