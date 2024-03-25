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
    cate_cols = {}
    cate_cols['other'] = ['summonerId', 'teamId', 'individualPosition', 'role', 'championId', 'win', 'defense', 'flex', 'offense', 'matchId' ]
    cate_cols['item'] = ['item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6']
    cate_cols['summonerSpell'] = ['summoner1Id', 'summoner2Id']

    batch_size = 100000
    client = MongoClient("mongodb://localhost:27017/")
    mongo = MongoDBController(client, batch_size)

    for cate in cate_cols.keys():
        if cate == 'other':
            for col in cate_cols[cate]:
                print(f"Processing {col}...")

                unique = mongo.get_unique_values('riot_match_modv1', col, batch_size)

                await mongo.make_index_table(unique, f'riot_match_modv1_{col}_to_index', batch_size)

        else:
            unique = set()
            for col in cate_cols[cate]:
                print(f"Processing {col}...")

                unique = unique.union(mongo.get_unique_values('riot_match_modv1', col, batch_size))

            await mongo.make_index_table(unique, f'riot_match_modv1_{cate}_to_index', batch_size)



if __name__ == "__main__":
    args = parse_args()

    asyncio.run(main(args))
