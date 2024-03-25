from pymongo import MongoClient
from tqdm import tqdm
import asyncio


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument("--tier", type=str, help="tier", required=True)
    parser.add_argument("--num_of_semaphore", type=int, help="num_of_semaphore")

    args = parser.parse_args()

    return args


class MongoDB:
    def __init__(self, batch_size, num_of_semaphore: int = 100):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(num_of_semaphore)


    async def generator_json(self, collection):
        json_gen = collection.find({}, batch_size=self.batch_size)
        for ajson in tqdm(json_gen, mininterval=2):
            async with self.semaphore:
                yield ajson


    async def save_to_mongo(self, collection, dump: list):
        async with self.semaphore:
            collection.insert_many(dump)


async def make_index_table(unique, batch_size, cate, db, mongo):
    to_index = []
    for i, value in tqdm(enumerate(unique, start=1), desc='Unique Value'):
        to_index.append({'key': value, 'value': i})

        if len(to_index) >= batch_size:
            await mongo.save_to_mongo(db[f'match_{cate}_to_index'], to_index)
            to_index = []
        
    await mongo.save_to_mongo(db[f'match_{cate}_to_index'], to_index)


def get_unique_values(unique, db, col, batch_size):
    for tier in tqdm(['diamond', 'emerald', 'platinum'], desc='Tier'):
        collection = db[f'{tier}_match_modv1']
        cursor = collection.find({}, {col: 1, '_id': 0}, batch_size=batch_size)
        for doc in tqdm(cursor, mininterval=2, desc='Document'):
            unique.add(doc[col])

async def main(args):
    cate_col = {}
    cate_col['other'] = ['match_id', 'summoner_id', 'team_key', 'position', 'champion_id', 'trinket_item', 'result']
    cate_col['item'] = ['item_0', 'item_1', 'item_2', 'item_3', 'item_4', 'item_5']
    cate_col['rune'] = ['rune_0', 'rune_1']
    cate_col['spell'] = ['spell_0', 'spell_1']

    client = MongoClient("mongodb://localhost:27017/")
    db = client["loldb"]

    batch_size = 100000
    mongo = MongoDB(batch_size)

    for cate in cate_col.keys():
        if cate == 'other':
            for col in cate_col[cate]:
                print(f"Processing {col}...")
                unique = set()

                get_unique_values(unique, db, col, batch_size)

                await make_index_table(unique, batch_size, cate, db, mongo)

        else:
            unique = set()
            for col in cate_col[cate]:
                print(f"Processing {col}...")

                get_unique_values(unique, db, col, batch_size)

            await make_index_table(unique, batch_size, cate, db, mongo)



if __name__ == "__main__":
    args = parse_args()

    asyncio.run(main(args))
