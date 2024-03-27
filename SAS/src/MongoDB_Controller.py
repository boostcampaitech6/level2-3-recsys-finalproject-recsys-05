import asyncio
from tqdm import tqdm
import pymongo


class MongoDBController:
    def __init__(self, client, batch_size, db_name: str = 'loldb', num_of_semaphore: int = 100):
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(num_of_semaphore)
        self.db = client[db_name]

    async def generator_json(self, collection):
        json_gen = collection.find({}, batch_size=self.batch_size)
        for ajson in tqdm(json_gen, mininterval=2):
            async with self.semaphore:
                yield ajson


    async def save_to_mongo(self, collection_name, dump: list):
        async with self.semaphore:
            self.db[collection_name].insert_many(dump)


    async def make_index_table(self, unique, collection_name, batch_size) -> None:
        to_index = []
        for i, value in tqdm(enumerate(unique, start=1), desc='Unique Value'):
            to_index.append({'key': value, 'value': i})
            
            if len(to_index) >= batch_size:
                await self.save_to_mongo(collection_name, to_index)
                to_index = []
            
        await self.save_to_mongo(collection_name, to_index)


    def get_unique_values(self, collection_name: str, col, batch_size) -> set:
        unique = set()
        cursor = self.db[collection_name].find({}, {col: 1, '_id': 0}, batch_size=batch_size)
        for doc in tqdm(cursor, mininterval=2, desc='Document'):
            unique.add(doc[col])

        return unique
    

    def __getitem__(self, key) -> pymongo.collection.Collection:
        return self.db[key]
    

    def get_cate_to_index(self, prefix:str, cate: dict) -> dict:
        cate_to_index = {}
        for key in cate.keys():
            if key == 'other':
                for col in cate[key]:
                    docs = self.db[f'{prefix}_{col}_to_index'].find({}, batch_size=self.batch_size)
                    cate_to_index[col] = {d['key']: d['value'] for d in docs}

            else:
                docs = self.db[f'{prefix}_{key}_to_index'].find({}, batch_size=self.batch_size)
                temp = {d['key']: d['value'] for d in docs}

                for col in cate[key]:
                    cate_to_index[col] = temp

        return cate_to_index
    

    def get_cate_to_index_len(self, prefix:str, cate: dict) -> dict:
        cate_len= {}
        for key in cate.keys():
            if key == 'other':
                for col in cate[key]:
                    cate_len[col] = self.db[f'{prefix}_{col}_to_index'].count_documents({})

            else:
                cate_len[key] = self.db[f'{prefix}_{key}_to_index'].count_documents({})

        return cate_len