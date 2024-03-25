from pymongo import MongoClient
from tqdm import tqdm
from collections import defaultdict
import asyncio


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--tier", type=str, help="tier", required=True)
    parser.add_argument("--num_of_semaphore", type=int, default=100, help="num_of_semaphore")

    args = parser.parse_args()

    return args



class MongoDB:
    def __init__(self, tier: str, num_of_semaphore: int = 100):
        self.tier = tier
        self.semaphore = asyncio.Semaphore(num_of_semaphore)

        self.selected_cols1 = ['champion_id', 'team_key', 'position', 'trinket_item']

        self.selected_cols2 = ['champion_level', 'damage_self_mitigated', 'damage_dealt_to_objectives', 'damage_dealt_to_turrets',
                        'total_damage_taken', 'total_damage_dealt', 'total_damage_dealt_to_champions', 'time_ccing_others',
                        'time_ccing_others', 'vision_wards_bought_in_game', 'sight_wards_bought_in_game', 'ward_kill', 'ward_place',
                        'turret_kill', 'kill', 'death', 'assist', 'neutral_minion_kill', 'gold_earned', 'total_heal']

        client = MongoClient("mongodb://localhost:27017/")
        db = client["loldb"]
        self.load_collection = db[f'{self.tier}_match']
        self.save_match_collection = db[f'{self.tier}_match_modv1']

        self.gen_json = self.generator_json()


    async def generator_json(self):
        json_gen = self.load_collection.find({}, batch_size=100000)
        for ajson in tqdm(json_gen, mininterval=2):
            async with self.semaphore:
                yield ajson


    async def save_to_mongo(self, collection, dump: list):
        async with self.semaphore:
            collection.insert_many(dump)


    def parse_match_by_match(self, participant) -> dict:
        match_by_match_id = {}
        match_by_match_id['summoner_id'] = participant['summoner']['summoner_id']
        match_by_match_id['summoner_level'] = participant['summoner']['level']

        for col in self.selected_cols1:
            match_by_match_id[col] = participant[col]

        for i, item in enumerate(participant['items']):
            match_by_match_id[f'item_{i}'] = item

        match_by_match_id['rune_0'] = participant['rune']["primary_rune_id"]
        match_by_match_id['rune_1'] = participant['rune']["secondary_page_id"]
        match_by_match_id['spell_0'] = participant['spells'][0]
        match_by_match_id['spell_1'] = participant['spells'][1]

        stats = participant['stats']
        for col in self.selected_cols2:
            match_by_match_id[col] = stats[col]

        match_by_match_id['vision_score'] = stats['vision_score']
        match_by_match_id['result'] = stats['result']

        return match_by_match_id

    
    async def start(self):
        dump = []

        async for match in self.gen_json:
            for participant in match['participants']:

                match_by_match_id = self.parse_match_by_match(participant)
                match_by_match_id['match_id'] = match['id']
                match_by_match_id['game_length_second'] = match['game_length_second']

                dump.append(match_by_match_id)

                ### 데이터를 20000개씩 끊어서 저장
                if len(dump) >= 20000:
                    await self.save_to_mongo(self.save_match_collection, dump)
                    dump = []

        await self.save_to_mongo(self.save_match_collection, dump)



async def main(args):
    mongodb = MongoDB(args.tier, args.num_of_semaphore)
    await mongodb.start()


if __name__ == "__main__":
    args = parse_args()

    asyncio.run(main(args))
