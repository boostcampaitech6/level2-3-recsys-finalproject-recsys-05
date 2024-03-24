from pymongo import MongoClient
from tqdm import tqdm
from collections import defaultdict


if __name__ == "__main__":
    tier = 'platinum'

    client = MongoClient("mongodb://localhost:27017/")
    db = client["loldb"]
    collection_match = db[f'{tier}_match']

    doc_ids = collection_match.find({}, {'id': 1})
    doc_id_count = defaultdict(list)
    for doc in tqdm(doc_ids):
        doc_id_count[doc['id']].append(doc['_id'])

    id_list = []
    # 중복된 docId를 가진 문서 삭제 (첫 번째 문서 제외)
    for docId, ids in tqdm(doc_id_count.items()):
        if len(ids) > 1:
            # 첫 번째 ID를 제외하고 삭제
            id_list += ids[1:]
            
    for i in tqdm(range(0, len(id_list), 1000)):
        collection_match.delete_many({'_id': {'$in': id_list[i:i+1000]}})
