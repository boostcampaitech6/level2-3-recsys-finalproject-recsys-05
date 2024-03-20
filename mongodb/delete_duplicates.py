from pymongo import MongoClient
from tqdm import tqdm
from collections import defaultdict
# import asyncio


# async def delete_duplicates(semaphore, collection_match, doc):
#     async with semaphore:
#         collection_match.delete_many({
#                 "_id": { "$in": doc['uniqueIds'] } # 나머지 _id를 가진 문서들을 삭제
#             })


# async def main():
#     tier = 'emerald'
#     semaphore = asyncio.Semaphore(100)

#     client = MongoClient("mongodb://localhost:27017/")
#     db = client["loldb"]
#     collection_match = db[f'{tier}_match']

#     doc_ids = collection_match.find({}, {'id': 1})
#     doc_id_count = defaultdict(list)
#     for doc in doc_ids:
#         doc_id_count[doc['id']].append(doc['_id'])

#     # 중복된 docId를 가진 문서 삭제 (첫 번째 문서 제외)
#     for docId, ids in doc_id_count.items():
#         if len(ids) > 1:
#             # 첫 번째 ID를 제외하고 삭제
#             collection_match.delete_many({'_id': {'$in': ids[1:]}})


if __name__ == "__main__":
    # asyncio.run(main())
    tier = 'emerald'
    # semaphore = asyncio.Semaphore(100)

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
            
    collection_match.delete_many({'_id': {'$in': id_list}})
