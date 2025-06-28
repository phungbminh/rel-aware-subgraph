import lmdb
import pickle
import os

lmdb_path = "./data/subgraph_db/lmdb_train"
env = lmdb.open(lmdb_path, readonly=True, lock=False, max_dbs=4)


dbs = env.stat()
print(dbs)

db_pos = env.open_db(b'positive')
db_neg = env.open_db(b'negative')

with env.begin(db=db_pos) as txn:
    # Đếm số record
    n = txn.stat()['entries']
    print(f"Positive DB có {n} sample")

    # Lấy 1 sample ra xem
    for key, value in txn.cursor():
        subgraph = pickle.loads(value)
        print(f"Key: {key}")
        print(f"Subgraph info: {subgraph}")
        break  # bỏ break nếu muốn in hết

with env.begin(db=db_neg) as txn:
    n = txn.stat()['entries']
    print(f"Negative DB có {n} sample")
    # Lấy 1 sample ra xem
    for key, value in txn.cursor():
        neg_samples = pickle.loads(value)
        print(f"Key: {key}")
        print(f"Negative sample(s): {neg_samples}")
        break