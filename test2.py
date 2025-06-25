import lmdb, pickle
env = lmdb.open('./data/subgraph_db/lmdb_train', readonly=True, lock=False, max_dbs=4)
db_pos = env.open_db(b'positive')
with env.begin(db=db_pos) as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        try:
            obj = pickle.loads(value)
            print(key, type(obj), obj)
        except Exception as e:
            print(f"Lỗi ở key {key}: {e}")