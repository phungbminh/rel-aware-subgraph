import lmdb
def inspect_bad_lmdb_keys(db_path):
    bad_keys = []
    good_keys = []
    with lmdb.open(db_path, readonly=True, lock=False, readahead=False) as env:
        with env.begin() as txn:
            for k, v in txn.cursor():
                if v is None or v[:1] not in [b'\x80', b'\x81']:
                    bad_keys.append((k, v[:8] if v else None))
                else:
                    good_keys.append(k)
    print(f"[INFO] Có {len(good_keys)} key hợp lệ, {len(bad_keys)} key lỗi hoặc không phải pickle.")
    if bad_keys:
        for i, (k, h) in enumerate(bad_keys):
            print(f"[BAD {i}] key: {k}, header: {h}")
    return good_keys, bad_keys

good, bad = inspect_bad_lmdb_keys("/Users/minhbui/Personal/Project/Master/rel-aware-subgraph/debug_subgraph_db/train.lmdb")
