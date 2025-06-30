import lmdb
import pickle
from pprint import pprint

class SubgraphLMDBInspector:
    def __init__(self, lmdb_path, max_print=5):
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(
            lmdb_path, readonly=True, lock=False, max_dbs=1
        )
        self.db = self.env.open_db()
        self.max_print = max_print

    def keys(self):
        """Liệt kê các key trong LMDB"""
        with self.env.begin(db=self.db) as txn:
            keys = [key for key, _ in txn.cursor()]
        return keys

    def get(self, key):
        """Đọc 1 key (string hoặc bytes)"""
        if isinstance(key, str):
            key = key.encode()
        with self.env.begin(db=self.db) as txn:
            val = txn.get(key)
            if val is not None:
                return pickle.loads(val)
            return None

    def print_some(self):
        """In thử vài subgraph đầu tiên"""
        print(f"Inspecting: {self.lmdb_path}")
        with self.env.begin(db=self.db) as txn:
            cursor = txn.cursor()
            count = 0
            for key, val in cursor:
                if key.startswith(b'_progress'):
                    continue
                obj = pickle.loads(val)
                print(f"Key: {key}")
                pprint(obj)
                count += 1
                if count >= self.max_print:
                    break
        print("...")

    def check_progress(self):
        """Kiểm tra trạng thái COMPLETE hoặc progress hiện tại"""
        with self.env.begin(db=self.db) as txn:
            prog = txn.get(b'_progress')
            if prog == b'COMPLETED':
                print("LMDB extraction: COMPLETED ✅")
            elif prog is not None:
                print(f"LMDB extraction progress: {int(prog)} items written so far")
            else:
                print("No progress info found (may be finished or not tracked)")

    def count_items(self):
        """Đếm số subgraph đã lưu (không tính _progress)"""
        with self.env.begin(db=self.db) as txn:
            count = 0
            for key, _ in txn.cursor():
                if not key.startswith(b'_'):
                    count += 1
            print(f"Total subgraphs: {count}")
            return count

    def close(self):
        self.env.close()


# Ví dụ sử dụng:
if __name__ == "__main__":
    inspector = SubgraphLMDBInspector("./debug_subgraph_db/train.lmdb")
    inspector.check_progress()
    inspector.count_items()
    inspector.print_some()
    inspector.close()
