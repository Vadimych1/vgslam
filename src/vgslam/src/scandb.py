import faiss as fs
import numpy as np
import m2dp
from io import BytesIO
import struct
from .models import PositionedCloud


class ScanDB:
    def __init__(self) -> None:
        self.sub_index = fs.IndexFlatL2(192)
        self.index = fs.IndexIDMap(self.sub_index)
        self.history_nodes = []

    def _get_descriptor(self, scan: PositionedCloud):
        if scan._descriptor is None:
            cloud = scan.cloud
            if scan.cloud.shape[1] == 2:
                zeros = np.zeros((cloud.shape[0], 1), dtype=np.float32)
                cloud = np.hstack((cloud, zeros))
            desc, _ = m2dp.M2DP(cloud)
            scan._descriptor = desc
        return scan._descriptor

    def add_vertex_scan(self, vertex_id: int, scan: PositionedCloud):
        desc = self._get_descriptor(scan)
        vector_matrix = np.expand_dims(desc, axis=0).astype(np.float32)
        idx_matrix = np.array([vertex_id], dtype=np.int64)     # faiss ждёт int64
        self.index.add_with_ids(vector_matrix, idx_matrix)
        self.history_nodes.append(vertex_id)

    def search_candidates(self, scan, top_k=3, exclusion_window=6):
        if self.index.ntotal < exclusion_window:
            return [], []

        query = self._get_descriptor(scan).astype(np.float32)
        query_mat = np.expand_dims(query, axis=0)

        raw_k = top_k + exclusion_window
        distances, indices = self.index.search(query_mat, k=raw_k)

        valid_candidates = []
        valid_distances = []
        current_max_id = self.history_nodes[-1] if self.history_nodes else 0

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if abs(current_max_id - idx) < exclusion_window:
                continue
            valid_candidates.append(int(idx))
            valid_distances.append(float(dist))
            if len(valid_candidates) >= top_k:
                break

        return valid_candidates, valid_distances

    def to_bytes(self) -> bytes:
        index_bytes = fs.serialize_index(self.index)   # np.ndarray uint8

        hist = np.asarray(self.history_nodes, dtype=np.int64)

        buf = BytesIO()
        buf.write(struct.pack(">I", index_bytes.size))
        buf.write(index_bytes.tobytes())
        buf.write(struct.pack(">Q", hist.size))
        buf.write(hist.tobytes())
        return buf.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> "ScanDB":
        db = ScanDB.__new__(ScanDB)
        off = 0

        n_idx = struct.unpack_from(">I", data, off)[0]; off += 4
        idx_arr = np.frombuffer(data[off:off + n_idx], dtype=np.uint8)
        off += n_idx
        db.index = fs.deserialize_index(idx_arr)

        if isinstance(db.index, fs.IndexIDMap):
            db.sub_index = fs.downcast_index(db.index.index)
        else:
            db.sub_index = db.index

        n_hist = struct.unpack_from(">Q", data, off)[0]; off += 8
        hist = np.frombuffer(data[off:off + n_hist * 8], dtype=np.int64).copy()
        db.history_nodes = hist.tolist()

        return db