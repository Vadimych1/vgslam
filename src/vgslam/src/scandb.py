import faiss as fs
# import map_closures.map_closures as mc
import numpy as np
import m2dp
from .models import PositionedCloud

class ScanDB:
    def __init__(self) -> None:
        # config = mc.MapClosuresConfig()
        # self.closure_detector = mc.MapClosures(config)
        
        self.sub_index = fs.IndexFlatL2(192) # M2DP returns 192-dim descriptor
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
        
        vector_matrix = np.expand_dims(desc, axis=0)
        idx_matrix = np.array([vertex_id], dtype=np.uint64)
        
        self.index.add_with_ids(vector_matrix, idx_matrix)
        self.history_nodes.append(vertex_id)
    
    def search_candidates(self, scan: PositionedCloud, top_k: int = 3, exclusion_window: int = 6):
        if self.index.ntotal < exclusion_window:
            return [], []
        
        query = self._get_descriptor(scan)
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
