from dataclasses import dataclass
from .models import PositionedCloud
from .icp_matcher import ICPMatcher
from .graph import PoseGraph
from .scandb import ScanDB
from .helpers import *
import numpy as np

@dataclass
class LocalizationHypothesis:
    pose: np.ndarray
    vertex_id: int
    log_weight: float = 0.0

class LocalizationHypothesisSelection:
    def __init__(
        self,
        saved_scans: dict[int, PositionedCloud],
        matcher: ICPMatcher,
        graph: PoseGraph,
        scan_db: ScanDB,
        loc_max_hypotheses: int = 8,
        loc_init_min_fitness: float = 0.3,
        loc_commit_prob: float = 0.85,          # absolute posterior for the winner
        loc_commit_margin: float = 3.0,         # log-odds vs. runner-up
        loc_min_scans: int = 3,                 # evidence before we commit
        loc_rmse_sigma: float = 0.05,           # for ICP likelihood
        loc_odom_sigma: float = 0.3,            # m, for odometry consistency
        loc_odom_ang_sigma: float = float(np.deg2rad(10)),
        loc_prune_thresh: float = 1e-3,
        loc_dup_dist: float = 0.5,              # m, dedup threshold
        loc_dup_ang: float = float(np.deg2rad(30)),
    ) -> None:
        self._hypotheses: list[LocalizationHypothesis] = []
        self._prev_scan: PositionedCloud | None = None
        self._scan_count: int = 0
        
        self._max_hypotheses   = loc_max_hypotheses
        self._init_min_fitness = loc_init_min_fitness
        self._commit_prob      = loc_commit_prob
        self._commit_margin    = loc_commit_margin
        self._min_scans        = loc_min_scans
        self._rmse_sigma       = loc_rmse_sigma
        self._odom_sigma       = loc_odom_sigma
        self._odom_ang_sigma   = loc_odom_ang_sigma
        self._prune_thresh     = loc_prune_thresh
        self._dup_dist         = loc_dup_dist
        self._dup_ang          = loc_dup_ang
        
        self._graph = graph
        self._saved_scans = saved_scans
        self._matcher = matcher
        self._scandb = scan_db
    
    def reset(self):
        self._hypotheses.clear()
        self._prev_scan = None
        self._scan_count = 0
    
    def find_on_map(self, scan: PositionedCloud) -> tuple[LocalizationHypothesis, PositionedCloud] | None:
        if not self._hypotheses:
            self._hypotheses = self._init_hypotheses(scan)
            self._prev_scan = scan
            self._scan_count = 1
            return
        
        if self._prev_scan is None:
            return
        
        self._scan_count += 1
        
        d_odom = relative_pose(self._prev_scan.pos, scan.pos)
        T_delta = pose2d_to_transform(d_odom)
        
        for h in self._hypotheses:
            T_pred = pose2d_to_transform(h.pose) @ T_delta
            pred_pose = transform_4x4_to_2d_pose(T_pred)
            
            ref_scan = self._saved_scans[h.vertex_id]
            ref_vertex = self._graph.vertices[h.vertex_id]
            
            rel_override = relative_pose_4x4(ref_vertex.pose, pred_pose)
            T_ref_query, (fitness, rmse) = self._matcher.match(
                scan, ref_scan, rel_override,
            )
            
            if fitness <= 0.0:
                h.log_weight -= 50.0
                continue
            
            T_map_query = pose2d_to_transform(ref_vertex.pose) @ T_ref_query
            new_pose = transform_4x4_to_2d_pose(T_map_query)
            
            corr_xy = float(np.linalg.norm(new_pose[:2] - pred_pose[:2]))
            corr_th = float(abs(normalize_angle(new_pose[2] - pred_pose[2])))
            
            h.pose = new_pose
            h.log_weight += self._log_likelihood(fitness, rmse, corr_xy, corr_th)
        
        self._prune()
        self._prev_scan = scan
        
        if not self._hypotheses:
            return

        log_ws = np.array([h.log_weight for h in self._hypotheses])
        log_ws -= log_ws.max()
        
        probs = np.exp(log_ws)
        probs /= probs.sum()
        
        order = np.argsort(probs)[::-1]
        best = self._hypotheses[order[0]]
        best_prob = float(probs[order[0]])

        if len(order) > 1:
            margin = float(np.log(probs[order[0]] + 1e-12) - np.log(probs[order[1]] + 1e-12))

        else:
            margin = float('inf')

        if (best_prob >= self._commit_prob
            and margin >= self._commit_margin
            and self._scan_count >= self._min_scans
        ):
            return best, scan

    def _init_hypotheses(self, scan: PositionedCloud) -> list[LocalizationHypothesis]:
        candidates, _ = self._scandb.search_candidates(
            scan,
            top_k=self._max_hypotheses,
            exclusion_window=0,
            current_max_id=-1,
        )
        
        if len(candidates) == 0:
            return []
        
        hyps: list[LocalizationHypothesis] = []
        for cid in candidates:
            ref_scan = self._saved_scans[cid]
            ref_vertex = self._graph.vertices[cid]
            
            T_ref_query, (fitness, rmse) = self._matcher.match(
                scan, ref_scan, rel_override=None
            )
            
            if fitness < self._init_min_fitness:
                continue
            
            T_map_query = pose2d_to_transform(ref_vertex.pose) @ T_ref_query
            pose = transform_4x4_to_2d_pose(T_map_query)
            
            log_w = self._log_likelihood(fitness, rmse, 0.0, 0.0)
            hyps.append(LocalizationHypothesis(
                pose=pose,
                vertex_id=cid,
                log_weight=log_w,
            ))
            
        return self._dedup(hyps)

    def _log_likelihood(
        self,
        fitness: float,
        rmse: float,
        corr_xy: float,
        corr_th: float,
    ) -> float:
        if fitness <= 0.0:
            return -1e9
        
        log_p_icp = np.log(max(fitness, 1e-6)) - rmse / self._rmse_sigma
        log_p_odom = (
            -(corr_xy ** 2) / (2.0 * self._odom_sigma ** 2)
            -(corr_th ** 2) / (2.0 * self._odom_ang_sigma ** 2)
        )
        
        return float(log_p_icp + log_p_odom)
    
    def _prune(self):
        if not self._hypotheses:
            return
        
        log_ws = np.array([
            h.log_weight for h in self._hypotheses
        ])
        log_ws -= log_ws.max()
        probs = np.exp(log_ws)
        
        keep = np.argsort(probs)[::-1][: self._max_hypotheses]
        keep = [int(i) for i in keep if probs[i] > self._prune_thresh]

        self._hypotheses = [self._hypotheses[i] for i in keep]

        if self._hypotheses:
            m = max(h.log_weight for h in self._hypotheses)
            for h in self._hypotheses:
                h.log_weight -= m
    
    def _dedup(self, hyps: list[LocalizationHypothesis]):
        dist_thresh = self._dup_dist
        ang_thresh = self._dup_ang
        
        result: list[LocalizationHypothesis] = []
        for h in sorted(hyps, key=lambda x: -x.log_weight):
            dup = False
            for r in result:
                d = float(np.linalg.norm(h.pose[:2] - r.pose[:2]))
                a = float(abs(normalize_angle(h.pose[2] - r.pose[2])))
                if d < dist_thresh and a < ang_thresh:
                    dup = True
                    break
            if not dup:
                result.append(h)
        return result
