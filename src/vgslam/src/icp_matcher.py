from typing import Literal
import numpy as np
import small_gicp

from .models import PositionedCloud
from .helpers import relative_pose_4x4, evaluate_registration, transform_4x4_to_2d_pose

ICP_MATCHER_MODE = Literal['ICP', 'GICP', 'PLANE_ICP']


class ICPMatcher:
    def __init__(
        self,
        matcher_max_correspondence_distance: float = 0.5,
        matcher_mode: ICP_MATCHER_MODE = "GICP",
    ) -> None:
        self._matcher_max_corr_distance = matcher_max_correspondence_distance
        self._matcher_mode = matcher_mode
    
    def match(self, source: PositionedCloud, target: PositionedCloud, rel_override: np.ndarray | None = None):
        """
        returns T_target_source, (fitness, inlier_rmse)
        """
        
        # grab scan data
        pcd_src = source.gicp
        pcd_tgt = target.gicp
        tree = target.kdtree
        
        # perform icp scan match
        result = small_gicp.align(
            pcd_tgt, pcd_src, tree,
            
            # use odometry initial transform guess or given transform override
            relative_pose_4x4(target.pos, source.pos) if rel_override is None else rel_override,
            registration_type=self._matcher_mode,
            max_correspondence_distance=self._matcher_max_corr_distance,
            verbose=False
        )
        T_target_source = result.T_target_source

        # calculate transform accuracy metrics
        fitness, inlier_rmse = evaluate_registration(
            source.cloud,
            tree,
            T_target_source,
            max_correspondence_distance=self._matcher_max_corr_distance
        )
        
        return T_target_source, (fitness, inlier_rmse)
    
    
    def find_best_match(
        self, 
        sources: list[PositionedCloud], 
        target: PositionedCloud, 
        min_confidence = 0.9, 
        rmse_max = 0.1
    ) -> tuple[int, np.ndarray | None, tuple]:
        best_score = -float('inf')
        best = (-1, None, (0, 0))

        # check every source candidate to match the target
        # and return best match
        for i, source in enumerate(sources):
            T_new, (fitness, rmse) = self.match(source, target)

            if fitness < min_confidence or rmse > rmse_max:
                continue
            
            score = fitness - 0.5 * rmse

            if best is None or score > best_score:
                best_score = score
                best = (i, T_new, (fitness, rmse))
        
        i, T, stats = best
        if i > -1:
            return (i, transform_4x4_to_2d_pose(T), stats)
    
        else:
            return best
