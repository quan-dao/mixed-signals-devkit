import argparse
import numpy as np

from mixedsignals.mixed_signals import MixedSignalsExplorer
from mixedsignals.utils.mixed_signals_utils import AGENT_COLOR as agent_color
from mixedsignals.utils.mixed_signals_utils import VIEW_POINT as view_point
from mixedsignals.utils.geometry import apply_se3_, apply_se3
from mixedsignals.utils.o3d_viz_utils import PointPainter


def main(dataset_root: str,
         chosen_sequence_index: int = 18, 
         labeled_frame_idx: int = 0):
    
    print(f'showing aggregate pc of sequence {chosen_sequence_index} in TOP frame')

    msig = MixedSignalsExplorer(dataset_root, verbose=True)
    seq_exist_cavs = msig.return_name_cavs_in_seq(chosen_sequence_index)
    print(f"name CAVs in sequecne {chosen_sequence_index}: {seq_exist_cavs}")

    seq_labeled_sync_time_ids = msig.return_labeled_sync_time_ids_of_seq(chosen_sequence_index)

    sync_time_idx = seq_labeled_sync_time_ids[labeled_frame_idx]
    
    top_se3_map = msig.top_se3_map
    
    painter = PointPainter()

    # ===
    for agent_name in msig.name_rsu_lidars + seq_exist_cavs:
        # get point cloud in agent's body frame
        agent_pc, agent_timestamp = \
            msig.return_agent_point_cloud(chosen_sequence_index, agent_name, sync_time_idx)

        # transform point cloud from agent's body frame to `TOP`
        map_se3_agent = msig.return_map_se3_agent(chosen_sequence_index, 
                                                    agent_name, 
                                                    agent_timestamp)            
        apply_se3_(top_se3_map @ map_se3_agent, agent_pc)

        # add point cloud to visualization
        painter.add_pointclouds_(agent_pc, agent_color[agent_name])

    # === 
    # get annotation in `MAP` frame
    gt_boxes_in_map = msig.return_gt_boxes_in_map(chosen_sequence_index, sync_time_idx)
    gt_boxes_in_top = apply_se3(top_se3_map, boxes_=gt_boxes_in_map)

    # add gt_boxes to visualization
    painter.add_boxes_(gt_boxes_in_top, np.zeros(3))

    # ===
    painter.show(view_points=view_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize 1 early fusion point cloud & ground truth")
    parser.add_argument('--dataset_root', type=str, default='data/mixed-signals-mini', 
                        help='index of the sequence chosen for visualization')
    parser.add_argument('--seq_idx', type=int, default=20, 
                        help='index of the sequence chosen for visualization')
    parser.add_argument('--time_idx', type=int, default=0, 
                        help='index of time step of the sequence chosen for visualization')
    args = parser.parse_args()
    main(args.dataset_root, args.seq_idx, args.time_idx)
