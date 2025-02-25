import argparse
import time
import open3d as o3d
from matplotlib import cm
import numpy as np

from mixedsignals.mixed_signals import MixedSignalsExplorer
from mixedsignals.utils.mixed_signals_utils import VIEW_POINT_TRACKING as view_point
from mixedsignals.utils.geometry import apply_se3_
from mixedsignals.utils.o3d_viz_utils import PointPainter
from mixedsignals.utils.bbox_utils import get_boxes_vertices_coord


def main(chosen_sequence_index: int = 20, 
         len_traj: int = 2):
    print(f'showing aggregate pc of sequence {chosen_sequence_index} in TOP frame')

    msig = MixedSignalsExplorer('data/mixed-signals-mini', verbose=False)
    seq_exist_cavs = msig.return_name_cavs_in_seq(chosen_sequence_index)
    print(f"name CAVs in sequecne {chosen_sequence_index}: {seq_exist_cavs}")

    top_se3_map = msig.top_se3_map

    seq_labeled_sync_time_ids = msig.return_labeled_sync_time_ids_of_seq(chosen_sequence_index)
    print("seq_labeled_sync_time_ids: ", seq_labeled_sync_time_ids)
    
    # ===
    # set up visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.clear_geometries()
    
    o3d_pc = o3d.geometry.PointCloud()
    o3d_boxes = [o3d.geometry.LineSet() for _ in range (1000)]
    
    # register placeholder of pc & boxes with the visualization
    vis.add_geometry(o3d_pc)

    for b in o3d_boxes:
        vis.add_geometry(b)

    time_colors = cm.rainbow(np.linspace(0, 1, 10))[::-1, :3]
    # past       ==========> presence
    # cold color ==========> hot color
    trk_id_to_contiguous_id = dict()

    # ===
    # main loop
    ii = 0
    for sync_time_idx in seq_labeled_sync_time_ids:
        print('sync_time_idx: ', sync_time_idx)
        agg_pc = list()
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

            agg_pc.append(agent_pc)

        agg_pc = np.concatenate(agg_pc)

        # ===
        dict_tracks = msig.return_tracks_traj(chosen_sequence_index, 
                                         sync_time_idx, 
                                         len_traj=1 if ii < len_traj else len_traj, 
                                         return_in_top_frame=True)
        # === 
        # viz
        o3d_pc.points = o3d.utility.Vector3dVector(agg_pc[:, :3])
        o3d_pc.colors = o3d.utility.Vector3dVector(np.zeros((agg_pc.shape[0], 3)) + 0.44)
        
        vis.update_geometry(o3d_pc)

        # show tracks
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # front
            [4, 5], [5, 6], [6, 7], [7, 4],  # back
            [0, 4], [1, 5], [2, 6], [3, 7],  # connecting front & back
            [0, 2], [1, 3]  # denote forward face
        ]

        # remove track that are no longer visible
        for trk_id in trk_id_to_contiguous_id.keys():
            if trk_id not in dict_tracks:
                for b_ii in range(len_traj):
                    idx_in_o3d_boxes = trk_id_to_contiguous_id[trk_id] * len_traj + b_ii
                    o3d_boxes[idx_in_o3d_boxes].clear()    


        for trk_id, track_boxes in dict_tracks.items():
            if trk_id not in trk_id_to_contiguous_id:
                trk_id_to_contiguous_id[trk_id] = len(trk_id_to_contiguous_id)
            
            print(f"trk_id {trk_id} | track_boxes: {track_boxes.shape}, cont id {trk_id_to_contiguous_id[trk_id]}")

            for b_ii, box in enumerate(track_boxes):
                vers = get_boxes_vertices_coord(box.reshape(1, -1))[0]
                idx_in_o3d_boxes = trk_id_to_contiguous_id[trk_id] * len_traj + b_ii

                o3d_boxes[idx_in_o3d_boxes].lines = o3d.utility.Vector2iVector(lines)
                o3d_boxes[idx_in_o3d_boxes].points = o3d.utility.Vector3dVector(vers)
                o3d_boxes[idx_in_o3d_boxes].colors = \
                    o3d.utility.Vector3dVector(np.zeros((14, 3)) + time_colors[int(box[-1])])

                vis.update_geometry(o3d_boxes[idx_in_o3d_boxes])


        vis.set_view_status(view_point)
        vis.get_render_option().point_size = 0.5
        vis.poll_events()
        vis.update_renderer()
        time.sleep(1.0)

        ii += 1

    vis.destroy_window()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="visualize track")
    parser.add_argument('--seq_idx', type=int, default=20, 
                        help='index of the sequence chosen for visualization')
    
    parser.add_argument('--len_traj', type=int, default=5, 
                        help='number of boxes inisde each traj to be shown')
    args = parser.parse_args()
    main(args.seq_idx, args.len_traj)

