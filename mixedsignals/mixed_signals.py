import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

from mixedsignals.utils.geometry import apply_se3_
from mixedsignals.utils.mixed_signals_utils import (
    SequenceOdomAgent, SequenceLabeledFrames, find_available_sequences,
    SequencePointClouds,
    TYPE_AGENT_NAME, 
    TYPE_TIMESTAMP
)


# TYPE alias
TYPE_SEQ_INDEX = int


class MixedSignalsExplorer(object):
    def __init__(self, dataset_root: str, verbose: bool = False) -> None:
        self.root = Path(dataset_root)
        self.verbose = verbose

        self.name_vehicles = ['laser', '003', '004']
        self.name_rsu_lidars = ['top', 'dome']
        self.name_agents = self.name_rsu_lidars + self.name_rsu_lidars
        self.dict_agent_indicator = dict(zip(self.name_agents, range(len(self.name_agents))))

        self.available_sequences_index = find_available_sequences(self.root)
        if self.verbose:
            print(f"MixedSignalsExplorer | available_sequences_index: {self.available_sequences_index}")

        self.label_generators  = self._make_label_generator_for_avail_seqs()

        self.odom_generators, self.dict_exist_cavs = self._make_odom_generator_for_avail_seqs()

        self.pc_generators = self._make_pc_generator_for_avail_seqs()

        # === pose of RSU
        self.map_se3_dome = np.array([
            [1.0, 0.0, 0.0, -41.507],
            [0.0, 1.0, 0.0, -51.864],
            [0.0, 0.0, 1.0, -1.340],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.dome_se3_map = np.linalg.inv(self.map_se3_dome)

        self.map_se3_top = np.array([
            [1.0, 0.0, 0.0, -41.551],
            [0.0, 1.0, 0.0, -51.878],
            [0.0, 0.0, 1.0, -1.077],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.top_se3_map = np.linalg.inv(self.map_se3_top)

        # === other constant
        self.time_anchor_agent = 'top'

    def _make_label_generator_for_avail_seqs(self) -> Dict[TYPE_SEQ_INDEX, SequenceLabeledFrames]:

        label_file = self.root / "V2X_dataset-v0.4-labels.json"
        assert label_file.exists(), f"{label_file} does not exist"
        with open(label_file, 'r') as f:
            raw_infos = json.load(f)
        all_samples = raw_infos['dataset']['samples']  # list

        dict_label_generators = dict()

        for idx in range(len(all_samples)):
            seq_name = all_samples[idx]['name']  # point_cloud_{seq_index}
            seq_index = int(seq_name.split('_')[-1])
            if seq_index in self.available_sequences_index:
                dict_label_generators[seq_index] = SequenceLabeledFrames(
                    seq_index,
                    all_samples[idx]['labels']['ground-truth']['attributes']['frames']
                )

        assert len(dict_label_generators) == len(self.available_sequences_index), \
            f"{len(dict_label_generators)} != len({self.available_sequences_index})"
        
        return dict_label_generators
    
    def _make_odom_generator_for_avail_seqs(self) -> Tuple[
            Dict[TYPE_SEQ_INDEX, Dict[TYPE_AGENT_NAME, SequenceOdomAgent]],
            Dict[TYPE_SEQ_INDEX, List[TYPE_AGENT_NAME]]  # exist_cavs
        ]:
        """
        {
            seq_idx (e.g., 4) : {
                'laser': SequenceOdomAgent,
                '003': SequenceOdomAgent,
                '004': SequenceOdomAgent,
            }
        }
        """
        odom_generators = dict()
        dict_exist_cavs = dict()
        for seq_idx in self.available_sequences_index:
            # init output of this sequence
            odom_generators[seq_idx] = dict()
            dict_exist_cavs[seq_idx] = list()

            for cav_name in self.name_vehicles:
                if SequenceOdomAgent.check_agent_exist_in_seq(self.root, seq_idx, cav_name):
                    dict_exist_cavs[seq_idx].append(cav_name)
                    odom_generators[seq_idx][cav_name] = SequenceOdomAgent(self.root, seq_idx, cav_name)

            if self.verbose:
                exist_cav = dict_exist_cavs[seq_idx]
                print(f"MixedSignalsExplorer | seq_idx {seq_idx} has {len(exist_cav)} CAVs : {exist_cav}")

        return odom_generators, dict_exist_cavs
    
    def _make_pc_generator_for_avail_seqs(self) -> Dict[TYPE_SEQ_INDEX, SequencePointClouds]:
        pc_generators = dict()
        for seq_idx in self.available_sequences_index:
            pc_generators[seq_idx] = SequencePointClouds(self.root, seq_idx)
        return pc_generators

    def return_agent_point_cloud(self, 
                                 seq_idx: TYPE_SEQ_INDEX, 
                                 agent_name: TYPE_AGENT_NAME, 
                                 sync_time_idx: int) -> Tuple[np.ndarray, TYPE_TIMESTAMP]:
        """
        Return point cloud of agent @ sync_time_idx in agent's body frame

        Return:
        ======
            pc: (N, 4) - x, y, z, intensity in agent's body frame
            timestamp:
        """
        pc, timestamp = \
            self.pc_generators[seq_idx].return_point_cloud_of_agent(agent_name, sync_time_idx, self.verbose)
        
        if agent_name == 'top':
            # pc is in `map` frame -> need to map it back to agent's frame
            apply_se3_(self.top_se3_map, pc)
        elif agent_name == 'dome':
            # pc is in `map` frame -> need to map it back to agent's frame
            apply_se3_(self.dome_se3_map, pc)

        return pc, timestamp

    def return_map_se3_agent(self, 
                             seq_idx: TYPE_SEQ_INDEX, 
                             agent_name: TYPE_AGENT_NAME, 
                             query_timestmap: TYPE_TIMESTAMP):
        if agent_name == 'top':
            return self.map_se3_top
        elif agent_name == 'dome':
            return self.map_se3_dome
        else:
            if self.verbose:
                print(f"DEBUG | odom agent_name: {agent_name}")
            return self.odom_generators[seq_idx][agent_name].return_map_se3_agent(query_timestmap, self.verbose)

    def return_name_cavs_in_seq(self, seq_idx: TYPE_SEQ_INDEX) -> List[TYPE_AGENT_NAME]:
        return self.dict_exist_cavs[seq_idx]
    
    def return_labeled_sync_time_ids_of_seq(self, seq_idx: TYPE_SEQ_INDEX) -> np.ndarray:
        anchor_agent_timestamps = \
            self.pc_generators[seq_idx].return_all_timestamps_of_agent(self.time_anchor_agent)
        
        labeled_frames_timestamp = self.label_generators[seq_idx].return_labeled_frames_timestamp()

        min_labeled = np.searchsorted(anchor_agent_timestamps, labeled_frames_timestamp[0])
        max_labeled = np.searchsorted(anchor_agent_timestamps, labeled_frames_timestamp[-1]) - 1

        labeled_sync_time_ids = np.arange(min_labeled, max_labeled + 1)
        # this is to index to anchor_agent_timestamps
        return labeled_sync_time_ids

    def return_timestamp_for_query_gt(self, seq_idx: TYPE_SEQ_INDEX, sync_time_idx: int) -> TYPE_TIMESTAMP:
        return self.pc_generators[seq_idx].return_all_timestamps_of_agent(
            self.time_anchor_agent)[sync_time_idx]

    def is_sync_time_idx_labeled(self, seq_idx: TYPE_SEQ_INDEX, sync_time_idx: int) -> bool:
        timestamp = self.return_timestamp_for_query_gt(seq_idx, sync_time_idx)
        labeled_frames_timestamp = self.label_generators[seq_idx].return_labeled_frames_timestamp()
        return labeled_frames_timestamp[0] <= timestamp and timestamp <= labeled_frames_timestamp[-1]

    def return_gt_boxes_in_map(self, seq_idx: TYPE_SEQ_INDEX, sync_time_idx: int) -> np.ndarray:
        """
        Return:
        =======
        gt_boxes: (N, 7+2) - cx, cy, cz, dx, dy, dz, yaw, category_id, track_id
        """
        query_timestamp = self.return_timestamp_for_query_gt(seq_idx, sync_time_idx)
        return self.label_generators[seq_idx].return_annos(query_timestamp, self.verbose)

    def return_tracks_traj(self, 
                           seq_idx: int, 
                           presence_sync_time_idx: int,
                           len_traj: int = 10,
                           return_in_top_frame: bool = False) -> Dict[str, np.ndarray]:
        """
        Parameters:
        ==========
        seq_idx :

        presence_sync_time_idx : sync_time_idx of the presence (i.e., now)

        len_traj : number of frame of a track to return

        return_in_top_frame: if True, return in `TOP` instead of `MAP`

        Returns:
        =======
        dict_tracks : {trk_id0: (len_trk_id0, 7+2+1), ...}; box-7, cat_id, trk_id, time_lag
        """
        dict_tracks_boxes = dict()
        dict_tracks_dim = dict()

        for idx_time_lag in range(len_traj):
            sync_time_idx = presence_sync_time_idx - idx_time_lag
            if self.is_sync_time_idx_labeled(seq_idx, sync_time_idx):
                boxes = self.return_gt_boxes_in_map(seq_idx, 
                                                    sync_time_idx)  # (N, 7+2) - box-7, cat_id, track_id

                assert boxes.shape[0] > 0, f"seq {seq_idx}, sync_time_idx {sync_time_idx} is not labeled"

                if return_in_top_frame:
                    apply_se3_(self.top_se3_map, boxes_=boxes)

                boxes = np.pad(boxes, pad_width=[(0, 0), (0, 1)], constant_values=idx_time_lag)  # (N, 7+2+1)

                for box in boxes:
                    trk_id = int(box[-2])
                    if trk_id not in dict_tracks_boxes:
                        dict_tracks_boxes[trk_id] = list()
                        dict_tracks_dim[trk_id] = box[3: 6]
                    
                    box[3: 6] = dict_tracks_dim[trk_id]
                    dict_tracks_boxes[trk_id].append(box)

        for trk_id in dict_tracks_boxes.keys():
            dict_tracks_boxes[trk_id] = np.stack(dict_tracks_boxes[trk_id], axis=0)

        return dict_tracks_boxes
