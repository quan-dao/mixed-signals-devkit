from pathlib import Path
from typing import Dict, Tuple, List
import csv

import numpy as np
from pypcd4 import PointCloud

from mixedsignals.utils.geometry import make_se3, linear_interp, interpolate_yaw


TIMESTAMP_OFFSET = np.float128(1.7e18)
TIMESTMAP_MULTIPLIER = np.float128(1e9)

# TYPE ALIAS
TYPE_AGENT_NAME = str
TYPE_TIMESTAMP = np.float128


def _legacy_check_stamp(stamp: str):
    """
    Get timestamp of pointcloud from the timestamp part of the file name: 
    Filename: {agent}_{sync_time_idx}_{timestamp_nanoseconds}.{timestamp_9numbers_after_the_decimal}.pcd

    Parameter
    ---------
    stamp : {timestamp_nanoseconds}.{timestamp_9numbers_after_the_decimal}

    Return
    ------
    stamp_with_the part after the "." prepended with 0
    """
    assert stamp[-4:] != '.pcd'
    stamp_split = stamp.split(".")
    if len(stamp_split[-1]) < 9:
        stamp_split[-1] = "0" * (9 - len(stamp_split[-1])) + stamp_split[-1]
    return ".".join(stamp_split)


def read_pointcloud(pointcloud_file: Path) -> np.ndarray:
    """
    Read point cloud from file

    Parameter
    ---------
    pointcloud_file : path to point cloud file

    Return
    ------
    pc_numpy : (N, 4) - x, y, z, intensity
    """
    pc = PointCloud.from_path(pointcloud_file)
    return pc.numpy(('x', 'y', 'z', 'intensity'))


def read_odometry_file(path_odom_file: Path) -> Dict:
    """
    Parse csv file storing agent's odometry into dict
    """
    assert path_odom_file.exists(), f"{str(path_odom_file)} does not exist"

    odom_data = {}
    header_idx_map = {}
    with open(path_odom_file, mode ='r')as file:
        odo_reader = csv.reader(file)
        for odom_time_idx, lines in enumerate(odo_reader):
            if odom_time_idx == 0:  # header line
                for j, header_title in enumerate(lines):
                    odom_data[header_title] = []
                    header_idx_map[j] = header_title
            else:  # datas
                for j, data in enumerate(lines):
                    odom_data[header_idx_map[j]].append(data)
    return odom_data


class SequenceOdomAgent(object):
    def __init__(self, dataset_root: Path, seq_index: int, agent_name: str) -> None:
        if not isinstance(dataset_root, Path):
            dataset_root = Path(dataset_root)
        
        odom_file = dataset_root / "Odometry" / f"mini_{seq_index}" / f'odometry_{agent_name}.csv'
        odom_data = read_odometry_file(odom_file)
        len_odom = len(odom_data['field.header.stamp'])

        self.all_position = list()
        self.all_orientation = list()
        self.all_timestamp = list()

        for odom_time_idx in range(len_odom):
            ts = np.uint64(odom_data['field.header.stamp'][odom_time_idx]) - TIMESTAMP_OFFSET

            translation = np.array([
                odom_data['field.pose.pose.position.x'][odom_time_idx],
                odom_data['field.pose.pose.position.y'][odom_time_idx],
                odom_data['field.pose.pose.position.z'][odom_time_idx]
            ])
            
            quaternion = np.array([
                odom_data['field.pose.pose.orientation.w'][odom_time_idx],
                odom_data['field.pose.pose.orientation.x'][odom_time_idx],
                odom_data['field.pose.pose.orientation.y'][odom_time_idx],
                odom_data['field.pose.pose.orientation.z'][odom_time_idx]
            ])

            self.all_timestamp.append(ts)
            self.all_position.append(translation)
            self.all_orientation.append(quaternion)

        # sanity check
        sorted_ids = np.argsort(self.all_timestamp)
        assert np.all(sorted_ids == np.arange(len_odom))

        self._list_time_idx_offset = np.array([0, 1, -1])

    def _find_indices_odom(self, query_timestamp: np.float128, debug: bool = False) -> int:
        odom_idx = np.searchsorted(self.all_timestamp, query_timestamp)
        if odom_idx == 0 or odom_idx == len(self.all_timestamp):
            raise ValueError("query_timestamp is outside of odometry range")

        left_ts_diff = np.abs(self.all_timestamp[odom_idx - 1] - query_timestamp)
        right_ts_diff = np.abs(self.all_timestamp[odom_idx] - query_timestamp)

        if left_ts_diff < right_ts_diff:
            if debug:
                print(f"\tDEBUG | ts_diff: {left_ts_diff}")
            return odom_idx - 1
        else:
            if debug:
                print(f"\tDEBUG | ts_diff: {right_ts_diff}")
            return odom_idx
        
    def return_map_se3_agent(self, query_timestamp: np.float128, debug: bool = False) -> np.ndarray:
        odom_idx = self._find_indices_odom(query_timestamp, debug)
        return make_se3(self.all_position[odom_idx], 
                        quaternion=self.all_orientation[odom_idx])

    @staticmethod
    def check_agent_exist_in_seq(dataset_root: Path, seq_index: int, agent_name: str) -> bool:
        odom_file = dataset_root / "Odometry" / f"mini_{seq_index}" / f'odometry_{agent_name}.csv'
        if odom_file.exists():
            odom_data = read_odometry_file(odom_file)
            return len(odom_data) > 0
        else:
            return False


class SequenceLabeledFrames(object):
    def __init__(self, seq_index: int, seq_frames: List[dict]) -> None:
        self.seq_index = seq_index
        self.num_labeled_frames = len(seq_frames)

        self.labeled_frames_timestamp: List[TYPE_TIMESTAMP] = list()
        self.labeled_frames_gt: List[np.ndarray] = list()

        for frame_dict in seq_frames:
            # ---
            # timestamp
            label_timestamp = np.uint64(frame_dict['timestamp']) - TIMESTAMP_OFFSET
            self.labeled_frames_timestamp.append(label_timestamp)

            # ---
            # gt_boxes
            gt_boxes_in_map = list()
            for anno in frame_dict['annotations']:
                box = np.array([
                    anno['position']['x'], 
                    anno['position']['y'], 
                    anno['position']['z'],

                    anno['dimensions']['x'],
                    anno['dimensions']['y'],
                    anno['dimensions']['z'],

                    anno['yaw'],

                    anno['category_id'],
                    anno['track_id'],
                ]).astype(np.float32)
                gt_boxes_in_map.append(box)
            # ---
            gt_boxes_in_map = np.stack(gt_boxes_in_map, axis=0)

            self.labeled_frames_gt.append(gt_boxes_in_map)

        # sanity check timestamp of labeled frame
        sorted_ids = np.argsort(self.labeled_frames_timestamp)
        assert np.all(sorted_ids == np.arange(self.num_labeled_frames))
    
    def return_labeled_frames_timestamp(self) -> List[TYPE_TIMESTAMP]:
        return self.labeled_frames_timestamp

    def _find_indices_annotated_frames(self, query_timestamp: np.float128) -> Tuple[int]:
        labeled_idx = np.searchsorted(self.labeled_frames_timestamp, query_timestamp)

        if labeled_idx == 0 and query_timestamp < self.labeled_frames_timestamp[0]:
            # extreme left
            return -100, -100  # -100 is the dummy value
        
        if labeled_idx == len(self.labeled_frames_timestamp):
            # extreme right
            return -100, -100  # -100 is the dummy value
        
        if np.abs(query_timestamp - self.labeled_frames_timestamp[labeled_idx]) < 1e-3:
            # a keyframe because the query timestamp matches 
            # perfectly with a labeled timestamp
            return -100, labeled_idx
        else:
            return labeled_idx - 1, labeled_idx
    
    def return_annos(self, timestamp: np.float128, debug: bool = False) -> np.ndarray:
        """
        Return annotation of a query timestamp 
        (might be timestamp of labeled_frame or might not)

        Parameter
        ---------
        timestamp:
            query timestamp comes from timestamp of pc of an agent
        
        debug:
            if True, print keyframe (i.e., is_labeled) status of timestamp

        Return
        ------
        gt_boxes:
            (N, 7+2) - [x, y, z, l, w, h, yaw, category_id, track_id]
        """
        labeled_ids = self._find_indices_annotated_frames(timestamp)

        # check if this timestamp is annotated
        if labeled_ids[0] < 0 and labeled_ids[1] < 0:
            if debug:
                print(f'DEBUG | {timestamp} is not labeled')
            return np.zeros((0, 9))
        
        # check if this timestamp is a keyframe
        if labeled_ids[0] < 0 and labeled_ids[1] >= 0:
            if debug:
                print(f"DEBUG | {timestamp} is a keyframe")
                print(f"DEBUG | gt_boxes.shape: {self.labeled_frames_gt[labeled_ids[1]].shape}")
            return self.labeled_frames_gt[labeled_ids[1]]
        
        if debug:
            print(f'DEBUG | {timestamp} is inside the labeled range, but not a keyframe')

        # -------------------------------
        # interpolate gt_boxes in frames 
        # -------------------------------
        prev_gt = self.labeled_frames_gt[labeled_ids[0]]
        next_gt = self.labeled_frames_gt[labeled_ids[1]]

        # only interpolate on objects that present in both prev & next frame
        common_gt = dict()
        """
        {
          track_id_1: (pose_t-1, pose_t),
          track_id_2: (pose_t-1, pose_t),
          ...
          track_id_N: (pose_t-1, pose_t)
        }
        """
        
        for prev_box in prev_gt:
            _track_id = int(prev_box[-1])
            common_gt[_track_id] = [prev_box,]
        
        for next_box in next_gt:
            _track_id = int(next_box[-1])
            if _track_id in common_gt:
                common_gt[_track_id].append(next_box)

        interp_boxes = list()

        t_start = self.labeled_frames_timestamp[labeled_ids[0]]
        t_end = self.labeled_frames_timestamp[labeled_ids[1]]

        for _track_id, _item in common_gt.items():
            if len(_item) < 2:
                # this _track_id doesn't have 2 items
                # --> the obj doesn't present in the next frame
                continue

            prev_box, next_box = _item
            x = linear_interp(timestamp, prev_box[0], next_box[0], t_start, t_end)
            y = linear_interp(timestamp, prev_box[1], next_box[1], t_start, t_end)
            z = linear_interp(timestamp, prev_box[2], next_box[2], t_start, t_end)
            yaw = interpolate_yaw(timestamp, prev_box[6], next_box[6], t_start, t_end)

            box = [x, y, z, 
                   *next_box[3: 6].tolist(),  # dimension
                   yaw, 
                   next_box[7],  # category_id
                   next_box[8],  # track_id
            ]
            
            interp_boxes.append(box)

        interp_boxes = np.stack(interp_boxes)
        if debug:
            print(f"DEBUG | gt_boxes: {interp_boxes.shape}")
        return interp_boxes.astype(np.float32)
    

def find_available_sequences(dataset_root: Path) -> List[int]:
    """
    Find the name of available sequences in the dataset root

    Return
    ------
    avail_seq_pcs:
        sorted list of name of sequences (e.g., [11, 18])
    """
    data_dirs = [f for f in dataset_root.iterdir() if f.is_dir()]
    avail_seq_pcs = set()
    avail_seq_odom = set()
    for directory_data in data_dirs:
        data_type = directory_data.parts[-1]
        if data_type == 'PointClouds':
            save_to = avail_seq_pcs
        elif data_type == 'Odometry':
            save_to = avail_seq_odom
        else:
            RuntimeWarning(f'{directory_data} is unknown')
    
        for directory_seq in directory_data.iterdir():
            if directory_seq.is_dir():
                seq_name = int(directory_seq.parts[-1].split('_')[1])
                save_to.add(seq_name)
    
    assert avail_seq_pcs == avail_seq_odom, f"{avail_seq_pcs} != {avail_seq_odom}"
    avail_seq_pcs = sorted([int(seq_name) for seq_name in avail_seq_pcs])
    return avail_seq_pcs


class SequencePointClouds(object):
    def __init__(self, dataset_root: Path, seq_index: int):
        self.pc_dir = dataset_root / "PointClouds" / f"mini_{seq_index}"

        self.agents_pc_files: Dict[TYPE_AGENT_NAME, List[Path]] = dict()
        self.agents_timestamp: Dict[TYPE_AGENT_NAME, List[TYPE_TIMESTAMP]] = dict()
        self._parse_pointcloud_dir_()  # populate 2 dict above

    def _parse_pointcloud_dir_(self) -> None:
        for path_pc_file in self.pc_dir.glob('*.pcd'):
            name_pc_file = str(path_pc_file.parts[-1])[:-4]  # ":-4" to exclude ".pcd"
            agent_name = name_pc_file.split('_')[0]

            if agent_name not in self.agents_pc_files:
                self.agents_pc_files[agent_name] = list()

                assert agent_name not in self.agents_timestamp, \
                    f"{agent_name} appear in self.agents_timestamp before added to self.agents_pc_files"
                
                self.agents_timestamp[agent_name] = list()

            self.agents_pc_files[agent_name].append(path_pc_file)

            timestamp = np.float128(_legacy_check_stamp(name_pc_file.split('_')[-1])) * TIMESTMAP_MULTIPLIER \
                - TIMESTAMP_OFFSET
            self.agents_timestamp[agent_name].append(timestamp)

        # sort timestamp in ascending order
        for agent_name, agent_all_ts in self.agents_timestamp.items():
            sorted_ids = np.argsort(agent_all_ts)  # ascending
            # apply sorted_ids to 2 lists self.agents_timestamp & self.agents_pc_files
            self.agents_timestamp[agent_name] = [
                self.agents_timestamp[agent_name][i] for i in sorted_ids
            ]

            self.agents_pc_files[agent_name] = [
                self.agents_pc_files[agent_name][i] for i in sorted_ids
            ]
        
        # sanity check: every agent has the same number of pointclouds
        seq_length = -1
        for agent_name, agent_timestamp_list in self.agents_timestamp.items():
            if seq_length < 0:
                seq_length = len(agent_timestamp_list)
            else:
                assert len(agent_timestamp_list) == seq_length, \
                    f"{agent_name} has {len(agent_timestamp_list)} != {seq_length}"
        
        return

    def return_point_cloud_of_agent(self, 
                                    agent_name: TYPE_AGENT_NAME,
                                    sync_time_idx: int,
                                    debug: bool = False) -> \
            Tuple[np.ndarray, TYPE_TIMESTAMP]:
        """
        Parameters:
        ==========
        agent_name :

        sync_time_idx : contiguous integer going from 0 to 299 (vary depends on seq len)
        """
        if debug:
            print(f"DBUG | {self.agents_pc_files[agent_name][sync_time_idx]}")

        pc = read_pointcloud(self.agents_pc_files[agent_name][sync_time_idx])
        timestamp_of_pc = self.agents_timestamp[agent_name][sync_time_idx]
        return pc, timestamp_of_pc
    
    def return_all_timestamps_of_agent(self, agent_name: TYPE_AGENT_NAME) -> List[TYPE_TIMESTAMP]:
        return self.agents_timestamp[agent_name]
    

AGENT_COLOR = {
    '003': np.array([255, 59, 48]) / 255.,
    '004': np.array([137, 68, 171]) / 255.,
    'laser': np.array([255, 204, 0]) / 255.,
    'top': np.array([0, 122, 255]) / 255.,
    'dome': np.array([36, 138, 61]) / 255.,
}

VIEW_POINT = """{
        "class_name" : "ViewTrajectory",
        "interval" : 29,
        "is_loop" : false,
        "trajectory" : 
        [
            {
                "boundingbox_max" : [ 83.23907470703125, 127.98077392578125, 15.209506034851074 ],
                "boundingbox_min" : [ -202.80641174316406, -107.07141876220703, -13.27166748046875 ],
                "field_of_view" : 60.0,
                "front" : [ 0.0, 0.0, 1.0 ],
                "lookat" : [ 36.473745825447239, -10.824860529514885, 6.4467616081237793 ],
                "up" : [ 0.0, 1.0, 0.0 ],
                "zoom" : 0.1799999999999996
            }
        ],
        "version_major" : 1,
        "version_minor" : 0
    }"""


VIEW_POINT_TRACKING = """{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 83.23907470703125, 127.98077392578125, 15.209506034851074 ],
			"boundingbox_min" : [ -202.80641174316406, -107.07141876220703, -13.27166748046875 ],
			"field_of_view" : 60.0,
			"front" : [ 0.0, 0.0, 1.0 ],
			"lookat" : [ 36.473745825447239, -10.824860529514885, 6.4467616081237793 ],
			"up" : [ 0.0, 1.0, 0.0 ],
			"zoom" : 0.25999999999999956
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}"""
