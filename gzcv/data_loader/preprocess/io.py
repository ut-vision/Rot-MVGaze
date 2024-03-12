import copy
import csv
import os
from typing import Any, Dict

import cv2
import h5py
import numpy as np
from scipy.io import \
    loadmat as scipy_loadmat  # scipy.io insn't a regular package.

from gzcv.utils.xml import parse_xml


class UnpackHdf(object):
    def __init__(self):
        super().__init__()

    def __call__(self, idx_data):
        data = copy.deepcopy(idx_data)
        hdf_path = data.pop("hdf_path")
        img_idx = data.pop("img_index")
        with h5py.File(hdf_path, swmr=True) as hdf:
            for key in hdf.keys():
                if not "index" in key:
                    data[key] = hdf[key][img_idx]
        return data


class OpenImage(object):
    def __init__(self, path_key="img_path"):
        self.path_key = path_key

    def __call__(self, data):
        img_path = data[self.path_key]
        img = cv2.imread(img_path)
        data["img"] = img
        return data


class ReadMPIIAnnotation(object):
    def __init__(self):
        self.annos_subj = {}

    def __call__(self, data):
        anno_path = data["annotation_path"]
        if anno_path not in self.annos_subj:
            self.annos_subj[anno_path] = self.organize_csv(anno_path)
        img_path = data["img_path"]
        img_name = os.path.basename(img_path)
        day_idx = data["day_index"]
        img_key = os.path.join(day_idx, img_name)

        raw_anno = self.annos_subj[anno_path][img_key]
        anno = self.extract_anno(raw_anno)
        data.update(anno)
        return data

    @staticmethod
    def organize_csv(anno_path):
        annos = {}
        with open(anno_path, "r") as af:
            csv_annos = csv.reader(af, delimiter=" ")
            for row in csv_annos:
                img_key = row[0]
                anno = row[1:]
                annos[img_key] = anno
        return annos

    def extract_anno(self, raw_anno):
        landmarks = np.array(raw_anno[2:14], dtype="f").reshape(-1, 2)
        gaze_target3d = np.array(raw_anno[23:26], dtype="f").reshape((3, 1))
        return {"landmarks": landmarks, "gaze_target3d": gaze_target3d}


class AppendObject3D(object):
    def __init__(self, root_orig, root_3d):
        self.root_orig = root_orig
        self.root_3d = root_3d

    def __call__(self, data):
        img_path = data["img_path"]
        non_ext_path, _ = os.path.splitext(img_path)
        obj_path = non_ext_path.replace(self.root_orig, self.root_3d) + ".obj"
        landmarks3d = np.loadtxt(obj_path.replace(".obj", "_lm.txt"))
        crop_params = np.loadtxt(obj_path.replace(".obj", "_crop_params.txt"))
        vert, face, color = self.load_obj(obj_path)
        reconstruction = {
            "landmarks3d": landmarks3d,
            "crop_params": crop_params,
            "vert": vert,
            "face": face,
            "color": color,
        }
        data.update(reconstruction)
        return data

    @staticmethod
    def load_obj(obj_path):
        vertices = []
        colors = []
        faces = []
        with open(obj_path, "r") as f:
            lines = csv.reader(f, delimiter=" ")
            for line in lines:
                if len(line) == 0:
                    continue
                elif line[0] == "v":
                    vertices.append([line[1:4]])
                    colors.append([line[4:7]])
                elif line[0] == "f":
                    faces.append(line[1:4])
        vertices = np.vstack(vertices).astype(np.float32)
        colors = np.vstack(colors).astype(np.float32)
        faces = np.vstack(faces).astype(np.int32) - 1
        return vertices, faces, colors


class AppendMPIICalib(object):
    def __init__(self):
        pass

    def __call__(self, data):
        calib_path = data["calib_path"]
        camera = scipy_loadmat(os.path.join(calib_path, "Camera.mat"))
        data.update(camera)
        return data


class AppendXGazeCalib(object):
    def __init__(self, path_pattern, num_cameras: int = 18):
        self.calibs = []
        # Do not use glob without sorting. Glob does not ensure its order.
        for i in range(num_cameras):
            path = path_pattern.format(str(i).zfill(2))
            calib = parse_xml(path)
            self.calibs.append(calib)

    def __call__(self, data):
        cam_idx = data["cam_index"]
        calib = self.calibs[cam_idx]
        data.update(calib)
        return data


class RefineRotation(object):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __call__(self, data):
        subj_idx = os.path.basename(data["subject_index"])
        cam_idx = data["cam_index"]
        with h5py.File(self.path, "r", swmr=True) as h5f:
            data["rot"] = h5f[subj_idx][cam_idx]
        return data


class AppendLandmarks:
    def __init__(self, path: str) -> None:
        self._path = path

    def __call__(self, data: Dict[str, Any]):
        unique_id = data["id"]
        with h5py.File(self._path, swmr=True) as h5f:
            landmarks = h5f[unique_id]
            data["landmarks"] = landmarks[:]  # Slicing triggers the actual read.
        return data


class DeepCopy:
    def __init__(self, src_key: str, dst_key: str) -> None:
        self._src_key = src_key
        self._dst_key = dst_key

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        orig = data[self._src_key]
        data[self._dst_key] = copy.deepcopy(orig)
        return data
