import json
import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data.dataloader import default_collate

from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import common.data_utils as data_utils
import common.rot as rot
import common.transforms as tf
import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
from common.object_tensors import ObjectTensors
from src.datasets.dataset_utils import get_valid, pad_jts2d
from scripts_data.mod_visualizer import DataViewer


class SeqArcticDataset(Dataset):
    def __init__(self, args, split, seq=None):
        args.img_res = args.img_res[0]
        self._load_data(args, split, seq)
        self._process_imgnames(seq, split)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )

        self.seqs = list(set(['/'.join(x.split('/')[:-2]) for x in self.imgnames]))
        self.seqs = [x.replace('arctic_data/data/images', 'outputs/processed_verts/seqs') + '.npy' for x in self.seqs]
        self.viewer = DataViewer(interactive=False, size=(2024, 2024))

        with open(f"/home/relh/Code/hand_trajectories/third_party/arctic/data/arctic_data/data/meta/misc.json", "r") as f:
            self.subject_meta = json.load(f)


    def __len__(self):
        return len(self.seqs) #len(self.imgnames)


    def __getitem__(self, index):
        seq_p = self.seqs[index]
        return self.transform_collate(seq_p)


    def transform_collate(self, seq_p):
        seq_name = seq_p.split("/")[-1].split(".")[0]
        sid = seq_p.split("/")[-2]
        #out_name = f"{sid}_{seq_name}_{1}"

        cache_name = ('/home/relh/Code/hand_trajectories/cache/processed_batch/' + seq_p.split('/')[-1]).replace('.npy', '.pt')
        if op.exists(cache_name):
            return torch.load(cache_name) 
        else:
            meshes, data = self.viewer.load_data(
                seq_p,
                True, #args.mano,
                True, #args.object,
                False, #args.smplx,
                True, #args.no_image,
                True, #args.distort,
                1, #args.view_idx,
                self.subject_meta,
            )
            torch.save((meshes, data), cache_name)
        return meshes, data 


    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames


    def _load_data(self, args, split, seq):
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        # during inference, turn off
        if seq is not None:
            self.aug_data = False
        self.normalize_img = Normalize(mean=args.img_norm_mean, std=args.img_norm_std)

        if "train" in split:
            self.mode = "train"
        elif "val" in split:
            self.mode = "val"
        elif "test" in split:
            self.mode = "test"

        short_split = split.replace("mini", "").replace("tiny", "").replace("small", "")
        data_p = op.join(
            f"./data/arctic_data/data/splits/{args.setup}_{short_split}.npy"
        )
        logger.info(f"Loading {data_p}")
        data = np.load(data_p, allow_pickle=True).item()

        self.data = data["data_dict"]
        self.imgnames = data["imgnames"]

        with open("./data/arctic_data/data/meta/misc.json", "r") as f:
            misc = json.load(f)

        # unpack
        subjects = list(misc.keys())
        intris_mat = {}
        world2cam = {}
        image_sizes = {}
        ioi_offset = {}
        for subject in subjects:
            world2cam[subject] = misc[subject]["world2cam"]
            intris_mat[subject] = misc[subject]["intris_mat"]
            image_sizes[subject] = misc[subject]["image_size"]
            ioi_offset[subject] = misc[subject]["ioi_offset"]

        self.world2cam = world2cam
        self.intris_mat = intris_mat
        self.image_sizes = image_sizes
        self.ioi_offset = ioi_offset

        object_tensors = ObjectTensors()
        self.kp3d_cano = object_tensors.obj_tensors["kp_bottom"]
        self.obj_names = object_tensors.obj_tensors["names"]
        self.egocam_k = None
