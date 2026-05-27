#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from exact_gs.gaussian.gaussian_model import GaussianModel
from exact_gs.arguments import ModelParams
from exact_gs.utils.graphics_utils import fetchPly
from exact_gs.utils.system_utils import searchForMaxIteration


def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, load_gt=True,loaded_iter=None):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )
        assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
        gaussians.load_ply(ply_path)
        print("Loading trained model at iteration {}".format(loaded_iter))

    elif load_gt: # create vol_gt
        gt_ply_path = os.path.join(
                args.source_path,
                "gaussian",
                "point_cloud.pickle"
            )
        assert osp.exists(gt_ply_path), f"Cannot find {gt_ply_path} for loading."
        gaussians.load_ply(gt_ply_path)
       
    else:   # training initinalzation


        print(args.ply_path)
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".pickle"
                )
            else:
                raise ValueError("Could not recognize scene type!")
        else:
            ply_path = args.ply_path
           

        assert osp.exists(
            ply_path
        ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")

        
        ply_type = ply_path.split(".")[-1]
        
        if ply_type == "npy":   # for real-world data
            point_cloud = np.load(ply_path)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
            spatial_lr_scale=1.0    
            gaussians.create_from_pcd(xyz, density, spatial_lr_scale)

        elif ply_type == "pickle":  # for synthetic data
            gaussians.load_ply(ply_path)
          
    return loaded_iter
