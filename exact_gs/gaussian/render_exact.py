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
import sys
import torch
import math
from exact_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

sys.path.append("./")
from exact_gs.gaussian.gaussian_model import GaussianModel
from exact_gs.dataset.cameras import Camera
from exact_gs.arguments import PipelineParams


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with exact rasterization.
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass
    

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:   #parallel
        tanfovx = 1.0
        tanfovy = 1.0
    elif mode == 1:   #cone
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

   
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,   
        projmatrix=viewpoint_camera.full_proj_transform,     
        campos=viewpoint_camera.camera_center,   
        prefiltered=False,
        mode=viewpoint_camera.mode,
        debug=pipe.debug,
        dDetector=float(viewpoint_camera.scanner_cfg['dDetector'][0]),  
        DSO=float(viewpoint_camera.scanner_cfg['DSO']),  
        DSD=float(viewpoint_camera.scanner_cfg['DSD']),  
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

  
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
      
        scales = pc.get_scaling
        rotations = pc.get_rotation

    
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
