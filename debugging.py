# import sys
# # sys.path.append('.')

# import torch


# from vggt.vggt.models.our_vggt import VGGT
# from vggt.vggt.utils.load_fn import load_and_preprocess_images
# from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
# from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
# from hugs.datasets import NeumanDataset
# import math

# from instantsplat.scene import GaussianModel
# from instantsplat.scene.cameras import Camera

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
# dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# # Initialize the model and load the pretrained weights.
# # This will automatically download the model weights the first time it's run, which may take a while.
# model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# # Freeze all model parameters
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.gs_head_xyz.parameters():
#     param.requires_grad = True
# for param in model.gs_head_feats.parameters():
#     param.requires_grad = True
# # for param in model.aggregator.parameters():
# #     param.requires_grad = True



# model.gs_head_xyz.train()
# model.gs_head_feats.train()


# from hugs.trainer.gs_trainer import get_train_dataset
# from omegaconf import OmegaConf
# from hugs.cfg.config import cfg as default_cfg
# from hugs.renderer.gs_renderer import render_human_scene
# import torchvision
# import torch.nn.functional as F


# cfg_file = "/home/emircan/projects/ml-hugs/output/human_scene/neuman/lab/hugs_trimlp/demo-dataset.seq=lab/2025-03-30_19-55-08/config_train.yaml"

# cfg_file = OmegaConf.load(cfg_file)

# cfg = OmegaConf.merge(default_cfg, cfg_file)


# import rerun as rr
# rr.init("debugging", recording_id="v0.1")
# rr.connect_tcp("0.0.0.0:9876")
# rr.log(f"world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]], colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

# from instantsplat.arguments import PipelineParams, ArgumentParser


# dataset = get_train_dataset(cfg)
# def get_data(idx):
#     data = dataset[idx]
#     return data

# def get_3d_loc_loss(gs_map, point_map):
#     t_loss = 0
#     for i in range(1):
#         pred_xyz = gs_map[0, i, :, :, 0:3].reshape(-1, 3)
#         # loss between 3d xyz coordinates and point_map
#         gt_xyz = point_map[0, i, :, :, :3].reshape(-1, 3)

#         # MSE loss between predicted and GT 3D coordinates
#         t_loss += F.mse_loss(pred_xyz, gt_xyz)
    
#     return t_loss / 4



# # def render_and_get_loss(gaussians, data):
# #     parser = ArgumentParser(description="Training script parameters")
# #     pipe = PipelineParams(parser)
# #     args = parser.parse_args(sys.argv[1:])
# #     pipe = pipe.extract(args)
# #     pose = gaussians.get_RT(viewpoint_cam.uid)


# #     bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

# #     render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)




# # def get_loss(gs_map, gs_map_xyz, data, step):
# #     # render
# #     t_loss = 0
# #     for i in range(1):
# #         sh0 = torch.sigmoid(gs_map[0, i, :, :, 3:6]).reshape(-1, 3)  # [N, 3]
# #         shs = sh0[:, None, :].expand(-1, 16, -1).clone()
# #         shs[:, 1:, :] = 0  # Only the first channel gets the SH coefficients

# #         sh0 = torch.sigmoid(gs_map[0, i, :, :, 3:6]).reshape(-1, 3)  # [N, 3]
# #         shs = torch.zeros(sh0.shape[0], 16, 3, device=sh0.device, dtype=sh0.dtype)  # [N, 16, 3]
# #         N = sh0.shape[0]
# #         n_red = N // 3
# #         n_green = N // 3
# #         n_blue = N - n_red - n_green  # Make sure all are assigned
# #         # Assign red
# #         shs[:n_red, 0, 0] = sh0[:n_red, 0]  # R
# #         # Assign green
# #         shs[n_red:n_red + n_green, 0, 1] = sh0[n_red:n_red + n_green, 1]  # G
# #         # Assign blue
# #         shs[n_red + n_green:, 0, 2] = sh0[n_red + n_green:, 2]  # B


# #         scene_gs_out = {
# #             "xyz":      gs_map_xyz[0, i, :, :, 0:3].reshape(-1, 3),
# #             "shs":      shs,
# #             "opacity":  torch.sigmoid(gs_map[0, i, :, :, 6].reshape(-1, 1)),
# #             # "scales":   torch.nn.functional.relu(gs_map[0, i, :, :, 7:10].reshape(-1, 3)),
# #             "scales":   torch.ones_like(gs_map[0, i, :, :, 7:10].reshape(-1, 3)) - 0.95,
# #             "rotq":     gs_map[0, i, :, :, 10:].reshape(-1, 4),
# #             "active_sh_degree": 0 
# #         }

# #         if step % 1 == 0:
# #             rr.set_time_seconds("frame", step)
# #             rr.log(f"world/human_{i}", rr.Points3D(positions=scene_gs_out["xyz"].detach().cpu().numpy(), colors=scene_gs_out["shs"][:, 0, :].detach().cpu().numpy()))

# #         render_pkg = render_human_scene(data[i], human_gs_out=None, scene_gs_out=scene_gs_out, bg_color=torch.tensor([0., 0., 0.], dtype=torch.float32, device="cuda"), render_mode="scene")

# #         gt_rgb = data[i]["rgb"]
# #         pred_rgb = render_pkg["render"]

# #         if step % 1 == 0:
# #             # rr.set_time_seconds("frame", step)
# #             # rr.log(f"world/human_{i}", rr.Points3D(positions=scene_gs_out["xyz"].detach().cpu().numpy(), colors=scene_gs_out["shs"][:, 0, :].detach().cpu().numpy()))
# #             if i == 0:
# #                 torchvision.utils.save_image(pred_rgb, f"./_irem/pred_{i}_{step}.png")
# #                 torchvision.utils.save_image(gt_rgb, f"./_irem/gt_{i}_{step}.png")

# #         # torchvision.utils.save_image(pred_rgb, f"./_irem/output_{step}.png")
# #         # torchvision.utils.save_image(gt_rgb, f"./_irem/gt_{step}.png")

# #         loss = torch.abs((pred_rgb - gt_rgb)).mean()

# #         t_loss += loss

# #         del render_pkg

# #     return t_loss, pred_rgb, gt_rgb

# image_names = [f"/home/emircan/projects/ml-hugs/data/neuman/dataset/lab/images/{i:05}.png" for i in range(1)]  
# images = load_and_preprocess_images(image_names).to(device)
# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         images = images[None]  # add batch dimension
#         aggregated_tokens_list, ps_idx = model.aggregator(images)

# data = [get_data(i) for i in range(1)]

# rr.set_time_seconds("frame", 0)
# rr.log("world/neuman_camera", rr.Pinhole(
#     image_from_camera=data[0]["cam_intrinsics"].cpu().numpy(),
#     resolution=[data[0]["image_width"], data[0]["image_height"]]  # width x height
# ))

# # Log the camera pose (as translation + rotation)
# rr.log("world/neuman_camera", rr.Transform3D(
#     translation=data[0]["camera_center"].cpu().numpy(),
# ))

# rr.log("world/neuman_camera/image", rr.Image(data[0]["rgb"].permute(1, 2, 0).cpu().numpy()))


# optimizer_xyz = torch.optim.AdamW(model.gs_head_xyz.parameters(), lr=2e-4)

# point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
# rr.set_time_seconds("frame", 0)
# rr.log(f"world/irem_inittt_{0}", rr.Points3D(positions=point_map[0, 0, :, :, 0:3].reshape(-1, 3).detach().cpu().numpy()))

# # pre-training
# from tqdm import tqdm
# for step in tqdm(range(600)):
#     # Forward pass through gs_head
#     gs_map, gs_map_conf = model.gs_head_xyz(aggregated_tokens_list, images, ps_idx)

#     if step % 50 == 0:
#         rr.set_time_seconds("frame", step)
#         rr.log(f"world/pred_pc{0}", rr.Points3D(positions=gs_map[0, 0, :, :, 0:3].reshape(-1, 3).detach().cpu().numpy()))

#     # Compute loss (you should define ground_truth appropriately)
#     loss = get_3d_loc_loss(gs_map, point_map)

#     print(loss)
#     loss.backward()
#     optimizer_xyz.step()
#     optimizer_xyz.zero_grad()

# torch.save(model.gs_head_xyz.state_dict(), "gs_head.pth")

# import numpy as np
# pose_enc = model.camera_head(aggregated_tokens_list)[-1]
# extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
# camera_to_world = np.eye(4, dtype=np.float32)
# camera_to_world[:3, :] = extrinsic[0, 0, :, :].cpu().numpy()  # Fill the top 3 rows

# rr.set_time_seconds("frame", 0)
# rr.log("world/camera", rr.Pinhole(
#     image_from_camera=intrinsic[0][0].cpu().numpy(),
#     resolution=[data[0]["image_width"], data[0]["image_height"]]  # width x height
# ))
# rr.log("world/camera", rr.Transform3D(
#     translation=extrinsic[0, 0, :, 3:].cpu().numpy(),
#     rotation=extrinsic[0, 0, :, :3].cpu().numpy()
# ))
# rr.log("world/camera/image", rr.Image(data[0]["rgb"].permute(1, 2, 0).cpu().numpy()))

# # for i in range(1):
# #     extr_4x4 = torch.eye(4, device='cuda:0')
# #     extr_4x4[:3, :] = extrinsic[0][i]  # Fill rotation and translation
# #     c2w = torch.inverse(extr_4x4)

# #     data[i]["world_view_transform"] = extr_4x4
# #     data[i]["c2w"] = c2w
# #     data[i]["cam_intrinsics"] = intrinsic[0][i]
# #     data[i]["camera_center"] = c2w[:3, 3]
# #     data[i]["transl"] = torch.tensor([0, 0, 0]).cuda()

# camera_list = []
# for i in range(1):
#     znear = data[i]["near"]
#     zfar = data[i]["far"]

#     # Step 1: Construct world_to_camera matrix (4x4)
#     extr_4x4 = torch.eye(4, device='cuda:0')
#     extr_4x4[:3, :] = extrinsic[0][i]  # [3x4] -> [4x4]
#     world_to_camera = extr_4x4
#     c2w = torch.inverse(world_to_camera)

#     # Step 2: Extract intrinsics
#     K = intrinsic[0][i]
#     width = data[i]["image_width"]
#     height = data[i]["image_height"]

#     # Step 3: Compute FoV from intrinsics
#     fovx = 2 * torch.atan(width / (2 * K[0, 0]))
#     fovy = 2 * torch.atan(height / (2 * K[1, 1]))

#     def get_projection_matrix(znear, zfar, fovX, fovY):
#         tanHalfFovY = math.tan((fovY / 2))
#         tanHalfFovX = math.tan((fovX / 2))

#         top = tanHalfFovY * znear
#         bottom = -top
#         right = tanHalfFovX * znear
#         left = -right

#         P = torch.zeros(4, 4)

#         z_sign = 1.0

#         P[0, 0] = 2.0 * znear / (right - left)
#         P[1, 1] = 2.0 * znear / (top - bottom)
#         P[0, 2] = (right + left) / (right - left)
#         P[1, 2] = (top + bottom) / (top - bottom)
#         P[3, 2] = z_sign
#         P[2, 2] = z_sign * zfar / (zfar - znear)
#         P[2, 3] = -(zfar * znear) / (zfar - znear)
#         return torch.tensor(P).cuda()

#     projection_matrix = get_projection_matrix(znear, zfar, fovx, fovy).transpose(0, 1)

#     # Step 5: Compute full projection transform
#     full_proj_transform = (world_to_camera.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

#     # Step 6: Camera center (in world space)
#     camera_center = c2w[:3, 3]

#     # Update dictionary
#     data[i].update({
#         "fovx": fovx,
#         "fovy": fovy,
#         "image_height": height,
#         "image_width": width,
#         "world_view_transform": world_to_camera,
#         "c2w": c2w,
#         "full_proj_transform": full_proj_transform,
#         "camera_center": camera_center,
#         "cam_intrinsics": K.float(),
#         "near": znear,
#         "far": zfar,
#     })


#     # Extract R and T from extrinsics
#     R = extrinsic[0][i][:, :3].cpu().numpy()
#     T = extrinsic[0][i][:, 3].cpu().numpy()

#     # Create Camera instance
#     cam = Camera(
#         colmap_id=i,
#         R=R,
#         T=T,
#         FoVx=fovx,
#         FoVy=fovy,
#         image=data[i]["image"],  # assuming shape [3, H, W] and values in [0, 1]
#         gt_alpha_mask=data[i].get("alpha_mask", None),
#         image_name=data[i].get("image_name", f"img_{i}.png"),
#         uid=i,
#         trans=np.array([0.0, 0.0, 0.0]),
#         scale=1.0,
#         data_device="cuda:0"
#     )

#     camera_list.append(cam)




# def set_gaussian_model_features(model: GaussianModel, features: dict):
#     """
#     Sets feature tensors on a GaussianModel instance without wrapping in nn.Parameter,
#     preserving gradient history.

#     Args:
#         model (GaussianModel): The target GaussianModel.
#         features (dict): Dictionary with keys and shapes:
#             - 'xyz': (P, 3)
#             - 'features_dc': (P, 3, 1)
#             - 'features_rest': (P, 3, (sh_degree+1)^2 - 1)
#             - 'opacity': (P, 1)
#             - 'scaling': (P, 3)
#             - 'rotation': (P, 4)
#     """
#     model._xyz = features["xyz"]
#     model._features_dc = features["features_dc"].transpose(1, 2).contiguous()
#     model._features_rest = features["features_rest"].transpose(1, 2).contiguous()
#     model._opacity = features["opacity"]
#     model._scaling = features["scaling"]
#     model._rotation = features["rotation"]

#     model.max_radii2D = torch.zeros((features["xyz"].shape[0]), device=features["xyz"].device)

#     print(f"Gaussian model features set. Num points: {features['xyz'].shape[0]}")






# cameras = {
#     1.0: camera_list
# }

# gaussians = GaussianModel(0)
# gaussians.init_RT_seq(cameras)




# model.gs_head_xyz.load_state_dict(torch.load("gs_head.pth"))
# optimizer_feats = torch.optim.AdamW(model.gs_head_feats.parameters(), lr=2e-4)
# for step in range(100):
#     # Forward pass through gs_head
#     gs_map_feats, _ = model.gs_head_feats(aggregated_tokens_list, images, ps_idx)
#     gs_map_xyz, _ = model.gs_head_xyz(aggregated_tokens_list, images, ps_idx)

#     features = {
#         "features_dc": gs_map_feats[:, 0:3].view(-1, 3, 1),
#         "features_rest": torch.empty((gs_map_feats.shape[0], 3, 0), device=gs_map_feats.device),
#         "opacity": gs_map_feats[:, 3:4],
#         "scaling": gs_map_feats[:, 4:7],
#         "rotation": gs_map_feats[:, 7:11],
#     }

#     set_gaussian_model_features(gaussians, features)

#     if step % 50 == 0:
#         rr.set_time_seconds("frame", step)
#         rr.log(f"world/pred_pc{0}", rr.Points3D(positions=gs_map_xyz[0, 0, :, :, 0:3].reshape(-1, 3).detach().cpu().numpy()))

#     # Compute loss (you should define ground_truth appropriately)
#     # loss, last_render, last_gt = get_loss(gs_map_feats, gs_map_xyz, data, step)

#     print(loss)

#     loss.backward()

#     optimizer_feats.step()

#     optimizer_feats.zero_grad()


#     if step % 10 == 0:
#         torchvision.utils.save_image(last_render, f"./_irem/output_{step}.png")
#         torchvision.utils.save_image(last_gt, f"./_irem/gt_{step}.png")
#         print("")




# # print("irem")
