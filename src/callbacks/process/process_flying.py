import torch
import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import src.callbacks.process.process_generic as generic


def process_data(inputs, targets, meta_info, img_res=224):
    gt_kp3d_b = targets["object.kp3d.full.b"].unsqueeze(0)
    gt_kp3d_r = targets['mano.j3d.full.r'].unsqueeze(0)
    gt_kp3d_l = targets['mano.j3d.full.l'].unsqueeze(0)

    gt_kp2d_b = targets["object.kp2d.norm.b"].unsqueeze(0)
    gt_kp2d_r = targets["mano.j2d.norm.r"].unsqueeze(0)  # 2D keypoints for object base
    gt_kp2d_l = targets["mano.j2d.norm.l"].unsqueeze(0)  # 2D keypoints for object base

    gt_kp2d_b = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)
    gt_kp2d_r = data_utils.unormalize_kp2d(gt_kp2d_r, img_res)
    gt_kp2d_l = data_utils.unormalize_kp2d(gt_kp2d_l, img_res)

    combo_kp3d = torch.cat([gt_kp3d_b, gt_kp3d_r, gt_kp3d_l], dim=1)
    combo_kp2d = torch.cat([gt_kp2d_b, gt_kp2d_r, gt_kp2d_l], dim=1)

    # estimate camera translation by solving 2d to 3d correspondence
    gt_transl = camera.estimate_translation_k(
        combo_kp3d,
        combo_kp2d,
        meta_info["intrinsics"].unsqueeze(0).cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    ).squeeze()

    # move to camera coord
    targets["mano.j3d.cam.r"] = targets['mano.j3d.full.r'] + gt_transl[None, :]
    targets["mano.j3d.cam.l"] = targets['mano.j3d.full.l'] + gt_transl[None, :]
    return inputs, targets, meta_info
