from boundary_iou.coco_instance_api.coco import COCO
from boundary_iou.coco_instance_api.cocoeval import COCOeval
from boundary_iou.lvis_instance_api.lvis import LVIS
from boundary_iou.lvis_instance_api.eval import LVISEval
from boundary_iou.lvis_instance_api.results import LVISResults
import pycocotools.mask as maskUtils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import cv2
import os

import torch


def coco_subsample(coco, num_samples, seed=None):
    if seed is not None:
        random.seed(seed)
    keys = random.sample(coco.imgs.keys(), num_samples)
    coco.imgs = {k: coco.imgs[k] for k in keys}
    coco.dataset['images'] = list(coco.imgs.values())
    coco.anns = {k: v for k, v in coco.anns.items()
                 if v['image_id'] in coco.imgs}
    coco.dataset['annotations'] = list(coco.anns.values())


@torch.no_grad()
def evaluate(vae, coco, target_size=(64, 64)):
    results_json = []
    for image_id in tqdm(coco.getImgIds(), mininterval=10):
        for anns_id in coco.getAnnIds(imgIds=image_id):
            anns = coco.loadAnns(anns_id)[0]
            mask_gt = coco.annToMask(anns)
            x, y, w, h = anns['bbox']
            x1, y1, x2, y2 = round(x), round(y), round(x + w), round(y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            mask = cv2.resize(
                mask_gt[y1:y2, x1:x2], target_size[::-1], interpolation=cv2.INTER_LINEAR)
            mask = torch.tensor(mask)[None, None, :, :].cuda()
            if hasattr(vae, 'get_codebook_indices'):
                code = vae.get_codebook_indices(mask)
                remask = vae.decode(code)[0, 0, :, :].cpu().numpy() * 0.5 + 0.5
            else:
                _, _, remask = vae(mask)
                remask = remask[0, 0, :, :].cpu().numpy()
            remask = cv2.resize(remask, (x2 - x1, y2 - y1),
                                interpolation=cv2.INTER_LINEAR)
            remask = (remask >= 0.5).astype(mask_gt.dtype)
            mask_dt = np.zeros_like(mask_gt)
            mask_dt[y1:y2, x1:x2] = remask

            results_json.append(dict(
                image_id=image_id,
                category_id=anns['category_id'],
                segmentation=maskUtils.encode(mask_dt),
                score=1.0,
            ))
    cocoDt = coco.loadRes(results_json)
    cocoEval = COCOeval(cocoGt=coco, cocoDt=cocoDt, iouType='segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    segm_ap = cocoEval.stats[:6]

    cocoEval = COCOeval(cocoGt=coco, cocoDt=cocoDt, iouType='boundary')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    boundary_ap = cocoEval.stats[:6]

    return segm_ap, boundary_ap


@torch.no_grad()
def evaluate_lvis(vae, coco, target_size=(64, 64)):
    results_json = []
    coco.getImgIds = coco.get_img_ids
    coco.getAnnIds = coco.get_ann_ids
    coco.loadAnns = coco.load_anns
    coco.annToMask = coco.ann_to_mask
    for image_id in tqdm(coco.getImgIds(), mininterval=10):
        for anns_id in coco.get_ann_ids(img_ids=[image_id]):
            anns = coco.loadAnns([anns_id])[0]
            mask_gt = coco.annToMask(anns)
            x, y, w, h = anns['bbox']
            img_h, img_w = mask_gt.shape
            x1, y1, x2, y2 = round(x), round(y), min(
                round(x + w), img_w), min(round(y + h), img_h)  # for lvis clamp outbound box
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            mask = cv2.resize(
                mask_gt[y1:y2, x1:x2], target_size[::-1], interpolation=cv2.INTER_LINEAR)
            mask = torch.tensor(mask)[None, None, :, :].cuda()
            if hasattr(vae, 'get_codebook_indices'):
                code = vae.get_codebook_indices(mask)
                remask = vae.decode(code)[0, 0, :, :].cpu().numpy() * 0.5 + 0.5
            else:
                _, _, remask = vae(mask)
                remask = remask[0, 0, :, :].cpu().numpy()
            remask = cv2.resize(remask, (x2 - x1, y2 - y1),
                                interpolation=cv2.INTER_LINEAR)
            remask = (remask >= 0.5).astype(mask_gt.dtype)
            mask_dt = np.zeros_like(mask_gt)
            mask_dt[y1:y2, x1:x2] = remask

            results_json.append(dict(
                image_id=image_id,
                category_id=anns['category_id'],
                segmentation=maskUtils.encode(mask_dt),
                score=1.0,
            ))
    cocoDt = LVISResults(coco, results_json)
    cocoEval = LVISEval(lvis_gt=coco, lvis_dt=cocoDt, iou_type='segm')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    cocoEval.print_results()
    segm_ap = (cocoEval.results['AP'], cocoEval.results['AP50'], cocoEval.results['AP75'],
               cocoEval.results['APs'], cocoEval.results['APm'], cocoEval.results['APl'])

    cocoEval = LVISEval(lvis_gt=coco, lvis_dt=cocoDt, iou_type='boundary')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    cocoEval.print_results()
    boundary_ap = (cocoEval.results['AP'], cocoEval.results['AP50'], cocoEval.results['AP75'],
                   cocoEval.results['APs'], cocoEval.results['APm'], cocoEval.results['APl'])
    return segm_ap, boundary_ap


def mask_evaluation(model, coco_dir, target_size, iou_type=None, max_samples=5000, seed=1234):
    model.eval()

    print('eval vae on instances_train2017.json')
    coco = COCO(os.path.join(coco_dir, 'annotations/instances_train2017.json'))
    coco_subsample(coco, num_samples=max_samples, seed=seed)
    segm_ap, boundary_ap = evaluate(model, coco, target_size=target_size)
    print(
        f"train segm_mAP_copypaste: {segm_ap[0]:.3f},{segm_ap[1]:.3f},{segm_ap[2]:.3f},{segm_ap[3]:.3f},{segm_ap[4]:.3f},{segm_ap[5]:.3f}")
    print(
        f"train boundary_mAP_copypaste: {boundary_ap[0]:.3f},{boundary_ap[1]:.3f},{boundary_ap[2]:.3f},{boundary_ap[3]:.3f},{boundary_ap[4]:.3f},{boundary_ap[5]:.3f}")

    print('eval vae on instances_val2017.json')
    coco = COCO(os.path.join(coco_dir, 'annotations/instances_val2017.json'))
    segm_ap, boundary_ap = evaluate(model, coco, target_size=target_size)
    print(
        f"val segm_mAP_copypaste: {segm_ap[0]:.3f},{segm_ap[1]:.3f},{segm_ap[2]:.3f},{segm_ap[3]:.3f},{segm_ap[4]:.3f},{segm_ap[5]:.3f}")
    print(
        f"val boundary_mAP_copypaste: {boundary_ap[0]:.3f},{boundary_ap[1]:.3f},{boundary_ap[2]:.3f},{boundary_ap[3]:.3f},{boundary_ap[4]:.3f},{boundary_ap[5]:.3f}")

    print('eval vae on lvis_v0.5_val_cocofied.json')
    coco = LVIS(os.path.join(
        coco_dir, 'annotations/lvis_v0.5_val_cocofied.json'))
    segm_ap, boundary_ap = evaluate_lvis(model, coco, target_size=target_size)
    print(
        f"cocofied val segm_mAP_copypaste: {segm_ap[0]:.3f},{segm_ap[1]:.3f},{segm_ap[2]:.3f},{segm_ap[3]:.3f},{segm_ap[4]:.3f},{segm_ap[5]:.3f}")
    print(
        f"cocofied val boundary_mAP_copypaste: {boundary_ap[0]:.3f},{boundary_ap[1]:.3f},{boundary_ap[2]:.3f},{boundary_ap[3]:.3f},{boundary_ap[4]:.3f},{boundary_ap[5]:.3f}")
