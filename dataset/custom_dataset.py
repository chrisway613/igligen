import os
import random
import easydict
import itertools

import torch
import numpy as np

from accelerate.logging import get_logger
from torch.utils.data import IterableDataset, get_worker_info


logger = get_logger(__name__, log_level='INFO')


# Reference: torchvision `_box_cxcywh_to_xyxy`
def cxcywh_to_xyxy(boxes, clip=False):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box.
    (w, h) are width and height of bounding box.
    
    Args:
        boxes (Array[N, 4]): boxes in (cx, cy, w, h) format which will be converted.
        clip (bool): whether to clip out-of-bound values.

    Returns:
        boxes (Array(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    delta_x, delta_y = 0.5 * w, 0.5 * h
    x1, y1 = cx - delta_x, cy - delta_y
    x2, y2 = cx + delta_x, cy + delta_y
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    if clip:
        boxes = np.clip(boxes, 0., 1.)
    
    return boxes


class InfiniteDataset(IterableDataset):
    def __init__(
        self, data_path, train_shards,
        prob_use_caption, prob_use_boxes,
        box_confidence_th, batch_size,
        transform, ddp_rank, num_ddp_processes, *,
        max_boxes_per_image=32, shard_shuffle_seed=None,
        no_caption_only=False, return_cxcywh=False
    ):
        self.train_shards = train_shards
        if shard_shuffle_seed is not None:
            self.train_shards = np.copy(train_shards)
            rng = np.random.defalut_rng(seed=shard_shuffle_seed)
            rng.shuffle(self.train_shards)

        self.data_path = data_path
        self.transform = transform
        self.batch_size = batch_size

        self.return_cxcywh = return_cxcywh
        self.no_caption_only = no_caption_only

        self.prob_use_boxes = prob_use_boxes
        self.prob_use_caption = prob_use_caption
        self.box_confidence_th = box_confidence_th
        self.max_boxes_per_image = max_boxes_per_image

        self.ddp_rank = ddp_rank
        self.num_ddp_processes = num_ddp_processes
        self.shard_shuffle_seed = shard_shuffle_seed

    def __iter__(self, worker_info=None):
        if worker_info is None:
            worker_info = get_worker_info()

        worker_id = worker_info.id + self.ddp_rank * worker_info.num_workers
        num_workers = worker_info.num_workers * self.num_ddp_processes

        seed = worker_info.seed % 2 ** 32
        random.seed(seed)
        np.random.seed(seed)

        # Reference: https://stackoverflow.com/a/69779930
        while True:
            # This is for simplicity: a worker always loads a certain set of numpy files
            worker_shards = itertools.islice(
                self.train_shards, worker_id, None, step=num_workers)
            for shard in worker_shards:
                print(f"Loading shard: {shard}")
                shard_iter = self.shard_iter(shard)

                batch = []
                for item in shard_iter:
                    batch.append(item)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch.clear()

    def shard_iter(self, shard):
        # Load latents and boxes
        latents_path = os.path.join(self.data_path, "latents", f"{shard}.npy")
        latents = np.load(latents_path, allow_pickle=True).items()
        
        boxes_path = os.path.join(self.data_path, "boxes", f"{shard}.npy")
        boxes = np.load(boxes_path, allow_pickle=True)
        # Shuffle the saved boxes
        np.random.shuffle(boxes)

        # Map images to latents
        image_to_latent_map = {
            image_idx: latent_idx
            for latent_idx, image_idx in enumerate(latents['indices'])
        }
        
        for image_idx, caption, boxes_raw, boxes_confidence, box_phrases in boxes:
            if image_idx not in latents['indices']:
                continue
            
            box_threshold = self.box_confidence_th
            if random.uniform(0., 1.) >= self.prob_use_boxes:
                # For `prob_use_boxes`, we use boxes. Otherwise, we ignore boxes.
                # Reference: https://github.com/gligen/GLIGEN/blob/f0ede1e5dc9e5f710fd564da297a3c1ba71a20b0/ldm/modules/diffusionmodules/openaimodel.py#L428
                box_threshold = 1.0
                if self.no_caption_only:
                    # also drop the caption if the boxes are dropped
                    # This is consistent with GLIGEN implementation: https://github.com/gligen/GLIGEN/blob/f9dccb9c6cf48bad03c3666290a7dec8c5e58f3c/demo/gligen/ldm/modules/diffusionmodules/openaimodel.py#L399
                    caption = ""
            else:
                if boxes_confidence.shape[0] > self.max_boxes_per_image:
                    box_threshold = np.max(self.box_confidence_th, np.sort(boxes_confidence)[-self.max_boxes_per_image])

            if not self.return_cxcywh:
                boxes_raw = cxcywh_to_xyxy(boxes_raw, clip=True)
            
            boxes_mask = boxes_confidence > box_threshold
            boxes_raw = boxes_raw[boxes_mask]
            box_phrases = [
                phrase for phrase, mask 
                in zip(box_phrases, boxes_mask) if mask
            ]
            
            if boxes_raw.shape[0] > self.max_boxes_per_image:
                boxes_raw = boxes_raw[:self.max_boxes_per_image]
                box_phrases = box_phrases[:self.max_boxes_per_image]
            
            num_boxes = boxes_raw.shape[0]
            
            boxes_padded = np.zeros((self.max_boxes_per_image, 4))
            masks_padded = np.zeros(self.max_boxes_per_image)
            
            boxes_padded[:num_boxes] = boxes_raw
            masks_padded[:num_boxes] = 1.
            
            if np.randomx.uniform(0., 1.) > self.prob_use_caption:
                # For `prob_use_caption`, we use caption. Otherwise, we ignore caption and only use boxes.
                caption = ""
            
            latents = latents['latents'][image_to_latent_map[image_idx]]
            
            # NOTE: image_index is numpy number
            outputs = dict(
                id=int(image_idx), caption=caption, 
                boxes=torch.tensor(boxes_padded), box_phrases=box_phrases, 
                text_masks=torch.tensor(masks_padded), latents=torch.tensor(latents)
            )
            if self.transform:
                outputs = self.transform(outputs)
            
            yield outputs
