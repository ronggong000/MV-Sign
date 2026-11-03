import os
import os.path as osp
from pathlib import Path
import numpy as np
import torch
import mmcv
from config.config import cfg
from util.preprocessing import load_img, augmentation_keep_size
from util.formatting import DefaultFormatBundle
from detrsmpl.data.datasets.pipelines.transforms import Normalize
import cv2


class INFERENCE_demo(torch.utils.data.Dataset):
    def __init__(self, video_path=None, out_path=None):
        """
        初始化函数，现在直接从视频文件加载所有帧到内存中。
        """
        # 1. 设置路径信息
        self.video_path = video_path
        input_path = Path(video_path)
        self.video_name = input_path.stem if input_path.is_file() else input_path.name
        #self.output_path = osp.join(out_path, self.video_name)
        self.output_path = out_path
        os.makedirs(self.output_path, exist_ok=True)

        # 2. 一次性将所有帧读入内存，避免重复磁盘读取
        print(f"Loading video frames from {video_path} into memory...")
        try:
            if input_path.is_dir():
                image_paths = [
                    p for p in sorted(input_path.iterdir())
                    if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
                ]
                self.frames = [mmcv.imread(str(p)) for p in image_paths]
            else:
                video_reader = mmcv.VideoReader(video_path)
                self.frames = [frame for frame in video_reader]
            print(f"Successfully loaded {len(self.frames)} frames.")
        except Exception as e:
            print(f"Error reading input {video_path}: {e}")
            self.frames = []

        # 3. 初始化一个列表，用于收集每一帧的推理结果
        self.all_outputs = []

        # 4. 保留必要的配置和数据转换流程
        self.num_person = cfg.num_person if 'num_person' in cfg else 0.1
        self.score_threshold = cfg.threshold if 'threshold' in cfg else 0.1
        self.format = DefaultFormatBundle()
        self.normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    def __len__(self):
        return len(self.frames)
    def pad_to_16_9(self, img):
        """
        将输入的图像（numpy array）用黑边填充到16:9的比例。
        """
        # 原始图像的高和宽
        h, w, _ = img.shape

        # 目标比例
        target_ratio = 16.0 / 9.0
        
        # 当前图像比例
        current_ratio = w / h

        if abs(current_ratio - target_ratio) < 1e-4: # 如果比例已经很接近16:9，直接返回
            return img

        if current_ratio > target_ratio:
            # 图像比16:9更“宽”，需要上下加黑边
            # 宽度保持不变，计算新的高度
            new_h = int(w / target_ratio)
            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            
            # 创建一个黑色的画布
            # 注意：这里使用 np.zeros，默认就是黑色
            canvas = np.zeros((new_h, w, 3), dtype=np.uint8)
            # 将原图粘贴到画布中央
            canvas[pad_top:pad_top + h, :, :] = img
            
        else:
            # 图像比16:9更“窄”（或更高），需要左右加黑边
            # 高度保持不变，计算新的宽度
            new_w = int(h * target_ratio)
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left
            
            # 创建一个黑色的画布
            canvas = np.zeros((h, new_w, 3), dtype=np.uint8)
            # 将原图粘贴到画布中央
            canvas[:, pad_left:pad_left + w, :] = img
            
        return canvas
    def __getitem__(self, idx):
        """
        从内存中直接获取帧进行处理。
        """
        img = self.frames[idx]  # 直接从内存列表读取，非常快
        # ==================== 新增代码 ====================
        # 在任何其他处理之前，先将图像填充到16:9
        img = self.pad_to_16_9(img)
        # ===============================================
        self.resolution = img.shape[:2]
        img_whole_bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        img, img2bb_trans, bb2img_trans, _, _ = \
            augmentation_keep_size(img, img_whole_bbox, 'test')

        img = (img.astype(np.float32))

        inputs = {'img': img}
        targets = {
            'body_bbox_center': np.array(img_whole_bbox[None]),
            'body_bbox_size': np.array(img_whole_bbox[None])
        }
        meta_info = {
            'ori_shape': np.array(self.resolution),
            'img_shape': np.array(img.shape[:2]),
            'img2bb_trans': img2bb_trans,
            'bb2img_trans': bb2img_trans,
            'ann_idx': idx
        }
        result = {**inputs, **targets, **meta_info}

        result = self.normalize(result)
        result = self.format(result)

        return result

    def inference(self, outs):
        """
        此函数不再写入任何文件，只负责收集模型输出。
        """
        for out in outs:
            #print(out.keys())
            # 将需要的数据从GPU转移到CPU，并转为numpy格式，方便后续保存
            # print(out['smplx_body_pose'].shape)
            # exit()
            processed_out = {
                'image_idx': out['image_idx'], # 保留帧索引以供排序检查
                'scores': out['scores'].cpu().numpy()[0],
                'body_bbox': out['body_bbox'].cpu().numpy()[0],
                'root_pose': out['smplx_root_pose'].cpu().numpy()[0],
                'body_pose': out['smplx_body_pose'].cpu().numpy()[0],
                'lhand_pose': out['smplx_lhand_pose'].cpu().numpy()[0],
                'rhand_pose': out['smplx_rhand_pose'].cpu().numpy()[0],
                'jaw_pose': out['smplx_jaw_pose'].cpu().numpy()[0],
                'shape': out['smplx_shape'].cpu().numpy()[0],
                'expr': out['smplx_expr'].cpu().numpy()[0],
                'cam_trans': out['cam_trans'].cpu().numpy()[0],
                #'smpl_verts': out['smpl_verts'][person_idx].cpu().numpy(),
                'smplx_joint_proj':out['smplx_joint_proj'].cpu().numpy()[0],
                'smpl_kp3d':out['smpl_kp3d'].cpu().numpy()[0],
            }
            self.all_outputs.append(processed_out)
