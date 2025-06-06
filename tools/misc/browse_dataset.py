# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from collections import Sequence
from pathlib import Path

import mmcv
from mmcv import Config, DictAction
from mmdet.datasets.builder import build_dataset

from mmrotate.core.visualization import imshow_det_rbboxes


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    """Retrieve the dataset config file.

    Args:
        config_path (str): Path of the config file.
        skip_type (list[str]): List of the useless pipeline to skip.
        cfg_options (dict): dict of configs to merge from.
    """

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)
    loaded_images = set()

    for item in dataset:
        loaded_images.add(os.path.basename(item['filename']))

    print(f"实际加载的图片数: {len(loaded_images)}")

# 保存加载的图片列表
    with open("loaded_images.txt", "w") as f:
        f.writelines("\n".join(sorted(loaded_images)))

    # total_images = len(dataset)
    # total_annotations = sum(len(item['gt_bboxes']) for item in dataset)

    # print(f"正在可视化的数据集：{dataset.ann_file}")
    # print(f"📂 数据集图片总数: {total_images}")
    # print(f"📝 总标注框数: {total_annotations}")

    progress_bar = mmcv.ProgressBar(len(dataset))

    for item in dataset:

        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels']
        
        print(f"📷 图片: {item['filename']}, 目标数: {len(gt_bboxes)}")

        imshow_det_rbboxes(
            item['img'],
            gt_bboxes,
            gt_labels,
            class_names=dataset.CLASSES,
            score_thr=0,
            show=not args.not_show,
            wait_time=args.show_interval,
            out_file=filename,
            bbox_color=dataset.PALETTE,
            text_color=(200, 200, 200))

        progress_bar.update()


if __name__ == '__main__':
    main()
