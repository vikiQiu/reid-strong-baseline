# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import os
import logging

import torch
import torch.nn as nn
from torchvision.utils import save_image
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking
from utils.utils import check_dir_exists
from data.datasets.eval_reid import eval_func_with_plot


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, fun=eval_func_with_plot)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    print('get_metrics')
    cmc, mAP, good_case = evaluator.state.metrics['r1_mAP']
    plot(val_loader, 'good_case', good_case, add=num_query)
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        
        
def plot(data_loader, d_name, inds=[[1,2,3],[1,4,5]], add=0):
    ds = data_loader.dataset
    d = os.path.join('output', d_name)
    check_dir_exists([d])

    for i in range(len(inds)):
        imgs = []
        for j in range(len(inds[i])):
            tmp = 0 if j == 0 else add
            imgs.append(ds[inds[i][j+tmp]][0])
        # imgs = torch.Tensor(imgs)
        save_image(imgs, os.path.join(d, 'img_%d.jpg' % (i)))
