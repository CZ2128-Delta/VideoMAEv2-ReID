import math
from typing import Iterable, Optional

import torch
from timm.utils import ModelEma

import utils
from reid.reid_metrics import compute_reid_metrics


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch[0]
        targets = batch[1]

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step

        if (lr_schedule_values is not None or wd_schedule_values is not None
                ) and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(samples)
            cls_score = outputs['cls_score']
            global_feat = outputs['global_feat']
            feat = outputs['feat']
            loss, loss_dict = criterion(cls_score, global_feat, feat, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        loss = loss / update_freq
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % update_freq == 0)

        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        acc1 = (cls_score.detach().max(1)[1] == targets).float().mean().item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(id_loss=loss_dict['id_loss'].item())
        if 'triplet_loss' in loss_dict:
            metric_logger.update(triplet_loss=loss_dict['triplet_loss'].item())
        metric_logger.update(class_acc=acc1)
        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = float('inf')
        max_lr = 0.0
        weight_decay_value = None
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        if weight_decay_value is not None:
            metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(id_loss=loss_dict['id_loss'].item(), head="loss")
            if 'triplet_loss' in loss_dict:
                log_writer.update(
                    triplet_loss=loss_dict['triplet_loss'].item(), head="loss")
            log_writer.update(class_acc=acc1, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            if weight_decay_value is not None:
                log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def _extract_features(data_loader, model, device, header):
    if data_loader is None:
        return None, None

    metric_logger = utils.MetricLogger(delimiter="  ")
    features_list = []
    labels_list = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0].to(device, non_blocking=True)
        labels = batch[1]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            feats = model(samples)
        features_list.append(feats.cpu())
        labels_list.append(labels.cpu())

    if not features_list:
        return None, None

    features = torch.cat(features_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    return features, labels


@torch.no_grad()
def validation_one_epoch(query_loader, gallery_loader, model, device):
    model.eval()
    q_features, q_labels = _extract_features(query_loader, model, device,
                                             'Val Query:')
    g_features, g_labels = _extract_features(gallery_loader, model, device,
                                             'Val Gallery:')
    stats = compute_reid_metrics(q_features, q_labels, g_features, g_labels)
    print(
        '* Rank-1 {:.2f}% Rank-5 {:.2f}% mAP {:.2f}%'.format(
            stats['rank1'] * 100, stats['rank5'] * 100, stats['mAP'] * 100))
    return stats


@torch.no_grad()
def final_test(query_loader, gallery_loader, model, device):
    model.eval()
    q_features, q_labels = _extract_features(query_loader, model, device,
                                             'Test Query:')
    g_features, g_labels = _extract_features(gallery_loader, model, device,
                                             'Test Gallery:')
    stats = compute_reid_metrics(q_features, q_labels, g_features, g_labels)
    print(
        '* Test Rank-1 {:.2f}% Rank-5 {:.2f}% mAP {:.2f}%'.format(
            stats['rank1'] * 100, stats['rank5'] * 100, stats['mAP'] * 100))
    return stats
