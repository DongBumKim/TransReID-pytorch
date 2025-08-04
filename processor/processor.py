import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist


import random 
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont



def get_fixed_random_queries(num_queries, num_samples=5, seed=42):
    rng = random.Random(seed)
    indices = list(range(num_queries))
    rng.shuffle(indices)
    return indices[:num_samples]

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


# def do_inference(cfg,
#                  model,
#                  val_loader,
#                  num_query):
#     device = "cuda"
#     logger = logging.getLogger("transreid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

#     evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()
#     img_path_list = []

#     for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
#         with torch.no_grad():
#             img = img.to(device)
#             camids = camids.to(device)
#             target_view = target_view.to(device)
#             feat = model(img, cam_label=camids, view_label=target_view)
#             evaluator.update((feat, pid, camid))
#             img_path_list.extend(imgpath)

#     cmc, mAP, _, _, _, _, _ = evaluator.compute()
#     logger.info("Validation Results ")
#     logger.info("mAP: {:.1%}".format(mAP))
#     for r in [1, 5, 10]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#     return cmc[0], cmc[4]


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query,
                 num_samples=5 # deafult as 5 for number query to visualize
                 ):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, distmat, all_pids, all_camids, qf, gf = evaluator.compute()

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # Prepare paths and labels
    query_paths = img_path_list[:num_query]
    gallery_paths = img_path_list[num_query:]
    q_pids = np.asarray(all_pids[:num_query])
    q_camids = np.asarray(all_camids[:num_query])
    g_pids = np.asarray(all_pids[num_query:])
    g_camids = np.asarray(all_camids[num_query:])

    output_dir = os.path.join(cfg.OUTPUT_DIR, 'rank_visualization')
    random_dir = os.path.join(output_dir, 'random')
    worst_dir = os.path.join(output_dir, 'worst')
    os.makedirs(random_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    # ---------- Get 5 random query indices ----------
    # #random_query_indices = random.sample(range(num_query), 5)
    random_query_indices = get_fixed_random_queries(num_query, num_samples, seed=cfg.SOLVER.SEED)
    # ---------- Get 5 worst-performing query indices ----------
    ranks = []
    for q_idx in range(num_query):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = np.argsort(distmat[q_idx])  # Sorted gallery indices
        remove = (g_pids == q_pid) & (g_camids == q_camid)
        keep = np.invert(remove)
        final_order = order[keep]
        match = (g_pids[final_order] == q_pid)
        rank_idx = np.where(match)[0]
        if len(rank_idx) == 0:
            ranks.append(9999)  # No correct match
        else:
            ranks.append(rank_idx[0])  # Position of correct match

    worst_query_indices = np.argsort(ranks)[-num_samples:].tolist()  # Highest rank = worst

    # ---------- Combine and visualize ----------
    # selected_queries = list(set(random_query_indices + worst_query_indices)) # if it overlaps, it deletes one
    selected_queries = random_query_indices + worst_query_indices # just allow overlap

    for idx in selected_queries:
        q_img_path = query_paths[idx]
        ranked_indices = distmat[idx].argsort()[:10]
        ranked_gallery_paths = [gallery_paths[g_idx] for g_idx in ranked_indices]
        ranked_gallery_pids = [g_pids[g_idx] for g_idx in ranked_indices]

        save_ranked_strip(
            query_idx=idx,
            query_path=q_img_path,
            ranked_gallery_paths=ranked_gallery_paths,
            ranked_gallery_pids=ranked_gallery_pids,
            query_pid=q_pids[idx],
            query_camid=q_camids[idx],
            gallery_pids=g_pids,
            gallery_camids=g_camids,
            output_dir=output_dir,
            mode="random" if idx in random_query_indices else "worst"
        )

    logger.info(f"Saved visualizations for {num_samples} random and {num_samples} worst queries.")

    return cmc[0], cmc[4]


def save_ranked_strip(query_idx, query_path, ranked_gallery_paths, ranked_gallery_pids,
                      query_pid, query_camid, gallery_pids, gallery_camids,
                      output_dir, mode="random"):
    
    # Settings
    image_width = 128
    image_height = 256
    spacing = 10  # Space between images
    label_height = 20

    sub_dir = os.path.join(output_dir, mode, f'query_{query_idx}_{os.path.splitext(os.path.basename(query_path))[0]}')
    os.makedirs(sub_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Load query image
    query_img = Image.open(query_path).convert('RGB').resize((image_width, image_height))

    # Load gallery images and prepare labels
    gallery_imgs = []
    gallery_labels = []
    for i, (g_path, g_pid) in enumerate(zip(ranked_gallery_paths, ranked_gallery_pids)):
        gallery_img = Image.open(g_path).convert('RGB').resize((image_width, image_height))
        gallery_imgs.append(gallery_img)
        is_match = (g_pid == query_pid)
        label_color = "green" if is_match else "red"
        gallery_labels.append((f"Rank {i+1}", label_color))

    # Canvas size: 1 query row + 2 gallery rows
    total_width = (5 * image_width) + (4 * spacing)
    total_height = (image_height * 3) + (label_height * 3) + (spacing * 3)
    canvas = Image.new('RGB', (total_width, total_height), color='black')
    draw = ImageDraw.Draw(canvas)

    # --- Draw query image centered ---
    q_x = (total_width - image_width) // 2
    canvas.paste(query_img, (q_x, 0))
    query_label = "Query"
    bbox = draw.textbbox((0, 0), query_label, font=font)
    q_label_w = bbox[2] - bbox[0]
    draw.text((q_x + (image_width - q_label_w) // 2, image_height + 2), query_label, fill="white", font=font)

    # --- Draw gallery images in 2 rows of 5 ---
    for row in range(2):
        for col in range(5):
            idx = row * 5 + col
            if idx >= len(gallery_imgs):
                continue

            x = col * (image_width + spacing)
            y = (image_height + label_height + spacing) * (row + 1)
            canvas.paste(gallery_imgs[idx], (x, y))

            # Draw label under each gallery image
            label_text, label_color = gallery_labels[idx]
            bbox = draw.textbbox((0, 0), label_text, font=font)
            label_w = bbox[2] - bbox[0]
            label_x = x + (image_width - label_w) // 2
            draw.text((label_x, y + image_height + 2), label_text, fill=label_color, font=font)

    # Save result
    canvas.save(os.path.join(sub_dir, "result.jpg"))
