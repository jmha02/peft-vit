# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.datasets import CIFAR100

import utils

import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.stat_scores import StatScores
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import PeftModel

from utils import block_expansion

def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = ReturnIndexDataset(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")


    # ============ building network ... ============
    model_path = args.model_path 
   
    if args.training_mode == "block":

        config = AutoConfig.from_pretrained(model_path)
        config.image_size = 224

        model = AutoModelForImageClassification.from_config(config)
        ckpt = model.state_dict()        
        _, selected_layers = block_expansion(ckpt, args.split, config.num_hidden_layers)
        config.num_hidden_layers += len(selected_layers)
        config.num_labels = args.num_classes

        model = AutoModelForImageClassification.from_config(config)
        model.classifier = torch.nn.Linear(config.hidden_size, args.num_classes)

    else:

        model = AutoModelForImageClassification.from_pretrained(
                model_path,
                num_labels=args.num_classes,
                ignore_mismatched_sizes=True,
                image_size=224,
            )

    if args.training_mode == "lora":
        # Wrap in LoRA model
        
        config = LoraConfig(
            r=args.r_lora,
            lora_alpha=args.r_lora,
            target_modules= ["query", "value"],
            lora_dropout=0.0,
            bias='none',
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)
        
    print(model)

    if args.training_mode != "pretrained":
        print(f"Loaded weights from {args.weights}")

        ckpt = torch.load(args.weights)["state_dict"]

        # Remove prefix from key names
        new_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("net"):
                k = k.replace("net" + ".", "")
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print(f"Use pretrained weights! {model_path}")
    
    if args.training_mode == "lora":
        for i in range(12):
            if f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.base_layer.weight' in new_state_dict.keys():
                new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.weight'] = new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.base_layer.weight']
                new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.query.weight'] = new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.query.base_layer.weight']
                new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.bias'] = new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.base_layer.bias']
                new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.query.bias'] = new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.query.base_layer.bias']

                del new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.query.base_layer.bias']
                del new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.base_layer.bias']
                del new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.query.base_layer.weight']
                del new_state_dict[f'base_model.model.vit.encoder.layer.{i}.attention.attention.value.base_layer.weight']


        model.load_state_dict(new_state_dict, strict=True)

    model.classifier = nn.Identity()

    model.cuda()
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).logits.clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='weights/dino_vitbase16_pretrain.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_base', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/scratch/alpine/reak3132/imagenet/ILSVRC/Data/CLS-LOC', type=str)
    parser.add_argument('--data_train_path', default='/scratch/alpine/reak3132/in_train', type=str)
    parser.add_argument('--data_val_path', default='/scratch/alpine/reak3132/in_val', type=str)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--training_mode', default='', type=str)
    parser.add_argument('--model_path', default='facebook/dino-vitb16', type=str)
    
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--port', default=12341, type=int)
    parser.add_argument('--split', default=4, type=int)
    parser.add_argument('--r_lora', default=8, type=int)
    
    args = parser.parse_args()


    # python eval_knn_cifar.py --training_mode=block --weights=/scratch/alpine/reak3132/code/vit-finetune/output/cifar100-block/version_0/checkpoints/best-step-step=4000-val_acc=0.8880.ckpt
    # python eval_knn_cifar.py --training_mode=lora --weights=/scratch/alpine/reak3132/code/vit-finetune/output/cifar100-lora-r8/version_0/checkpoints/best-step-step=5000-val_acc=0.8835.ckpt
    # python eval_knn_cifar.py --training_mode=full --weights=/scratch/alpine/reak3132/code/vit-finetune/output/cifar100/version_0/checkpoints/best-step-step=5000-val_acc=0.8647.ckpt

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")
    dist.barrier()
