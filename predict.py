#!/usr/bin/env python3

"""Predicts the category of a flower shown in an image.

This script accepts the path to an image file showing a flower and attempts 
to predict the flower's category along with the level of confidence in the 
prediction.

Usage:
    predict.py <image> <checkpoint> --top_k <number> --category_names <file> 
               --gpu

Args:
    <image>:  The path to an image file.
    <checkpoint>:  The path to a model checkpoint file created using the 
        train.py script.
    --top_k <number>:  Return the top <number> most likely categories.  If 
        top_k is not defined, only one prediction is returned.
    --category_names <path>:  When predicting a category, show the name of the 
        flower, rather than its category number, using the category to name 
        mapping provided the <file> in JSON format.
    --gpu:  Use the GPU for prediction, if available.

Example:
    $ python predict.py ./flwr_image.jpg ./flwr_model.pth --category_names 
    ./cat_to_name.json --top_k 3 --gpu
    
    colts foot (87.5%)
    common dandelion (11.3%)
    barbeton daisy (0.8%)
"""


import argparse
import json
import torch
import flower_net as fn
from PIL import Image
from pathlib import Path


def get_input_args():
    
    argp = argparse.ArgumentParser(
        description="""Predicts the category of a flower shown in an image.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argp.add_argument('image_path', type=Path,
                      help="""The path to an image file.""")
    argp.add_argument('checkpoint', type=Path,
                      help="""The path to a model checkpoint file created using 
                              the train.py script.""")
    argp.add_argument('--top_k', type=int, default=1,
                      help="""Return the top <number> most likely 
                              categories.""")
    argp.add_argument('--category_names', type=Path,
                      help="""The path to a JSON file mapping a flower category 
                              to its name.""")
    argp.add_argument('--gpu', action='store_true',
                      help="""Use the GPU for training, if available.""")

    args = argp.parse_args()
    
    if not args.image_path.exists():
        raise FileNotFoundError(
            "Image file '{}' doesn't exist.".format(args.image_path))
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "Checkpoint file '{}' doesn't exist.".format(args.checkpoint))
    if args.category_names is not None and not args.category_names.exists():
        raise FileNotFoundError(
            "Category name file '{}' doesn't exist.".format(
                args.category_names))
    if args.top_k < 1 or fn.FLOWER_CAT_COUNT < args.top_k:
        raise Exception("Top K is '{}'. Permitted range is [1, {}].".format(
            args.top_k, fn.FLOWER_CAT_COUNT))

    return args


def load_image(path):
    
    with Image.open(path) as im:
        return fn.pred_transform(im)


def load_cat_to_name(path):
    
    with open(path, 'r') as file:
        return json.load(file)


def predict(fnm, image, topk, on_gpu):
    
    device = fn.get_device(on_gpu)

    fnm.model.to(device)
    fnm.model.eval()
    with torch.no_grad():
        pred_results = torch.exp(fnm.model(image.unsqueeze(dim=0).to(device)))

    top_probs, top_results = pred_results.topk(topk, dim=1)

    top_probs = top_probs.squeeze_(dim=0).tolist()
    top_cats = [fnm.idx_to_cat[x.item()] for x in top_results.squeeze_(dim=0)]

    return top_probs, top_cats


def main():
    
    args = get_input_args()
    
    image = load_image(args.image_path)
    fnm = fn.load(args.checkpoint)
    
    cat_to_name = None
    if args.category_names is not None:
        cat_to_name = load_cat_to_name(args.category_names)

    top_probs, top_cats = predict(fnm, image, args.top_k, args.gpu)

    for prob, cat in zip(top_probs, top_cats):
        if cat_to_name is not None:
            cat = cat_to_name[cat]
        print('{} ({:.5f})'.format(cat, prob))


if __name__ == '__main__':
    main()
