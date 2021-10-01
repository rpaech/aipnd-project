#!/usr/bin/env python3

"""Predicts the category of a flower shown in an image.

This script accepts the path to an image file showing a flower and attempts 
to predict the flower's category along with the level of confidence in the 
prediction.

Usage:
    predict.py <image> <checkpoint> --top_k <number> --category_names <file> --gpu

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
    $ python predict.py ./flwr_image.jpg ./flwr_model.pth --category_names ./cat_to_name.json --top_k 3 --gpu
    colts foot (87.5%)
    common dandelion (11.3%)
    barbeton daisy (0.8%)
"""


def main():
    pass

if __name__ == '__main__':
    main()