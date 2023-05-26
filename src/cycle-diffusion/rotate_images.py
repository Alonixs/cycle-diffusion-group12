import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from PIL import Image
import os, os.path
from pathlib import Path


def load_data(path):
    imgs = []
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    # for f in os.listdir(path):
    for f in path.iterdir():
        ext = f.suffix

        if ext not in valid_images:
            print(f"Skipping {f} because it is not a valid image")
            continue
        imgs.append(Image.open(os.path.join(path, f)))
        print(f"Loaded {f}")

    return imgs


def horizontal_flip(img):
    path, name = Path(img.filename).parents[0], Path(img.filename).stem
    new_name = f"{name}rotated0flipped.png"

    img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
    img2.save(path / new_name)


def rotate_and_save(img, angle):
    path, name = Path(img.filename).parents[0], Path(img.filename).stem
    new_name = f"{name}rotated{str(angle)}.png"

    padded = np.pad(img, pad_width=((128, 128), (128, 128), (0, 0)), mode="symmetric")
    padded = Image.fromarray(padded)

    img2 = padded.rotate(angle)

    width, height = img2.size
    new_width, new_height = 512, 512

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    img2 = img2.crop(
        (left, top, right, bottom)
    )  # standard resize because images are 512x512

    img2.save(path / new_name)


def save_image_perturbations(imgs, angles=[0, 1, 5, 10, 30, 45, 90]):
    """
    Best variant using padding, other variants see below.
    """

    for img in imgs:
        # horizontal flip, and saving
        horizontal_flip(img)

        # rotations
        for angle in angles:
            rotate_and_save(img, angle)


cwd = (
    os.getcwd()
)  # should be cycle-diffusion-group12/src/cycle-diffusion folder, otherwise change this
print(f"BASE_DIR: {cwd}, should be the cycle-diffusion-group12 folder")

BASE_DIR = Path(cwd)

cat_path = BASE_DIR / "exp1_2_data/cat"
dog_path = BASE_DIR / "exp1_2_data/dog"
wild_path = BASE_DIR / "exp1_2_data/wild"

cat_dataset = load_data(cat_path)
# dog_dataset = load_data(dog_path)
# wild_dataset = load_data(wild_path)

print(len(cat_dataset))

# all_three = [cat_dataset, dog_dataset, wild_dataset]
angles = [0, 1, 5, 10, 30, 45, 90]  # 0 is the duplicate of the original

# ONLY RUN THIS ONCE
# for dataset in all_three:
#     save_image_perturbations(dataset, angles=angles)

save_image_perturbations(cat_dataset, angles=angles)
