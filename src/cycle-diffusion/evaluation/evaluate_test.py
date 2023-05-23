import os
import torch
from cleanfid import fid
from tqdm import tqdm
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image

from .utils import save_image, calculate_ssim, calculate_psnr
from utils.file_utils import list_image_files_recursively
from preprocess.afhqwild256 import INTERPOLATION


class Evaluator(object):

    def __init__(self, args, meta_args):
        self.args = args
        self.meta_args = meta_args

        self.ref_transform = transforms.Compose([
            transforms.Resize(256, interpolation=INTERPOLATION),  # 512 -> 256
            transforms.ToTensor()
        ])

    def evaluate(self, images, model, weighted_loss, losses, data, split):
        """

        Args:
            images: list of images, or list of tuples of images
            model: model to evaluate
            weighted_loss: list of scalar tensors
            losses: dictionary of lists of scalar tensors
            data: list of dictionary
            split: str

        Returns:

        """
        assert split in ['eval', 'test']

        # Add metrics here.
        f_gen = os.path.join(self.meta_args.output_dir, 'temp_gen')
        if os.path.exists(f_gen):
            os.remove(f_gen)
        os.mkdir(f_gen)
 
        n = len(images)
        all_psnr, all_ssim, l2 = 0, 0, 0
        idx = 0
        for original_img, img in tqdm(images):
            assert img.dim() == original_img.dim() == 3

            img = img.clamp(0, 1)
            original_img = original_img.clamp(0, 1)

            all_psnr += calculate_psnr(img, original_img)
            all_ssim += calculate_ssim(
                (img.numpy() * 255).transpose((1, 2, 0)),
                (original_img.numpy() * 255).transpose((1, 2, 0)),
            )
            l2 += torch.sqrt(
                ((img - original_img) ** 2).sum(2).sum(1).sum(0)
            ).item()

            assert img.shape == original_img.shape
            save_image(os.path.join(f_gen, '{}.png'.format(idx)), img)
            idx += 1

        summary = {
            "psnr": all_psnr / n,
            "ssim": all_ssim / n,
            "l2": l2 / n
        }

        return summary
