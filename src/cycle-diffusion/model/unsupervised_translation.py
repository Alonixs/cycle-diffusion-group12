import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import os
import pickle
from torchvision import utils


from .model_utils import requires_grad
from .gan_wrapper.get_gan_wrapper import get_gan_wrapper


class UnsupervisedTranslation(nn.Module):
    def __init__(self, args):
        super(UnsupervisedTranslation, self).__init__()

        # Set up source and target gan_wrapper
        self.source_gan_wrapper = get_gan_wrapper(args.gan)
        self.target_gan_wrapper = get_gan_wrapper(args.gan, target=True)
        # Freeze.
        requires_grad(self.source_gan_wrapper, True)  # Otherwise, no trainable params.
        requires_grad(self.target_gan_wrapper, True)  # Otherwise, no trainable params.

        assert self.source_gan_wrapper.resolution == self.target_gan_wrapper.resolution
        self.encode_transform = transforms.Compose(
            [
                transforms.Resize(self.source_gan_wrapper.resolution),
                transforms.ToTensor(),
            ]
        )

        self.args = args

    def forward(self, sample_id, class_label=None, original_image=None, file_name=None):
        # Eval mode for the source and target gan_wrapper.
        self.source_gan_wrapper.eval()
        self.target_gan_wrapper.eval()

        assert not self.training

        if getattr(self.source_gan_wrapper, "model_embedding_space", False):
            assert getattr(self.target_gan_wrapper, "model_embedding_space", False)
            assert not getattr(self.source_gan_wrapper, "enforce_class_input", False)
            assert not getattr(self.target_gan_wrapper, "enforce_class_input", False)
            raise NotImplementedError()
        elif getattr(self.source_gan_wrapper, "enforce_class_input", False):
            assert getattr(self.target_gan_wrapper, "enforce_class_input", False)
            assert not getattr(self.source_gan_wrapper, "model_embedding_space", False)
            assert not getattr(self.target_gan_wrapper, "model_embedding_space", False)
            assert class_label is not None
            z = self.source_gan_wrapper.encode(
                image=original_image, class_label=class_label
            )
            img = self.target_gan_wrapper(z=z, class_label=class_label)
        else:
            assert class_label is None

            z, extra_data = self.source_gan_wrapper.encode(
                image=original_image,
                custom_z_name=self.args.custom_z_name,
                seed=self.args.seed,
            )  # NEW, pass custom_z_name to get intermediate z's
            img = self.target_gan_wrapper(z=z)

        # Placeholders
        losses = dict()
        weighted_loss = torch.zeros_like(sample_id).float()

        # NEW
        if self.args.save_images and file_name is not None:
            img_name = os.path.basename(file_name[0])
            output_path = os.path.join(
                self.args.output_dir, img_name.replace("cat", "gen_dog")
            )

            img = img.clamp(0, 1)
            utils.save_image(img, output_path)

        # NEW Save z to file, if custom_z_name is set too
        if self.args.save_images and file_name is not None and self.args.custom_z_name:
            img_name = os.path.basename(file_name[0])
            z_name = str("z_" + img_name + ".npy")
            output_path = os.path.join(self.args.output_dir, z_name)

            np.save(
                output_path,
                z.cpu().numpy(),
            )
            print(f"====== Saved z to {output_path} =========")

        return (original_image, img), weighted_loss, losses

    @property
    def device(self):
        return next(self.parameters()).device


Model = UnsupervisedTranslation
