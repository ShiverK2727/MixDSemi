import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import imshow, imsave
from scipy.ndimage.interpolation import map_coordinates
import cv2
from scipy import ndimage
from torchvision import transforms
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from torch import nn
from scipy.ndimage.interpolation import zoom


def to_multilabel(pre_mask, classes = 2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask


class add_salt_pepper_noise():
    def __call__(self, sample):
        image = sample['image']
        X_imgs_copy = np.asarray(image).copy()

        salt_vs_pepper = 0.2
        amount = 0.004

        num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))

        seed = random.random()
        if seed > 0.75:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 1
        elif seed > 0.5:
            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 0
        sample['image'] = X_imgs_copy
        return sample

class adjust_light():
    def __call__(self, sample):
        image = sample['image']
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
            sample['image'] = image
        return sample


class AdaptiveCLAHE():
    """Adaptive histogram equalization (CLAHE) with a probability.

    Operates on a PIL Image. For color images we convert to LAB and apply CLAHE
    on the L channel (preserves color). Parameters:
      p: probability to apply (default 0.5)
      clipLimit: CLAHE clipLimit (default 2.0)
      tileGridSize: tile grid size (default (8,8))
    """
    def __init__(self, p=0.5, clipLimit=2.0, tileGridSize=(8, 8)):
        self.p = float(p)
        self.clipLimit = float(clipLimit)
        self.tileGridSize = tuple(tileGridSize)

    def __call__(self, img_or_sample):
        """Accept either a PIL Image or a sample dict {'image': PIL, ...}.
        If given a dict, modifies and returns the dict; otherwise returns PIL Image.
        """
        is_dict = isinstance(img_or_sample, dict)
        img = img_or_sample['image'] if is_dict else img_or_sample

        # probability gate
        if random.random() >= self.p:
            return img_or_sample if is_dict else img

        try:
            arr = np.array(img)
            if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                # grayscale
                gray = arr if arr.ndim == 2 else arr[:, :, 0]
                clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
                out = clahe.apply(gray.astype(np.uint8))
                out_img = Image.fromarray(out.astype(np.uint8))
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # color: convert to LAB, apply to L channel
                try:
                    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
                    l2 = clahe.apply(l.astype(np.uint8))
                    lab2 = cv2.merge([l2, a, b])
                    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
                    out_img = Image.fromarray(out.astype(np.uint8))
                except Exception:
                    # fallback: apply per-channel CLAHE
                    out = np.zeros_like(arr)
                    clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
                    for c in range(3):
                        out[:, :, c] = clahe.apply(arr[:, :, c].astype(np.uint8))
                    out_img = Image.fromarray(out.astype(np.uint8))
            else:
                # unexpected shape -> return original
                out_img = img
        except Exception:
            # any error -> return original
            out_img = img

        if is_dict:
            img_or_sample['image'] = out_img
            return img_or_sample
        else:
            return out_img


class AdaptiveCLAHERandomized():
    """Adaptive CLAHE with randomized clipLimit and tileGridSize drawn from provided ranges.

    Usage:
      p: probability to apply
      clipLimit_range: (min, max) floats, sampled uniformly
      tileGridSize_range: either ((min_h, min_w), (max_h, max_w)) or (min_val, max_val)
                         integers; each dimension is sampled uniformly in the integer range.

    Behavior mirrors `AdaptiveCLAHE` but on each call samples new clipLimit and tileGridSize
    within the provided ranges.
    """
    def __init__(self, p=0.5, clipLimit_range=(1.0, 3.0), tileGridSize_range=((4, 4), (16, 16))):
        self.p = float(p)
        # normalize clip limit range
        if isinstance(clipLimit_range, (list, tuple)) and len(clipLimit_range) == 2:
            self.clip_min = float(clipLimit_range[0])
            self.clip_max = float(clipLimit_range[1])
        else:
            raise ValueError('clipLimit_range must be (min, max)')

        # normalize tileGridSize_range to ((min_h,min_w),(max_h,max_w))
        if isinstance(tileGridSize_range, (list, tuple)) and len(tileGridSize_range) == 2:
            a, b = tileGridSize_range
            # if provided as (min_val, max_val) -> make square ranges
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                self.tg_min = (int(a), int(a))
                self.tg_max = (int(b), int(b))
            else:
                # assume a and b are iterables of two ints
                self.tg_min = (int(a[0]), int(a[1]))
                self.tg_max = (int(b[0]), int(b[1]))
        else:
            raise ValueError('tileGridSize_range must be ((min_h,min_w),(max_h,max_w)) or (min_val,max_val)')

    def _sample_params(self):
        clip = random.uniform(self.clip_min, self.clip_max)
        # sample tile grid size integers between min and max (inclusive)
        th = random.randint(self.tg_min[0], max(self.tg_min[0], self.tg_max[0]))
        tw = random.randint(self.tg_min[1], max(self.tg_min[1], self.tg_max[1]))
        # ensure at least 1
        th = max(1, th)
        tw = max(1, tw)
        return clip, (th, tw)

    def __call__(self, img_or_sample):
        is_dict = isinstance(img_or_sample, dict)
        img = img_or_sample['image'] if is_dict else img_or_sample

        if random.random() >= self.p:
            return img_or_sample if is_dict else img

        # sample params per-call
        clip, tile = self._sample_params()

        try:
            arr = np.array(img)
            if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                gray = arr if arr.ndim == 2 else arr[:, :, 0]
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                out = clahe.apply(gray.astype(np.uint8))
                out_img = Image.fromarray(out.astype(np.uint8))
            elif arr.ndim == 3 and arr.shape[2] == 3:
                try:
                    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                    l2 = clahe.apply(l.astype(np.uint8))
                    lab2 = cv2.merge([l2, a, b])
                    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
                    out_img = Image.fromarray(out.astype(np.uint8))
                except Exception:
                    out = np.zeros_like(arr)
                    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
                    for c in range(3):
                        out[:, :, c] = clahe.apply(arr[:, :, c].astype(np.uint8))
                    out_img = Image.fromarray(out.astype(np.uint8))
            else:
                out_img = img
        except Exception:
            out_img = img

        if is_dict:
            img_or_sample['image'] = out_img
            return img_or_sample
        else:
            return out_img


class Brightness():# new defined
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, img):
        v = self.min_v + float(self.max_v-self.min_v)*random.random()
        return PIL.ImageEnhance.Brightness(img).enhance(v)

class Contrast():# new defined
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, img):
        v = self.min_v + float(self.max_v-self.min_v)*random.random()
        return PIL.ImageEnhance.Contrast(img).enhance(v)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, num_channels):
        self.num_channels = num_channels
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(num_channels, num_channels, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=num_channels)
        self.blur_v = nn.Conv2d(num_channels, num_channels, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=num_channels)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(self.num_channels, 1)

        self.blur_h.weight.data.copy_(x.view(self.num_channels, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(self.num_channels, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class reverse_aug():
    """blur a single image on CPU"""
    def __init__(self, kernel_size, num_channels, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v
        self.num_channels = num_channels
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(num_channels, num_channels, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=num_channels)
        self.blur_v = nn.Conv2d(num_channels, num_channels, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=num_channels)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img1, img2):
        v = self.min_v + float(self.max_v-self.min_v)*random.random()
        img1 =  PIL.ImageEnhance.Brightness(img1).enhance(v)
        img2 =  PIL.ImageEnhance.Brightness(img2).enhance(2-v)
        v = self.min_v + float(self.max_v-self.min_v)*random.random()
        img1 = PIL.ImageEnhance.Contrast(img1).enhance(v)
        img2 = PIL.ImageEnhance.Contrast(img2).enhance(2-v)
        
        img1 = self.pil_to_tensor(img1).unsqueeze(0)
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(self.num_channels, 1)
        self.blur_h.weight.data.copy_(x.view(self.num_channels, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(self.num_channels, 1, 1, self.k))
        with torch.no_grad():
            img1 = self.blur(img1)
            img1 = img1.squeeze()
        img1 = self.tensor_to_pil(img1)

        return img1, img2
    

class eraser():
    def __call__(self, sample, s_l=0.02, s_h=0.06, r_1=0.3, r_2=0.6, v_l=0, v_h=255, pixel_level=False):
        image = sample['image']
        img_h, img_w, img_c = image.shape


        if random.random() > 0.5:
            return sample

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        image[top:top + h, left:left + w, :] = c
        sample['image'] = image
        return sample

class elastic_transform():
    """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """

    # def __init__(self):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        alpha = image.size[1] * 2
        sigma = image.size[1] * 0.08
        random_state = None
        seed = random.random()
        if seed > 0.5:
            # print(image.size)
            assert len(image.size) == 2

            image_channel = len(np.array(image).shape)
            label_channel = len(np.array(label).shape)

            if random_state is None:
                random_state = np.random.RandomState(None)

            shape = image.size[0:2]
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            # transformed_label = np.zeros([image.size[0], image.size[1]])
            if image_channel == 3:
                transformed_image = np.zeros([image.size[0], image.size[1], 3])
                for i in range(3):
                    # print(i)
                    transformed_image[:, :, i] = map_coordinates(np.array(image)[:, :, i], indices, order=1).reshape(shape)
                    # break
            elif image_channel == 2:
                transformed_image = np.zeros([image.size[0], image.size[1]])
                transformed_image[:, :] = map_coordinates(np.array(image)[:, :], indices, order=1).reshape(shape)
            if label is not None:
                if label_channel == 3:
                    transformed_label = np.zeros([label.size[0], label.size[1], 3])
                    for i in range(3):
                        transformed_label[:, :, i] = map_coordinates(np.array(label)[:, :, i], indices, order=0, mode='nearest', prefilter=False).reshape(shape)
                elif label_channel == 2:
                    transformed_label = np.zeros([label.size[0], label.size[1]])
                    transformed_label[:, :] = map_coordinates(np.array(label)[:, :], indices, order=0, mode='nearest', prefilter=False).reshape(shape)
                # transformed_label[:, :] = map_coordinates(np.array(label)[:, :], indices, order=1, mode='nearest').reshape(shape)
            else:
                transformed_label = None
            transformed_image = transformed_image.astype(np.uint8)

            if label is not None:
                transformed_label = transformed_label.astype(np.uint8)
            sample['image'] = Image.fromarray(transformed_image)
            sample['label'] = transformed_label
        return sample


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        # print(img.size)
        w, h = img.size
        if self.padding > 0 or w < self.size[0] or h < self.size[1]:
            padding = np.maximum(self.padding,np.maximum((self.size[0]-w)//2+5,(self.size[1]-h)//2+5))
            img = ImageOps.expand(img, border=padding, fill=0)
            # Use 0 as mask padding value (background) to avoid inserting 255 which
            # previously represented background in some datasets and caused label confusion.
            mask = ImageOps.expand(mask, border=padding, fill=0)

        assert img.width == mask.width
        assert img.height == mask.height
        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            sample['image'] = img
            sample['label'] = mask
            return sample
            # return {'image': img,
            #         'label': mask,
            #         'img_name': sample['img_name'],
            #         'dc': sample['dc']}
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        # print(img.size)
        sample['image'] = img
        sample['label'] = mask
        return sample


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.width == mask.width
        # assert img.height == mask.height
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        # y1 = int(round((h - th) / 2.))
        y1 = int(round((h - th) / 2.))
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        sample['image'] = img
        sample['label'] = mask
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = mask
        return sample


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']

        assert img.width == mask.width
        assert img.height == mask.height
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'img_name': name}



class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.width == mask.width
        assert img.height == mask.height
        w, h = img.size

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,
                    'label': mask,
                    'img_name': sample['img_name']}
        oh, ow = self.size
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        assert img.width == mask.width
        assert img.height == mask.height
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                img = img.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': img,
                        'label': mask,
                        'img_name': name}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, size=512, fillcolor=0, img_fillcolor=None):
        self.degree = random.randint(1, 4) * 90
        self.size = size
        self.fillcolor = fillcolor  # fillcolor for mask
        # For image rotation, default to 0 (black background) if not specified
        self.img_fillcolor = img_fillcolor if img_fillcolor is not None else 0

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = random.random()
        if seed > 0.5:
            rotate_degree = self.degree
            # Fixed: use fillcolor instead of expand for filling rotated areas
            img = img.rotate(rotate_degree, Image.BILINEAR, fillcolor=self.img_fillcolor)
            mask = mask.rotate(rotate_degree, Image.NEAREST, fillcolor=self.fillcolor)
            sample['image'] = img
            sample['label'] = mask
        return sample

class RandomScaleRotate(object):
    def __init__(self, size=512, left=-20, right=20, fillcolor=0, img_fillcolor=None):
        self.size = size
        self.left = left
        self.right = right
        self.fillcolor = fillcolor  # fillcolor for mask
        # For image rotation, default to 0 (black background) if not specified
        # Medical images typically have black backgrounds
        self.img_fillcolor = img_fillcolor if img_fillcolor is not None else 0

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = random.random()
        if seed > 0.5:
            rotate_degree = random.randint(self.left, self.right)
            # Rotate image with explicit fillcolor (black for medical images)
            img = img.rotate(rotate_degree, Image.BILINEAR, fillcolor=self.img_fillcolor)
            # Rotate mask with dataset-specific fillcolor (background value)
            mask = mask.rotate(rotate_degree, Image.NEAREST, fillcolor=self.fillcolor)

            sample['image'] = img
            sample['label'] = mask
        return sample


class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print(img.size)
        assert img.width == mask.width
        assert img.height == mask.height

        seed = random.random()
        if seed > 0.5:
            w = int(random.uniform(1, 1.5) * img.size[0])
            h = int(random.uniform(1, 1.5) * img.size[1])

            img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
            sample['image'] = img
            sample['label'] = mask
        return self.crop(sample)


class ResizeImg(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        assert img.width == mask.width
        assert img.height == mask.height

        img = img.resize((self.size, self.size))
        # mask = mask.resize((self.size, self.size))

        sample = {'image': img, 'label': mask, 'img_name': name}
        return sample


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        assert img.width == mask.width
        assert img.height == mask.height

        img = img.resize((self.size, self.size))
        mask = mask.resize((self.size, self.size))

        sample = {'image': img, 'label': mask, 'img_name': name}
        return sample



class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class GetBoundary(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):
        cup = mask[:, :, 0]
        disc = mask[:, :, 1]
        dila_cup = ndimage.binary_dilation(cup, iterations=self.width).astype(cup.dtype)
        eros_cup = ndimage.binary_erosion(cup, iterations=self.width).astype(cup.dtype)
        dila_disc= ndimage.binary_dilation(disc, iterations=self.width).astype(disc.dtype)
        eros_disc= ndimage.binary_erosion(disc, iterations=self.width).astype(disc.dtype)
        cup = dila_cup + eros_cup
        disc = dila_disc + eros_disc
        cup[cup==2]=0
        disc[disc==2]=0
        size = mask.shape
        # boundary = np.zers(size[0:2])
        boundary = (cup + disc) > 0
        return boundary.astype(np.uint8)


class Normalize_tf(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), dataRange = [-1, 1]):
        self.mean = mean
        self.std = std
        self.get_boundary = GetBoundary()
        self.dataRange = dataRange

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        # __mask = np.array(sample['label']).astype(np.uint8)
        if self.dataRange == [-1, 1]:
            img /= 127.5
            img -= 1.0
        elif self.dataRange == [0, 1]:
            img /= 255
        else:
            assert(0)
        if 'strong_aug' in sample.keys():
            strong = np.array(sample['strong_aug']).astype(np.float32)
            if self.dataRange == [-1, 1]:
                strong /= 127.5
                strong -= 1.0
            elif self.dataRange == [0, 1]:
                strong /= 255
            else:
                assert(0)
            sample['strong_aug'] = strong
        # _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
        # _mask[__mask > 200] = 255
        # # index = np.where(__mask > 50 and __mask < 201)
        # _mask[(__mask > 50) & (__mask < 201)] = 128
        # _mask[(__mask > 50) & (__mask < 201)] = 128

        # __mask[_mask == 0] = 2
        # __mask[_mask == 255] = 0
        # __mask[_mask == 128] = 1

        # mask = to_multilabel(__mask)
        sample['image'] = img
        # sample['label'] = mask
        return sample


class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}

def ToMultiLabel(dc):
    new_dc = np.zeros([3])
    for i in range(new_dc.shape[0]):
        if i == dc:
            new_dc[i] = 1
            return new_dc

def SoftLable(label):
    new_label = label.copy()
    label = list(label)
    index = label.index(1)
    new_label[index] = 0.8+random.random()*0.2
    accelarate = new_label[index]
    for i in range(len(label)):
        if i != index:
            if i == len(label) - 1:
                new_label[i] = 1 - accelarate
            else:
                new_label[i] = random.random()*(1-accelarate)
                accelarate += new_label[i]
    return new_label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, low_res=0, unet_size=0):
        self.low_res = low_res
        self.unet_size = unet_size

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if len(np.array(sample['image']).shape) == 2:
            sample['image'] = np.expand_dims(np.array(sample['image']).astype(np.float32), 2)  # add channel dimension
        # if len(np.array(sample['label']).shape) == 2:
        #     sample['label'] = np.expand_dims(np.array(sample['label']).astype(np.float32), 2)  # add channel dimension
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        map = np.array(sample['label']).astype(np.uint8)#.transpose((2, 0, 1))
        img_x, img_y = img.shape[1:]
        x, y = map.shape
        if self.low_res > 0:
            low_res_label = zoom(map, (self.low_res / x, self.low_res / y), order=0)
            low_res_label = torch.from_numpy(low_res_label).float()
            sample['low_res_label']=low_res_label
        if self.unet_size > 0:
            unet_size_img = zoom(img, (1, self.unet_size / img_x, self.unet_size / img_y), order=0)
            unet_size_img = torch.from_numpy(unet_size_img).float()
            sample['unet_size_img']=unet_size_img
            unet_size_label = zoom(map, (self.unet_size / x, self.unet_size / y), order=0)
            unet_size_label = torch.from_numpy(unet_size_label).float()
            sample['unet_size_label']=unet_size_label
        if 'strong_aug' in sample.keys():
            if len(np.array(sample['strong_aug']).shape) == 2:
                sample['strong_aug'] = np.expand_dims(np.array(sample['strong_aug']).astype(np.float32), 2)  # add channel dimension
            strong = np.array(sample['strong_aug']).astype(np.float32).transpose((2, 0, 1))
            if self.unet_size > 0:
                unet_size_strong_aug = zoom(strong, (1, self.unet_size / img_x, self.unet_size / img_y), order=0)
                unet_size_strong_aug = torch.from_numpy(unet_size_strong_aug).float()
                sample['unet_size_strong_aug']=unet_size_strong_aug
            strong = torch.from_numpy(strong).float()
            sample['strong_aug'] = strong
        img = torch.from_numpy(img).float()
        map = torch.from_numpy(map).float()
        sample['image']=img
        sample['label']=map
        # domain_code = torch.from_numpy(SoftLable(ToMultiLabel(sample['dc']))).float()
        # sample['dc'] = domain_code
        return sample
    

class RandomPatchSamplerWithClass(object):
    """
    扩展版的随机 patch 采样器，支持一次采样多个 patch 并返回对应的二分类标签（是否包含前景）。

        行为：
        - 该变换应当在 Compose 之前直接被 dataset 调用（不要放入 Compose）。
        - 每次调用返回多个 patch，并把结果写回 `sample`：
                - 'patches'：list of PIL.Image，已被缩放到原图大小（w,h）。
                - 'patch_masks'：list of PIL.Image（二值或标签掩码），对应每个 patch 的裁剪 mask 并缩放到原图大小（w,h），插值使用 NEAREST。
                - 'patch_labels'：list of int（0/1），表示该 patch 是否包含前景。
            不再修改 `sample['image']` 或 `sample['label']`（不兼容旧 API）。

    参数说明：
    :param num_patches: 每次返回的 patch 数量（int），例如 4
    :param num_fg: 每次必须返回的前景 patch 个数（int），例如 2
    :param min_ratio: patch 的最小长/宽比例（float in (0,1]），例如 0.5 表示裁剪框的宽和高至少为原图相应维度的 50%
    :param fg_threshold: 判定前景的阈值（比例），patch 中前景像素比例 > fg_threshold 即视为包含前景
    :param num_attempts: 单个目标标签寻找的最大尝试次数
    :param fg_func: 自定义前景判断函数 func(mask_np) -> binary_mask,默认为 (mask_np > 0)
                    用于支持特殊情况如 Prostate (0=前景,需要 mask_np==0)
    """
    def __init__(self,
                 num_patches=4,
                 num_fg=2,
                 min_ratio=0.5,
                 fg_threshold=0.01,
                 num_attempts=50,
                 return_coords=False,
                 fg_func=None):
        self.num_patches = int(num_patches)
        self.num_fg = int(num_fg)
        if self.num_fg > self.num_patches:
            raise ValueError('num_fg must be <= num_patches')
        self.min_ratio = float(min_ratio)
        assert 0.0 < self.min_ratio <= 1.0
        self.fg_threshold = float(fg_threshold)
        self.num_attempts = int(num_attempts)
        self.return_coords = bool(return_coords)
        self.fg_func = fg_func if fg_func is not None else (lambda m: m > 0)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        # 转为 numpy mask（灰度）
        if not isinstance(mask, np.ndarray):
            mask_np = np.array(mask.convert('L'))
        else:
            mask_np = mask

        w, h = img.size

        # 最小裁剪尺寸（整数像素）
        min_w = max(1, int(round(self.min_ratio * w)))
        min_h = max(1, int(round(self.min_ratio * h)))

        patches = []
        labels = []
        masks = []
        patch_coords = []

        # 预计算二值前景 mask 与积分图，便于快速计算任意窗口内前景像素数量
        # 使用自定义前景判断函数(支持 Prostate 的 0=前景情况)
        mask_bin = self.fg_func(mask_np).astype(np.uint8)
        # integral image: integral[y, x] = sum over mask_bin[:y+1, :x+1]
        integral = mask_bin.cumsum(axis=0).cumsum(axis=1)
        total_fg = float(integral[-1, -1])

        # 固定第一个采样为整张图（原始大小），并把它计入前景统计
        full_fg_count = int(total_fg)
        is_full_foreground = (full_fg_count > (w * h * self.fg_threshold))
        # append full image as the first patch
        try:
            full_mask_pil = Image.fromarray(mask_np.astype(np.uint8))
        except Exception:
            full_mask_pil = Image.fromarray(np.asarray(mask_np, dtype=np.uint8))
        patches.append(img if isinstance(img, Image.Image) else Image.fromarray(np.array(img)))
        masks.append(full_mask_pil)
        labels.append(1 if is_full_foreground else 0)
        patch_coords.append((0, 0, int(w), int(h)))

        # 构造剩余的目标标签队列（去掉已经固定的第一个），并打乱顺序
        remaining_patches = max(0, self.num_patches - 1)
        # 如果第一个整图已被判断为前景，则需要的前景数量减一
        remaining_needed_fg = max(0, self.num_fg - (1 if is_full_foreground else 0))
        targets = [1] * remaining_needed_fg + [0] * (remaining_patches - remaining_needed_fg)
        random.shuffle(targets)

        def area_sum(x, y, ww, hh):
            # x,y are left,top; ww,hh are width/height
            x2 = x + ww - 1
            y2 = y + hh - 1
            if x2 < 0 or y2 < 0:
                return 0
            s = integral[y2, x2]
            if x > 0:
                s -= integral[y2, x - 1]
            if y > 0:
                s -= integral[y - 1, x2]
            if x > 0 and y > 0:
                s += integral[y - 1, x - 1]
            return s

        for target_label in targets:
            # 每个目标尝试多次采样；对于 background (0) 我们会在候选中选最小前景比例的
            best_candidate = None
            best_fg_prop = 1.0
            last_crop = None
            for _ in range(self.num_attempts):
                # 随机采样裁剪尺寸（在 [min_w, w] 和 [min_h, h] 之间）
                cw = random.randint(min_w, w)
                ch = random.randint(min_h, h)
                # 随机左上角
                if w - cw <= 0:
                    x1 = 0
                else:
                    x1 = random.randint(0, w - cw)
                if h - ch <= 0:
                    y1 = 0
                else:
                    y1 = random.randint(0, h - ch)

                fg_count = area_sum(x1, y1, cw, ch)
                is_foreground = (fg_count > (cw * ch * self.fg_threshold))
                last_crop = (x1, y1, cw, ch, is_foreground)

                # 如果目标为前景，优先接受第一个满足阈值的裁剪
                if target_label == 1:
                    if is_foreground:
                        best_candidate = (x1, y1, cw, ch, is_foreground)
                        break
                else:
                    # 目标为背景：记录前景比例最小的候选
                    fg_prop = fg_count / float(max(1, cw * ch))
                    if fg_prop < best_fg_prop:
                        best_fg_prop = fg_prop
                        best_candidate = (x1, y1, cw, ch, is_foreground)
                        # 在完全无前景的候选时可以提前退出
                        if fg_prop <= 0.0:
                            break

            # 选择最终裁剪：前景优先选中满足阈值的裁剪；背景选最低前景比例候选；若均无则退回最后一次尝试
            if best_candidate is None and last_crop is None:
                # 兜底：整图
                x1, y1, cw, ch = 0, 0, w, h
                is_foreground = (np.sum(mask_np > 0) > (w * h * self.fg_threshold))
            elif best_candidate is None:
                x1, y1, cw, ch, is_foreground = last_crop
            else:
                x1, y1, cw, ch, is_foreground = best_candidate

            # 裁剪并缩放到原图大小以兼容后续变换
            cropped = img.crop((x1, y1, x1 + cw, y1 + ch))
            if (cw, ch) != (w, h):
                cropped = cropped.resize((w, h), Image.BILINEAR)

            patches.append(cropped)
            # 返回的标签以实际是否含前景为准（优先保证 target 条件，但在失败时可能不一致）
            labels.append(1 if is_foreground else 0)

            # 同时裁剪并保存 mask（PIL），最近邻重采样以保留标签值
            mask_patch_np = mask_np[y1:y1+ch, x1:x1+cw]
            try:
                mask_pil = Image.fromarray(mask_patch_np.astype(np.uint8))
            except Exception:
                mask_pil = Image.fromarray(np.asarray(mask_patch_np, dtype=np.uint8))
            if (cw, ch) != (w, h):
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
            masks.append(mask_pil)

            # 记录裁剪坐标（相对于原始 resize 后的图像尺寸）
            patch_coords.append((int(x1), int(y1), int(cw), int(ch)))

        # 将裁剪结果写回 sample
        sample['patches'] = patches
        sample['patch_masks'] = masks if masks else [Image.fromarray(np.zeros((h, w), dtype=np.uint8)) for _ in range(len(patches))]
        sample['patch_labels'] = labels
        if getattr(self, 'return_coords', False):
            sample['patch_coords'] = patch_coords

        return sample
    



