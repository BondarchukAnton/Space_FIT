from typing import Dict

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import RandomAffine, RandomHorizontalFlip, ColorJitter, ToTensor
import numpy as np
from random import choice, sample, randrange, shuffle
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IMode
import random
from skimage.util import random_noise
import albumentations as A


#len_data = list(range(len(train_data.imgs)))




# img = '/home/user/PycharmProjects/MRInet/Dataset/LGG/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_15.tif'
# mask = '/home/user/PycharmProjects/MRInet/Dataset/LGG/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_15_mask.tif'
#
# img = Image.open(img)
# mask = Image.open(mask)
#
# ra = RandomAffine(degrees=35)
rgba_imodes = [IMode.NEAREST, IMode.BILINEAR]
imodes = [IMode.NEAREST, IMode.BILINEAR, IMode.BICUBIC, IMode.BOX, IMode.HAMMING, IMode.LANCZOS]
# background_images = Path('/media/user/Новый том/SAR Segmentation Datasets/WHU-OPT-SAR dataset/sar')
# background_masks = Path('/media/user/Новый том/SAR Segmentation Datasets/WHU-OPT-SAR dataset/lbl')
# background_rgbmasks = Path('/media/user/Новый том/SAR Segmentation Datasets/WHU-OPT-SAR dataset/predict')
# background_opticals = Path('/media/user/Новый том/SAR Segmentation Datasets/WHU-OPT-SAR dataset/optical')
# img_names = [x.name for x in background_images.glob('*')]

def blend(img, mask):
    aimg = np.squeeze(np.array(img), axis=0)
    amask = np.squeeze(np.array(mask), axis=0)
    aimg[amask > 0] = 1

    return aimg


class RandomMirror():
    def __init__(self, p=0.25, resolution=512):
        self.p = p
        self.resolution = resolution

    # Высота и ширина должны быть чётными
    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:

            if self.resolution:
                resolution_x, resolution_y = self.resolution, self.resolution
            else:
                resolution_x, resolution_y = img.size

            if img.size != (resolution_x,resolution_y) or mask.size != (resolution_x,resolution_y):
                img = F.resize(img, [resolution_x, resolution_y])
                mask = F.resize(mask, [resolution_x, resolution_y], interpolation=IMode.NEAREST)
            b_img = np.array(img)
            b_mask = np.array(mask)
            new_image = np.zeros_like(b_img).astype(np.uint8)
            new_mask = np.zeros_like(b_mask).astype(np.uint8)
            size_x = np.random.randint(((resolution_x * 50) / 100), ((resolution_x * 93) / 100))
            size_y = np.random.randint(((resolution_y * 50) / 100), ((resolution_y * 93) / 100))
            cut_x = np.random.randint(0, resolution_x - size_x)
            cut_y = np.random.randint(0, resolution_y - size_y)
            b_img = b_img[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...]
            b_mask = b_mask[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...]
            cut_x = np.random.randint(0, resolution_x - size_x)
            cut_y = np.random.randint(0, resolution_y - size_y)
            new_image[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...] = b_img
            new_mask[cut_y:cut_y + size_y, cut_x:cut_x + size_x, ...] = b_mask

            if new_image[cut_y:cut_y + size_y, (cut_x + size_x):, ...].shape != b_img[:, ::-1][:, :resolution_x - (cut_x + size_x)].shape:
                print(new_image.shape, b_img.shape, cut_x, cut_y, size_x, size_y)

            if new_mask[cut_y:cut_y + size_y, (cut_x + size_x):, ...].shape != b_mask[:, ::-1][:, :resolution_x - (cut_x + size_x)].shape:
                print(new_mask.shape, b_mask.shape, cut_x, cut_y, size_x, size_y)

            new_image[cut_y:cut_y + size_y, (cut_x + size_x):, ...] = b_img[:, ::-1][:, :resolution_x - (cut_x + size_x)]
            new_mask[cut_y:cut_y + size_y, (cut_x + size_x):, ...] = b_mask[:, ::-1][:, :resolution_x - (cut_x + size_x)]

            new_image[cut_y:cut_y + size_y, :cut_x, ...] = b_img[:, ::-1][:, size_x - cut_x:]
            new_mask[cut_y:cut_y + size_y, :cut_x, ...] = b_mask[:, ::-1][:, size_x - cut_x:]

            new_image[:cut_y, :, ...] = new_image[cut_y:2 * cut_y, :][::-1, :]
            new_mask[:cut_y, :, ...] = new_mask[cut_y:2 * cut_y, :][::-1, :]

            new_image[cut_y + size_y:, :, ...] = new_image[2 * (cut_y + size_y) - resolution_y:cut_y + size_y, :][::-1, :]
            new_mask[cut_y + size_y:, :, ...] = new_mask[2 * (cut_y + size_y) - resolution_y:cut_y + size_y, :][::-1, :]

            new_image = Image.fromarray(new_image)
            new_mask = Image.fromarray(new_mask)
            return new_image, new_mask
        else:
            return img, mask

class RandomResizeCrop():
    def __init__(self, p1=0.5, p2=0.5, erode=True):
        self.p1 = p1
        self.p2 = p2
        self.erode = erode


    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            return img, mask
        else:
            w, h = img.size
            if np.random.rand() < self.p2:
                scale = np.random.randint(80, 100) / 100
                scale2 = np.random.randint(70, 100) / 100
                crop_size = [int(w * scale), int(h * scale)]
                indx = np.random.randint(0, 2)
                crop_size[indx] = int(crop_size[indx] * scale2)
                point = [np.random.randint(0, w - crop_size[0] + 1), np.random.randint(0, h - crop_size[1]) + 1]
                new_image = F.crop(img, point[1], point[0], crop_size[1], crop_size[0])
                new_image = F.resize(new_image, [w, h], interpolation=IMode.BILINEAR)
                new_mask = F.crop(mask, point[1], point[0], crop_size[1], crop_size[0])
                new_mask = F.resize(new_mask, [w, h], interpolation=IMode.NEAREST)
                return new_image, new_mask
            else:
                if self.erode:
                    scale = np.random.randint(80, 100) / 100
                    new_image = F.resize(img, [int(w * scale), int(h * scale)], interpolation=IMode.BILINEAR)
                    new_mask = F.resize(mask, [int(w * scale), int(h * scale)], interpolation=IMode.NEAREST)
                    l = np.random.randint(0, int(w - w * scale))
                    t = np.random.randint(0, int(h - h * scale))
                    r = int(w - w * scale) - l
                    d = int(h - h * scale) - t
                    new_image = F.pad(new_image, padding=(l, t, r, d))
                    new_mask = F.pad(new_mask, padding=(l, t, r, d))
                    return new_image, new_mask
                else:
                    return img, mask


class RandomShift():
    def __init__(self, p1=0.5, p2=0.5):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            w, h = img.size
            left = np.random.randint(0, int(w * 0.3))
            top = np.random.randint(0, int(h * 0.3))
            right = np.random.randint(0, int(w * 0.3))
            down = np.random.randint(0, int(h * 0.3))
            new_image = F.pad(img, padding=(left, top, right, down))
            new_mask = F.pad(mask, padding=(left, top, right, down))
            if np.random.rand() < self.p2:
                new_image = F.crop(new_image, 0, 0, h, w)
                new_mask = F.crop(new_mask, 0, 0, h, w)
            else:
                new_image = F.crop(new_image, top + down, left + right, h, w)
                new_mask = F.crop(new_mask, top + down, left + right, h, w)
            return new_image, new_mask

        else:
            return img, mask


class RandomGaussianBlur():
    def __init__(self, p1=0.3, p2=0.3):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            r_gb = T.GaussianBlur(randrange(3, 15+1, 2), (0.01, 1.5))
        elif np.random.rand() < self.p2:
            r_gb = T.RandomAdjustSharpness(np.random.randint(0, 4), 1)
        else:
            return img, mask

        return r_gb(img), mask


class RandomFilter():
    def __init__(self, p1=0.3, p2=0.3, p3=0.3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def average_filter(self, img):
        img1 = torch.zeros_like(img)
        img1[1:, 1:] = img[:-1, :-1]
        img2 = torch.zeros_like(img)
        img2[1:, :] = img[:-1, :]
        img3 = torch.zeros_like(img)
        img3[1:, :-1] = img[:-1, 1:]
        img4 = torch.zeros_like(img)
        img4[:, :-1] = img[:, 1:]
        img5 = torch.zeros_like(img)
        img5[:-1, :-1] = img[1:, 1:]
        img6 = torch.zeros_like(img)
        img6[:-1, :] = img[1:, :]
        img7 = torch.zeros_like(img)
        img7[:-1, 1:] = img[1:, :-1]
        img8 = torch.zeros_like(img)
        img8[:, 1:] = img[:, :-1]
        filtr_img = (img + img1 + img2 + img3 + img4 + img5 + img6 + img7 + img8) / 9
        return filtr_img

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            perc = np.random.randint(5, 24)  # 26 or 20
            iter = np.random.randint(1, 26 - perc)  # 30 or 26
            imge_new = img[0, :, :].clone()
            for i in range(iter):
                mask_seg = imge_new < 0.5
                imge_new[mask_seg] = imge_new[mask_seg] - ((imge_new[mask_seg] * perc) / 100)
                imge_new[~mask_seg] = imge_new[~mask_seg] - (((1.0 - imge_new[~mask_seg]) * perc) / 100)
            if np.random.rand() < self.p3:
                imge_new = self.average_filter(imge_new)
            return imge_new[None, ...], mask
        elif np.random.rand() < self.p2:
            perc = np.random.randint(5, 24)
            iter = np.random.randint(1, 26 - perc)
            imge_new = img[0, :, :].clone()
            for i in range(iter):
                mask_seg = imge_new < 0.5
                imge_new[mask_seg] = imge_new[mask_seg] + ((imge_new[mask_seg] * perc) / 100)
                imge_new[~mask_seg] = imge_new[~mask_seg] + (((1.0 - imge_new[~mask_seg]) * perc) / 100)
            if np.random.rand() < self.p3:
                imge_new = self.average_filter(imge_new)
            return imge_new[None, ...], mask
        else:
            return img, mask


class SyncRandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            return F.hflip(img), F.hflip(mask)

        return img, mask


class SyncRandomVerticalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            return F.vflip(img), F.vflip(mask)

        return img, mask


class SyncRandomDiagonalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            imgfl = F.vflip(img)
            maskfl = F.vflip(mask)
            return F.hflip(imgfl), F.hflip(maskfl)

        return img, mask


class Background_clases():
    def __init__(self, p=0.85):
        self.p = p
        # self.names = names
        # self.img_path = img_path
        # self.mask_path = mask_path
    def _flip_img(self, b_img, b_mask):
        if random.random() < 0.5:
            b_img = b_img
            b_mask = b_mask

        else:
            b_img = b_img[::-1, :, ...]
            b_mask = b_mask[::-1, :, ...]

        if random.random() < 0.5:
            b_img = b_img
            b_mask = b_mask

        else:
            b_img = b_img[:, ::-1, ...]
            b_mask = b_mask[:, ::-1, ...]

        return b_img, b_mask

    def _random_crop(self, b_name, img_path, mask_path):
        b_img = np.array(img_path[b_name])
        b_mask = np.array(mask_path[b_name])

        b_img, b_mask = self._flip_img(b_img, b_mask)
        cut_x = np.random.randint(0, 256)
        cut_y = np.random.randint(0, 256)
        b_img = b_img[cut_y:cut_y + 256, cut_x:cut_x + 256, ...]
        b_mask = b_mask[cut_y:cut_y + 256, cut_x:cut_x + 256, ...]

        return b_img, b_mask

    def _choice(self, img, mask, names, img_path, mask_path):
        try:
            b_name1 = choice(names)
            b_name2 = choice(names)
            b_name3 = choice(names)
            b_name4 = choice(names)
        except:
            raise Exception('Изображения для фона не найдены')

        b_img1, b_mask1 = self._random_crop(b_name1, img_path, mask_path)
        b_img2, b_mask2 = self._random_crop(b_name2, img_path, mask_path)
        b_img3, b_mask3 = self._random_crop(b_name3, img_path, mask_path)
        b_img4, b_mask4 = self._random_crop(b_name4, img_path, mask_path)

        part_1, part_2, part_3, part_4 = sample([[0, 0], [0, 256], [256, 0], [256, 256]], 4)
        new_img = np.zeros_like(img)
        new_mask = np.zeros_like(mask).astype(float)

        new_img[part_1[1]:part_1[1] + 256, part_1[0]:part_1[0] + 256, ...] = b_img1
        new_mask[part_1[1]:part_1[1] + 256, part_1[0]:part_1[0] + 256] = b_mask1

        new_img[part_2[1]:part_2[1] + 256, part_2[0]:part_2[0] + 256, ...] = b_img2
        new_mask[part_2[1]:part_2[1] + 256, part_2[0]:part_2[0] + 256] = b_mask2

        new_img[part_3[1]:part_3[1] + 256, part_3[0]:part_3[0] + 256, ...] = b_img3
        new_mask[part_3[1]:part_3[1] + 256, part_3[0]:part_3[0] + 256] = b_mask3

        new_img[part_4[1]:part_4[1] + 256, part_4[0]:part_4[0] + 256, ...] = b_img4
        new_mask[part_4[1]:part_4[1] + 256, part_4[0]:part_4[0] + 256] = b_mask4

        return Image.fromarray(new_img), Image.fromarray(new_mask).convert('L')

    def __call__(self, img, mask, names, img_path, mask_path):
        if np.random.rand() < self.p:
            back_img, back_mask = self._choice(img, mask, names, img_path, mask_path)
            ch_zone = Image.fromarray(np.uint8((np.array(mask).astype(bool)) * 255))
            back_img.paste(img, mask=ch_zone)
            back_mask.paste(mask, mask=ch_zone)

            return back_img, back_mask
        return img, mask


class TrickyResize_UpDwn():
    def __init__(self, resolution=512, minmax_size_up=[117, 200]):
        self.resolution = resolution
        self.minmax_size_up = minmax_size_up


    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < 0.8:
            if self.resolution:
                size_down = np.array([self.resolution, self.resolution])
            else:
                size_down = img.size

            up_l = np.random.randint(self.minmax_size_up[0], self.minmax_size_up[1])
            rand_size_up = np.array([int((size_down[0] * up_l) / 100), int((size_down[1] * up_l) / 100)])
            interpolation = choice(imodes)
            img = F.resize(img, tuple(rand_size_up), interpolation=interpolation, antialias=True)
            interpolation = choice(imodes)
            img = F.resize(img, tuple(size_down), interpolation=interpolation, antialias=True)
            return img, mask

        return img, mask


class SyncRotate360_plus():
    def __init__(self, p=0.8, p_c=0.7, resolution=512):
        self.p = p
        self.p_c = p_c
        self.resolution = resolution

    @staticmethod
    def _crop(X1, Y1, resolution_x, resolution_y, img_fc, mask_fc):
        X2, Y2 = X1 + resolution_x, Y1 + resolution_y
        new_img = img_fc.crop([X1, Y1, X2, Y2])
        new_mask = mask_fc.crop([X1, Y1, X2, Y2])

        return new_img, new_mask

    def __call__(self, img, mask, **kwargs):
        rnd_p = np.random.rand()

        if self.resolution:
            resolution_x, resolution_y = self.resolution, self.resolution
        else:
            resolution_x, resolution_y = img.size

        if rnd_p < self.p:
            resample = choice(rgba_imodes)
            angle = (random.random() * 90) # поменял * 45 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if np.random.rand() < self.p_c:
                expand = True
            else:
                expand = False
            img_r = F.rotate(img, -angle, resample, expand, None, list(np.zeros(len(img.getbands()))))
            mask_r = F.rotate(mask, -angle, resample, expand, None, list(np.zeros(len(mask.getbands()))))

            if expand:
                w, h = img_r.size
                if w < ((w * 127) / 100) and np.random.rand() < 0.5:
                    if np.random.rand() < 0.7:
                        lenimg = np.random.randint(88, 142)
                        rand_size_up = np.array([int((w * lenimg) / 100), int((h * lenimg) / 100)])
                        interpolation = choice(imodes)
                        img_r = F.resize(img_r, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                        mask_r = F.resize(mask_r, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                        w, h = img_r.size

                    delt_sz1 = (((w * 142) / 100) - w) // 2
                    delt_sz2 = (((h * 142) / 100) - h) // 2
                    pad_img = T.Pad(padding=(int(delt_sz1), int(delt_sz2)), fill=0)
                    img_r = pad_img(img_r)
                    mask_r = pad_img(mask_r)
                    w, h = img_r.size
                X1, Y1 = np.random.randint(0, w - resolution_x), np.random.randint(0, h - resolution_y)
                img_r, mask_r = self._crop(X1, Y1, resolution_x, resolution_y, img_r, mask_r)

            return img_r, mask_r

        elif rnd_p < 0.7: # корректно работает только для изображений с равными сторонами
            w, h = img.size
            if np.random.rand() < 0.7:
                lenimg = np.random.randint(88, 142)
                rand_size_up = np.array([int((w * lenimg) / 100), int((h * lenimg) / 100)])
                interpolation = choice(imodes)
                img = F.resize(img, tuple(rand_size_up), interpolation=interpolation, antialias=True)
                mask = F.resize(mask, tuple(rand_size_up), interpolation=interpolation, antialias=True)

            delt_sz1 = (((w * 142) / 100) - w) // 2
            delt_sz2 = (((h * 142) / 100) - h) // 2
            pad_img = T.Pad(padding=(int(delt_sz1), int(delt_sz2)), fill=0)
            img_r = pad_img(img)
            mask_r = pad_img(mask)
            X1, Y1 = np.random.randint(0, w - resolution_x), np.random.randint(0, h - resolution_y)
            img_r, mask_r = self._crop(X1, Y1, resolution_x, resolution_y, img_r, mask_r)
            return img_r, mask_r
        return img, mask

class SyncRandomAffine():
    # Держим в голове, что scale, shear, translate передаются в виде двух чисел.
    def __init__(
            self,
            p,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            interpolation=F.InterpolationMode.NEAREST,
            center=None,
    ):
        self.p = p
        self.degrees = [-degrees, degrees]
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = self.interpolation = interpolation
        self.center = center
        self.fill = 0

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            fill = self.fill
            channels, height, width = F.get_dimensions(img)
            img_size = [width, height]  # flip for keeping BC on get_params call
            ret = RandomAffine.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

            img = F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
            # mask = F.affine(mask, *ret, interpolation=self.interpolation, fill=fill, center=self.center)

        return img, mask


class SyncColorJitter:
    def __init__(self, p, brightness=0, contrast=0, saturation=0, hue=0):
        self.cj = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            return self.cj(img), mask
        else:
            return img, mask


class chanel_replace:
    def __init__(self, p):
        self.p = p
    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            r, g, b = img.split()
            ch = [r, g, b]
            shuffle(ch)
            img = Image.merge('RGB', (ch[0], ch[1], ch[2]))
        return img, mask


class Change_colors:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            data = img.getdata()
            resul = np.array(data)
            sign = np.random.randint(-2,2)
            if sign < 0:
                resul[:,0] = np.array(data)[:,0] - ((20*np.array(data)[:,0])//100)
                resul[:,1] = np.array(data)[:,1] - ((20*np.array(data)[:,1])//100)
                resul[:,2] = np.array(data)[:,2] - ((20*np.array(data)[:,2])//100)
            else:
                resul[:, 0] = np.array(data)[:, 0] + ((20*(255-np.array(data)[:,0]))//100)
                resul[:, 1] = np.array(data)[:, 1] + ((20*(255-np.array(data)[:,1]))//100)
                resul[:, 2] = np.array(data)[:, 2] + ((20*(255-np.array(data)[:,2]))//100)

            res = [tuple(x) for x in resul.tolist()]
            img = Image.new(data.mode, data.size)
            img.putdata(res)

        return img, mask

class SyncResize:
    def __init__(self, resolution=512):
        self.resolution = resolution

    def __call__(self, img, mask, **kwargs):
        if self.resolution:
            imsize = np.array([self.resolution, self.resolution])
            return F.resize(img, imsize), F.resize(mask, imsize, interpolation=IMode.NEAREST)
        else:
            return img, F.resize(mask, img.size, interpolation=IMode.NEAREST)



class RandomNoiseSP:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            amount = np.random.randint(0, 11) / 1000
            salt_img = torch.tensor(random_noise(img, mode='salt', amount=amount))
            return salt_img, mask
        elif np.random.rand() < self.p:
            amount = np.random.randint(0, 11) / 1000
            salt_img = torch.tensor(random_noise(img, mode='pepper', amount=amount))
            return salt_img, mask
        elif np.random.rand() < self.p:
            amount = np.random.randint(0, 11) / 1000
            salt_img = torch.tensor(random_noise(img, mode='s&p', amount=amount))
            return salt_img, mask
        return img, mask


class RandomGridDistortion:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            image = np.array(img)
            mask = np.array(mask)
            aug = A.GridDistortion(num_steps=np.random.randint(2,6), distort_limit=(-0.3, 0.3), normalized=(np.random.rand() < 0.5), p=1)
            augmented = aug(image=image, mask=mask)
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            return image, mask
        else:
            return img, mask


class RandomElasticTransform:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p:
            image = np.array(img)
            mask = np.array(mask)
            aug = A.ElasticTransform(p=1, alpha=np.random.randint(5,16), sigma=50, approximate=False)
            augmented = aug(image=image, mask=mask)
            image = Image.fromarray(augmented['image'])
            mask = Image.fromarray(augmented['mask'])
            return image, mask
        else:
            return img, mask


class RandomPaste:
    def __init__(self, p1=0.9, resolution=512):
        self.p1 = p1
        self.resolution = resolution


    def __call__(self, img, mask, backimge, backmaska, **kwargs):
        img = img.copy()
        mask = mask.copy()
        if np.random.rand() < self.p1:
            backimg = backimge.copy()
            backmask = backmaska.copy()

            img = F.resize(img, [512, 512], interpolation=IMode.BILINEAR)
            mask = F.resize(mask, [512, 512], interpolation=IMode.NEAREST)
            backimg = F.resize(backimg, [512, 512], interpolation=IMode.BILINEAR)
            backmask = F.resize(backmask, [512, 512], interpolation=IMode.NEAREST)
            bin_mask = mask.convert('L')
            bin_mask = bin_mask.point(lambda p: 255 if p > 10 else 0).convert('1')
            backimg.paste(img, (0, 0), bin_mask)
            backmask.paste(mask, (0, 0), bin_mask)
            img = backimg.copy()
            mask = backmask.copy()

        return img, mask


class RandomMedianFilter:
    def __init__(self, p1=0.3, p2=0.1, p3=0.15,):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def __call__(self, img, mask, **kwargs):
        if np.random.rand() < self.p1:
            img = img.filter(ImageFilter.MedianFilter(size = 3))
        elif np.random.rand() < self.p2:
            img = img.filter(ImageFilter.MedianFilter(size=5))
        # elif np.random.rand() < self.p3:
        #     img = img.filter(ImageFilter.MedianFilter(size=7))

        return img, mask


class SyncCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, **kwargs):
        for transform in self.transforms:
            img, mask = transform(img, mask, **kwargs)
        return img, mask


class SyncToTensor:
    def __init__(self):
        self.tt = ToTensor()

    def __call__(self, img, mask, **kwargs):
        return self.tt(img), self.tt(mask)


class AffineAugmentation:

    def __init__(
            self,
            p = 0.5,
            p_affine=1.0,
            pad_to_original=True,
            pad_border_mode=cv2.BORDER_CONSTANT,
            pad_value=0,
            rotate=(0,0),
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            shear=(-10, 10)
    ):
        """
        Args:
            p_affine: Вероятность применения.
            pad_to_original: если True, то после Affine гарантируем возврат к исходному (H, W)
                         через PadIfNeeded + CenterCrop.
            pad_border_mode/pad_value: чем заполнять области, появившиеся после трансформации.
            affine_params: rotate=(-15,15), scale=(0.9,1.1), translate_percent=(-0.1,0.1), shear=(-10,10)
        """
        self.affine = A.Affine(p=p_affine, rotate=rotate, scale=scale, translate_percent=translate_percent, shear=shear)
        self.pad_to_original = pad_to_original
        self.pad_border_mode = pad_border_mode
        self.pad_value = pad_value
        self.p = p

    def __call__(self, img_pil, target, **kwargs):
        if np.random.rand() > self.p:
            return img_pil, target

        img_np = np.array(img_pil)
        h0, w0 = img_np.shape[:2]

        masks_np = np.array(target)
        if masks_np.ndim == 3 and masks_np.shape[0] in [1, 3]:
            masks_np = np.transpose(masks_np, (1, 2, 0))
        # Albumentations ожидает список масок: [mask_ch0, mask_ch1, mask_ch2]
        # Разбиваем (H, W, 3) на 3 канала
        if masks_np.ndim == 3 and masks_np.shape[-1] > 1:
            masks_list = [masks_np[..., i] for i in range(masks_np.shape[-1])]
        else:
            # Если канал один (H, W) или (H, W, 1)
            masks_list = [masks_np.squeeze()]
        # Динамически добавляем "починку размера" под исходный (h0,w0)
        transforms = [self.affine]
        if self.pad_to_original:
            transforms += [
                A.PadIfNeeded(
                    min_height=h0,
                    min_width=w0,
                    border_mode=self.pad_border_mode,
                    fill=self.pad_value,  # чем заполнять "новые" пиксели изображения при BORDER_CONSTANT
                    fill_mask=0,  # чем заполнять "новые" пиксели масок (фон)
                    p=1.0,
                ),
                A.CenterCrop(height=h0, width=w0, p=1.0),
            ]

        pipeline = A.Compose(
            transforms,
            # Глобально задаём nearest для масок в геометрии [page:2][page:1]
            mask_interpolation=cv2.INTER_NEAREST,
        )

        res = pipeline(image=img_np, masks=masks_list)
        img_aug = res["image"]
        masks_aug_list = res["masks"]

        # 5. Собираем маску обратно в (H, W, 3) и конвертируем в PIL
        masks_aug_np = np.stack(masks_aug_list, axis=-1)  # -> (H, W, 3)
        masks_aug_pil = Image.fromarray(masks_aug_np.astype(np.uint8))

        return Image.fromarray(img_aug), masks_aug_pil


class SyncRandomBrightnessContrastTarget:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img, target, **kwargs) -> tuple:
        if np.random.rand() > self.p:
            return img, target

        if np.random.rand() > self.p:
            jittered_img = self.jitter(img)
            return jittered_img, target

        enhanced_img = F.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness))
        enhanced_img = F.adjust_contrast(enhanced_img, 1 + random.uniform(-self.contrast, self.contrast))

        return enhanced_img, target


class Test_error:
    def __init__(self, indx):
        self.indx = indx
        self.count = 1

    def __call__(self, img, mask, names, img_path, mask_path):
        testimg = np.array(img)
        testmask = np.array(mask)
        # if testimg.shape != testmask.shape:
        #     print(f'image - {testimg.shape}, mask - {testmask.shape}')
        #     print(names, img_path, mask_path)


        if (((testimg[:,:,0] == 0) & (testimg[:,:,1] == 0) & (testimg[:,:,2] == 0)) == (testmask==0)).mean() < 0.9:
            print(f'индекс{self.indx}')
            img.save(f'/home/user/PycharmProjects/deeplab_RLINS3_Optic/notebook/images/{self.count}img_{self.indx}.png')
            mask.save(f'/home/user/PycharmProjects/deeplab_RLINS3_Optic/notebook/images/{self.count}masl_{self.indx}.png')
            self.count += 1
        return img, mask,

class RandomGrayMaker:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target: Dict[str, torch.Tensor], **kwargs) -> tuple:
        if random.random() > self.p:
            return img, target

        return img.convert('L'), target


class SyncTimeMosaicStitching:
    def __init__(self, p=0.5, resolution=512):
        """
        Аугментация для случайной сшивки фрагментов из одной временной группы.
        Args:
            p: Вероятность применения аугментации.
            resolution: Размер стороны квадратного фрагмента.
        """
        self.p = p
        self.resolution = resolution

    def __call__(self, img, mask, **kwargs):
        # Извлекаем переданный из датасета контекст группы
        group_patches_data = kwargs.get('group_patches_data', [])
        dataset_obj = kwargs.get('dataset_obj', None)
        patch_coords = kwargs.get('patch_coords', None)
        needs_padding = kwargs.get('needs_padding', False)
        original_size = kwargs.get('original_size', (img.width, img.height))

        # Если аугментация не выпала по вероятности или группа пуста — возвращаем как есть
        if np.random.rand() > self.p or not group_patches_data or dataset_obj is None or patch_coords is None:
            return img, mask

        # 1. Определяем, сколько избыточных изображений из группы мы возьмем (от 1 до 3, чтобы суммарно было от 2 до 4 фрагментов)
        max_available = len(group_patches_data)
        num_to_take = random.randint(1, min(3, max_available))
        chosen_alternatives = random.sample(group_patches_data, num_to_take)
        total_fragments = 1 + len(chosen_alternatives)

        # Подготавливаем списки всех фрагментов снимков и их масок
        all_imgs = [img]
        all_masks_np = [np.array(mask)]  # (H, W, 3)

        # 2. Извлекаем и подготавливаем фрагменты для выбранных альтернативных изображений
        for alt_data in chosen_alternatives:
            try:
                alt_img_full = Image.open(alt_data['image_path']).convert('RGB')
                # Масштабируем исходное изображение, если в датасете активен size_scale
                if getattr(dataset_obj, 'size_scale', 1.0) != 1.0:
                    new_w = int(alt_img_full.width * dataset_obj.size_scale)
                    new_h = int(alt_img_full.height * dataset_obj.size_scale)
                    alt_img_full = alt_img_full.resize((new_w, new_h), Image.Resampling.BILINEAR)

                # Вырезаем точно такой же фрагмент
                alt_patch = alt_img_full.crop(patch_coords)

                # Создаем маску для альтернативного патча через метод датасета
                alt_target = dataset_obj._create_target_for_patch(alt_data['annotations'], patch_coords)

                if needs_padding:
                    alt_patch = dataset_obj._pad_image(alt_patch, original_size)
                    alt_target = dataset_obj._pad_target(alt_target, self.resolution)

                # Конвертируем маску в привычный формат (H, W, 3) со значениями 0 и 255
                alt_mask_np = alt_target['masks'].cpu().numpy()
                alt_mask_np = np.transpose(alt_mask_np, (1, 2, 0))
                alt_mask_np = (alt_mask_np.astype(np.uint8) * 255)

                all_imgs.append(alt_patch)
                all_masks_np.append(alt_mask_np)
            except Exception as e:
                print(f"Ошибка загрузки альтернативного кадра при мозаике: {e}")
                continue

        if len(all_imgs) < 2:
            return img, mask

        # 3. Генерация случайных линий разделения через построение растровых регионов Вороного
        h, w = img.height, img.width
        num_regions = len(all_imgs)

        # Генерируем центры регионов так, чтобы они не лепились в одну точку
        centers = []
        for _ in range(num_regions):
            centers.append([random.randint(0, w), random.randint(0, h)])

        # Строим карту принадлежности каждого пикселя к ближайшему центру
        y_grid, x_grid = np.indices((h, w))
        dist_matrix = np.zeros((num_regions, h, w))
        for i, (cx, cy) in enumerate(centers):
            dist_matrix[i] = (x_grid - cx) ** 2 + (y_grid - cy) ** 2

        region_map = np.argmin(dist_matrix, axis=0)  # Матрица (H, W) со значениями от 0 до total_fragments-1

        # 4. Сборка сшитого (мозаичного) изображения и базовой маски
        stitched_img_np = np.zeros((h, w, 3), dtype=np.uint8)
        stitched_mask_np = np.zeros_like(all_masks_np[0], dtype=np.uint8)

        for i in range(len(all_imgs)):
            img_arr = np.array(all_imgs[i])
            mask_arr = all_masks_np[i]
            # Создаем бинарную маску (0 или 1) с размерностью (H, W, 1) для автоматического бродкастинга
            mask_multiplier = (region_map == i).astype(np.uint8)[..., None]

            # Надежная сборка через умножение и попиксельное сложение кадров
            stitched_img_np += (img_arr * mask_multiplier).astype(np.uint8)
            stitched_mask_np += (mask_arr * mask_multiplier).astype(np.uint8)

        # 5. Возвращаем результат (никакой дополнительной логики не нужно!)
        res_img = Image.fromarray(stitched_img_np)
        res_mask = Image.fromarray(stitched_mask_np)

        return res_img, res_mask



if __name__ == '__main__':
    sync_affine = SyncRandomAffine(60, [0.05, 0.05], [0.8, 1.2], [-20, 20])
    sync_collor = SyncColorJitter(brightness=0.7, contrast=0.9, saturation=0.6, hue=0)
    sync_hor = SyncRandomHorizontalFlip()
    sync_comp = SyncCompose([sync_hor, sync_affine, sync_collor])
    sync_comp = SyncCompose([])

    f, axes = plt.subplots(4, 3, figsize=(12, 20))
    blended = blend(img, mask)

    axes[0][0].imshow(img)
    axes[0][1].imshow(mask)
    axes[0][2].imshow(blended)

    for i in range(1, 4):
        timg, tmask = sync_comp(img, mask)
        blended = blend(timg, tmask)

        axes[i][0].imshow(timg)
        axes[i][1].imshow(tmask)
        axes[i][2].imshow(blended)

    plt.tight_layout()
    plt.show()
