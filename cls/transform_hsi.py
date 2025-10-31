import numpy as np
from scipy.ndimage import rotate


def rotate_hsi_mask(image, angle, mask=False):
    """
    旋转高光谱图像给定角度，保持图像尺寸不变。
    """
    # 获取图像尺寸
    if not mask:
        h, w, s = image.shape
    else:
        h, w = image.shape
    # 计算旋转后的图像尺寸
    # new_h = int(np.ceil(h * np.abs(np.cos(np.radians(angle))) + w * np.abs(np.sin(np.radians(angle)))))
    # new_w = int(np.ceil(h * np.abs(np.sin(np.radians(angle))) + w * np.abs(np.cos(np.radians(angle)))))
    new_h = h + w
    new_w = h + w
    # 计算旋转后的中心点
    center_x = new_w // 2
    center_y = new_h // 2

    # 创建一个扩展后的图像数组，用于保存旋转后的图像
    if not mask:
        expanded_image = np.zeros((new_h, new_w, s), dtype=image.dtype)
    else:
        expanded_image = np.zeros((new_h, new_w), dtype=image.dtype)
    # 将原始图像复制到扩展后的图像数组中心

    if not mask:
        # print(image.shape, angle, new_h, new_w)
        expanded_image[center_y - h // 2:center_y - h // 2 + h, center_x - w // 2:center_x - w // 2 + w, :] = image
    else:
        expanded_image[center_y - h // 2:center_y - h // 2 + h, center_x - w // 2:center_x - w // 2 + w] = image

    # 执行旋转
    rotated_image = rotate(expanded_image, angle, reshape=False, mode='nearest')

    if not mask:
        return rotated_image[center_y - h // 2:center_y - h // 2 + h, center_x - w // 2:center_x - w // 2 + w, :]
    else:
        return rotated_image[center_y - h // 2:center_y - h // 2 + h, center_x - w // 2:center_x - w // 2 + w]


def flip_hsi(image, direction):
    """
    在水平或垂直方向翻转高光谱图像。
    """
    if direction == 'horizontal':
        flipped_image = np.fliplr(image)
    elif direction == 'vertical':
        flipped_image = np.flipud(image)
    else:
        raise ValueError("Invalid direction. Choose 'horizontal' or 'vertical'.")

    return flipped_image


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio


def apply_aug(aug, image, mask=None):
    if mask is None:
        return aug(image=image)['image']
    else:
        augment = aug(image=image, mask=mask)
        return augment['image'], augment['mask']


class Transform:
    def __init__(self, size=None, Rotate_ratio=0., Flip_ratio=0., cls=False):

        self.size = size

        self.Rotate_ratio = Rotate_ratio
        self.Flip_ratio = Flip_ratio
        self.cls = cls

    def __call__(self, example):
        
        if self.cls:
            x = example
        else:
            x, y = example

        # --- Augmentation ---
        # --- Train/Test common preprocessing ---

        # albumentations...

        # # 1. blur

        if _evaluate_ratio(self.Rotate_ratio):
            angle = np.random.randint(-90, 90)
            x = rotate_hsi_mask(x, angle)
            if not self.cls:
                y = rotate_hsi_mask(y, angle, mask=True)


        if _evaluate_ratio(self.Flip_ratio):
            mode = np.random.randint(-1, 1)
            if not self.cls:
                if mode == 0: 
                    x = flip_hsi(x, 'vertical')
                    y = flip_hsi(y, 'vertical')
                elif mode == 1:
                    x = flip_hsi(x, 'horizontal')
                    y = flip_hsi(y, 'horizontal')
                else:
                    x = flip_hsi(x, 'vertical')
                    y = flip_hsi(y, 'vertical')
                    x = flip_hsi(x, 'horizontal')
                    y = flip_hsi(y, 'horizontal')

            else:
                if mode == 0: 
                    x = flip_hsi(x, 'vertical')
                elif mode == 1:
                    x = flip_hsi(x, 'horizontal')
                else:
                    x = flip_hsi(x, 'vertical')
                    x = flip_hsi(x, 'horizontal')
        if self.cls:
            return x
        return x, y
            


        

# if __name__ == '__main__':
#     from scipy.io import loadmat
#     import cv2
#     hsi = loadmat('D:/PLGC_small_size/GIN/image_0/2020-80335-9-10x-roi1-mono.mat')['hsi']
#     hsi_r = rotate_hsi_mask(hsi, 45)
#     hsi_img = hsi[:, :, 0] * 255
#     hsi_img_r = hsi_r[:, :, 0] * 255
#     # hsi_img = cv2.normalize(hsi[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
#     # hsi_img_r = cv2.normalize(hsi_r[:, :, 0], None, 0, 255, cv2.NORM_MINMAX)
#     cv2.imwrite('1.png', hsi_img)
#     cv2.imwrite('2.png', hsi_img_r)

if __name__ == '__main__':
    import torch
    r = torch.randn((2, 196, 196))
    r = torch.argmax(r, dim=0, keepdim=True)
    print(r.shape)