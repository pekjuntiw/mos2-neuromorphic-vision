import matplotlib.image as mpimg
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import os

h, w = 100, 344
fs_img_pos = glob('karlsruhe\\objects_2011_a\\labeldata\\pos\\*.png')
fs_img_neg = glob('karlsruhe\\objects_2011_a\\labeldata\\neg\\*.png')
output_dir = f'karlsruhe\\objects_2011_a\\labeldata_{w}x{h}'

if not os.path.exists(f'{output_dir}\\pos'):
    os.makedirs(f'{output_dir}\\pos')
if not os.path.exists(f'{output_dir}\\neg'):
    os.makedirs(f'{output_dir}\\neg')

for f in tqdm(fs_img_pos):
    # load image
    img = mpimg.imread(f)

    # normalize and convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8') if img.max() <= 1 else img.astype('uint8')

    img_small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    f_name = f.split('\\')[-1]
    cv2.imwrite(f'{output_dir}\\pos\\{f_name}', img_small)

for f in tqdm(fs_img_neg):
    # load image
    img = mpimg.imread(f)

    # normalize and convert to uint8 if necessary
    if img.dtype != np.uint8:
        img = (img * 255).astype('uint8') if img.max() <= 1 else img.astype('uint8')

    img_small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    f_name = f.split('\\')[-1]
    cv2.imwrite(f'{output_dir}\\neg\\{f_name}', img_small)


