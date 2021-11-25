import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture


def main(inputDir, outputDir, gpu, isDlib=True):

    # ---- init PRN
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  # GPU number, -1 for CPU
    prn = PRN(is_dlib=True)

    # ------------- load data
    image_folder = inputDir
    save_folder = outputDir
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    types = ('*.jpg', '*.png')  # 如果有新的图片类型就添加到这里
    image_path_list = []
    for files in types:
        # 找到改目录下所有jpg和png文件
        image_path_list.extend(glob(os.path.join(image_folder, files)))
    # total_num = len(image_path_list)
    # print(total_num, image_path_list)

    for i, image_path in enumerate(image_path_list):
        name = image_path.strip().split('\\')[-1][:-4]
        # read image
        image = imread(image_path)
        print(image.shape)
        [h, w, c] = image.shape
        if c > 3:
            image = image[:, :, :3]

        # the core: regress position map
        if isDlib:
            max_size = max(image.shape[0], image.shape[1])
            if max_size > 1000:
                image = rescale(image, 1000. / max_size)
                image = (image * 255).astype(np.uint8)
            pos = prn.process(image)  # use dlib to detect face
        else:
            if image.shape[0] == image.shape[1]:
                image = resize(image, (256, 256))
                pos = prn.net_forward(image / 255.)  # input image has been cropped to 256x256
            else:
                box = np.array([0, image.shape[1] - 1, 0, image.shape[0] - 1])  # cropped with bounding box
                pos = prn.process(image, box)

        image = image / 255.
        if pos is None:
            continue
        vertices = prn.get_vertices(pos)
        depth_image = get_depth_image(vertices, prn.triangles, h, w, True)
        imsave(os.path.join(save_folder, name + '_depth.jpg'), depth_image)



if __name__ == '__main__':
    inputDir = 'OriginalImages'
    outputDir = 'DepthMaps'
    gpu = '0'  # set gpu id, -1 for CPU
    main(inputDir, outputDir, gpu, True)

