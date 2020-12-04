import random
import os
import sys
import numpy as np
import cv2
import math
from tqdm import tqdm


def make_voc_bg(root):
  """
  root: path to root voc2012 
  """

  jpegimages = "JPEGImages"
  trainval = 'ImageSets/Main/trainval.txt'
  bg_txt = (os.path.join(root, trainval))
  with open(bg_txt, 'r') as f:
    voc_image_path = f.read().split('\n')
  voc_image_path.pop(-1)
  bg_paths = []
  for each in voc_image_path:
    jpeg = os.path.join(root, jpegimages, each + '.jpg')
    bg_paths.append(jpeg)
  return bg_paths


def composite(inp, mask, bg):
  '''inp = cv2.imread(inp)
  trimap = cv2.imread(trimap, 0)
  mask = cv2.imread(mask, 0)
  bg = cv2.imread(bg)'''
  ###
  h, w = inp.shape[:2]
  bh, bw = bg.shape[:2]
  wratio = w / bw
  hratio = h / bh
  ratio = wratio if wratio > hratio else hratio
  if ratio > 1:
      bg = cv2.resize(src=bg, dsize=(math.ceil(bw * ratio),
                                     math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)
  #
  bg_h, bg_w = bg.shape[:2]
  #
  fg = np.array(inp, np.float32)
  x = 0
  if bg_w > w:
      x = np.random.randint(0, bg_w - w)
  y = 0
  if bg_h > h:
      y = np.random.randint(0, bg_h - h)
  bg = np.array(bg[y:y + h, x:x + w], np.float32)
  alpha = np.zeros((h, w, 1), np.float32)
  alpha[:, :, 0] = mask / 255.
  im = alpha * fg + (1 - alpha) * bg
  im = im.astype(np.uint8)

  return im


def composite_data_with_voc_bg(voc_bg_root,
                               input_image_folder,
                               input_alpha_folder,
                               output_folder,
                               input_trimap_folder=None,
                               num_random_bg=100,
                               ):
  """
  output_folder_ 
                |_ input
                |_ mask
                |_ trimap
  """
  # read txt JPEG image of voc
  voc_bgs = make_voc_bg(voc_bg_root)
  out_image_folder = os.path.join(output_folder, 'input')
  out_alpha_folder = os.path.join(output_folder, 'mask')
  inp_images = [os.path.join(input_image_folder, each)
                for each in os.listdir(input_image_folder)]
  inp_alphas = [os.path.join(input_alpha_folder, each)
                for each in os.listdir(input_alpha_folder)]
  out_masks_folder = [out_alpha_folder]
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  if not os.path.exists(out_image_folder):
    os.makedirs(out_image_folder)
  if not os.path.exists(out_alpha_folder):
    os.makedirs(out_alpha_folder)
  if input_trimap_folder:
    out_trimap_folder = os.path.join(output_folder, 'trimap')
    if not os.path.exists(out_trimap_folder):
        os.mkdir(out_trimap_folder)
    inp_trimaps = [os.path.join(input_trimap_folder, each)
                   for each in os.listdir(input_trimap_folder)]
    input_data = list(zip(inp_images, inp_alphas, inp_trimaps))
    out_masks_folder.append(out_trimap_folder)
  else:
    input_data = list(zip(inp_images, inp_alphas))

  def save(name, inp, mask):
    cv2.imwrite(os.path.join(out_image_folder, name), inp)
    for i in range(len(mask)):
      cv2.imwrite(os.path.join(out_masks_folder[i], name), mask[i])

  with tqdm(total=len(input_data) * (num_random_bg + 1)) as bar:

    for i in range(len(input_data)):
        inp, *masks = input_data[i]  # img, alpha, trimap if yes
        img = cv2.imread(inp)
        masks = [cv2.imread(mask, 0) for mask in masks]
        img_file = os.path.split(inp)[-1]
        img_name = img_file.split('.')[0]
        random_bgs = random.sample(voc_bgs, num_random_bg)
        save(img_file, img, masks)
        bar.set_description(f'{img_name}')
        bar.update(1)
        for bg in random_bgs:
          bg_name = os.path.split(bg)[-1].split('.')[0]
          img_bg = cv2.imread(bg)
          new_img = composite(img, masks[0], img_bg)
          new_img_name = img_name + '_' + bg_name + '.png'
          save(new_img_name, new_img, masks)
          bar.set_description(f'{img_name} with {bg_name}')
          bar.update(1)


'''
composite_data_with_voc_bg(voc_bg_root='/content/VOCdevkit/VOC2012',
                           input_image_folder='/content/alphamatting_input',
                           input_alpha_folder='/content/alphamatting_mask',
                           output_folder='/content/OUTPUT',
                           input_trimap_folder=None,  # '/content/alphamatting_trimap/Trimap1',
                           num_random_bg=100,
                           )
'''                           
