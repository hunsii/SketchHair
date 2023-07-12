import sys
sys.path.insert(0, "HairMapper/encoder4editing")

from argparse import Namespace

import torch
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
from PIL import ImageFile
import glob
import os
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True


from models.psp import pSp
from utils.common import tensor2im

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="test_data",
                        help='Directory to save the results. If not specified, '
                             '`data/double_chin_pair/images` will be used by default.')
    parser.add_argument('--bald', action='store_true',
                                 help='save image if you want')
    return parser.parse_args()

def run_on_batch(inputs, net):
    latents, images = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True, resize=False)
    return latents, images

def run(args):
    img_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    model_path = "HairMapper/ckpts/e4e_ffhq_encode.pt"
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts= Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()

    if args.bald:
        file_dir = os.path.join(args.data_dir,'origin_bald')
    else:
        file_dir = os.path.join(args.data_dir,'origin')
    code_dir = os.path.join(args.data_dir,'code')
    res_dir = os.path.join(args.data_dir,'mapper_res')
    final_dir = os.path.join(args.data_dir,'final')
    final2_dir = os.path.join(args.data_dir,'final_BM')
    if not os.path.exists(code_dir):
        os.mkdir(code_dir)
    if not os.path.exists(final_dir):
        os.mkdir(final_dir)
    if not os.path.exists(final2_dir):
        os.mkdir(final2_dir)
    for file_path in glob.glob(os.path.join(file_dir,'*.png'))+glob.glob(os.path.join(file_dir,'*.jpg')):
      name = os.path.basename(file_path)[:-4]
      code_path =os.path.join(code_dir,f'{name}.npy')

      input_image = PIL.Image.open(file_path)
      transformed_image = img_transforms(input_image)
      with torch.no_grad():
        latents, images = run_on_batch(transformed_image.unsqueeze(0), net)
        if not args.bald:
            latent = latents[0].cpu().numpy()
            latent = np.reshape(latent,(1,18,512))
            np.save(code_path, latent)
            # PIL_image = tensor2im(images[0])
            # PIL_image.save(os.path.join(res_dir, f'{name}.png'))
            print(f'save to {code_path}')
            
        else:
            image = images[0]
            PIL_image = tensor2im(image)

            PIL_image.save(os.path.join(final_dir, f'{name}.png'))

            # use numpy to convert the pil_image into a numpy array
            numpy_image=np.array(PIL_image)  

            # convert to a openCV2 image and convert from RGB to BGR format
            opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            origin_path = os.path.join(file_dir, f"{name}.png")
            mask_path = os.path.join("output", "matte", f"{name}.png")
            origin_img = cv2.imread(origin_path)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.resize(mask_img, (1024, 1024))
            # mask_img = np.where(mask_img == 0, 255, 0).astype(np.uint8)
            # print(origin_img.shape)
            # print(opencv_image.shape)
            # print(mask_img.shape)
            cv2.imwrite(os.path.join(final2_dir, f'{name}.png'), blending_from_e4e(origin_img, opencv_image, mask_img))
            print(f"save to {name}.png")
        

import cv2
def blending_from_e4e(origin_img, edited_img, stroke_img):
    result_img = stroke_img.copy()

    mask_dilate = cv2.dilate(result_img,
                                kernel=np.ones((50, 50), np.uint8))
    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(30, 30))
    mask_dilate_blur = (result_img + (255 - result_img) / 255 * mask_dilate_blur).astype(np.uint8)

    face_mask = 255 - mask_dilate_blur

    row_mask = np.where(np.sum(mask_dilate_blur, axis=0) > 0, 1, 0)
    col_mask = np.where(np.sum(mask_dilate_blur, axis=1) > 0, 1, 0)

    min_x = np.argmax(row_mask)
    min_y = np.argmax(col_mask)
    max_x = 1023 - np.argmax(np.flip(row_mask, axis=0))
    max_y = 1023 - np.argmax(np.flip(col_mask, axis=0))

    cx = int((min_x + max_x) / 2)
    cy = int((min_y + max_y) / 2)
    center = (cx, cy)
    # cv2.imshow(f"test1, {origin_img.shape}", origin_img)
    # cv2.imshow(f"test2, {edited_img.shape}", edited_img)
    # cv2.imshow(f"test3, {face_mask.shape}", face_mask)
    

    mixed_clone = cv2.seamlessClone(origin_img, edited_img, face_mask, center, cv2.NORMAL_CLONE)
    # cv2.imshow("test4", mixed_clone)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return mixed_clone

import argparse
if __name__ == '__main__':
    run(parse_args())