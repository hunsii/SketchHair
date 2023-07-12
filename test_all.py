import argparse
from argparse import Namespace

import os
import cv2
from e4e import E4E
from sketch import Sketch
from mapper import run_mapper

import yaml

def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    opt = Namespace(**data)
    return opt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', help='path to config yaml file')
    args = parse_yaml(parser.parse_args().config)

    e4e = E4E(args)
    latent, img = e4e.run(file_path=args.img_path)
    mapper, mapperBM = run_mapper(args, latent, img)

    args.save_dict = {
        'Origin': cv2.imread(args.img_path), 
        'E4E': img, 
        'HairMapper': mapper, 
        'HairMapperBM':mapperBM, 
        'HairSalon': None, 
        'HairSalonSketch1': None, # user sketch
        'HairSalonSketch2': None, # for S2M
        'HairSalonSketch3': None, # for S2I
        'HairSalonMatte': None, 
        'HairSalonHair': None, 
        'HairSalonE4E': None, 
        'HairSalonE4EBM': None, 
    }
    sketch = Sketch(args, e4e)
    args, sketched_img, hair_matte = sketch.draw(mapperBM)
    cv2.imwrite('test.png', sketched_img)

    print("done!")