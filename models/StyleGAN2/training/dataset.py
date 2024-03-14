import os
import math
import numpy as np
import PIL.Image
import json
import functools
import io
from typing import Union
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
from utils.visualize import visualize_training

def error(msg):
    print('Error: ' + msg)
    sys.exit(1)

def make_transform(
    transform: str,
    output_width: int,
    output_height: int,
    resize_filter: str
):
    resample = { 'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS }[resize_filter]
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            error('must specify --width and --height when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            error('must specify --width and --height when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'

def open_image_folder(source_dir, max_images=None):
    input_images = []
    for f in sorted(Path(source_dir).rglob('*')):
        ext = str(f).split('.')[-1].lower()
        if f'.{ext}' in PIL.Image.EXTENSION:
            input_images.append(str(f))
    # Load labels.
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    if max_images is not None:
        max_idx = max_images
    else:
        max_idx = len(input_images)

    def iterate_images():
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()

def construct_dataset():
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('--source', default=None, type=str, help='path of data')
    parser.add_argument('--save-path', default=None, type=str, help='output path')
    parser.add_argument('--transform', default='center-crop-wide', type=str, help='transform method')
    parser.add_argument('--resize-filter', default='box', type=str, help='resize filter')
    parser.add_argument('--width', default=128, type=int, help='image width')
    parser.add_argument('--height', default=128, type=int, help='image height')
    parser.add_argument('--max-images', default=None, type=int, help='max image number')
    args = parser.parse_args()
    
    PIL.Image.init()

    d_path = os.path.join(args.save_path, os.path.basename(args.source))
    if os.path.isdir(d_path) and len(os.listdir(d_path)) != 0:
        error('dataset save path must be empty')
    os.makedirs(d_path, exist_ok=True)

    num_files, input_iter = open_image_folder(args.source, max_images=args.max_images)
    transform_image = make_transform(args.transform, args.width, args.height, args.resize_filter)

    dataset_attrs = None

    def save_file(fname: str, data: Union[bytes, str]):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as fout:
            if isinstance(data, str):
                data = data.encode('utf8')
            fout.write(data)

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        lab = image['label']
        if lab is None:
            archive_fname = f'0/img{idx_str}.png'
        else:
            archive_fname = f'{lab}/img{idx_str}.png'

        img = transform_image(image['img'])

        if img is None:
            continue
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            args.width = dataset_attrs['width']
            height = dataset_attrs['height']
            if args.width != height:
                error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                error('Input images must be stored as RGB or grayscale')
            if args.width != 2 ** int(np.floor(np.log2(args.width))):
                error('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_file(os.path.join(args.save_path, os.path.basename(args.source), archive_fname), image_bits.getbuffer())
        # print("saved image at", os.path.join(args.save_path, archive_fname))
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {
        'labels': labels if all(x is not None for x in labels) else None
    }
    save_file(os.path.join(args.save_path, os.path.basename(args.source), 'dataset.json'), json.dumps(metadata))

if __name__ == "__main__":
    construct_dataset()