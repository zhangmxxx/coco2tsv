# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os
import os.path as op
import json
import cv2
import base64
import numpy as np
import math
from tqdm import tqdm
import io
from PIL import Image
import yaml
import errno
from pycocotools.coco import COCO

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def tsv_writer(values, tsv_file, sep='\t'):
    mkdir(op.dirname(tsv_file))
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    idx = 0
    tsv_file_tmp = tsv_file + '.tmp'
    lineidx_file_tmp = lineidx_file + '.tmp'
    with open(tsv_file_tmp, 'w') as fp, open(lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value is not None
            value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
            v = '{0}\n'.format(sep.join(map(str, value)))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_tmp, tsv_file)
    os.rename(lineidx_file_tmp, lineidx_file)

def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

def config_save_file(tsv_file, save_file=None, append_str='.new.tsv'):
    if save_file is not None:
        return save_file
    return op.splitext(tsv_file)[0] + append_str

def generate_linelist_file(label_file, save_file=None, ignore_attrs=()):
    # generate a list of image that has labels
    # images with only ignore labels are not selected.
    line_list = []
    rows = tsv_reader(label_file)
    for i, row in tqdm(enumerate(rows)):
        labels = json.loads(row[1])
        if labels:
            if ignore_attrs and all([any([lab[attr] for attr in ignore_attrs if attr in lab]) \
                                for lab in labels]):
                continue
            line_list.append([i])

    save_file = config_save_file(label_file, save_file, '.linelist.tsv')
    tsv_writer(line_list, save_file)

json_file_path = "./zhiyuan_objv2_train.json" # path to anno file
coco = COCO(json_file_path)

# To generate a tsv file:
data_path = "./merged_dir"
output_path = "./output/"
tsv_file = op.join(output_path, "train.tsv")
label_file = op.join(output_path, "train.label.tsv")
hw_file = op.join(output_path, "train.hw.tsv")
linelist_file = op.join(output_path, "train.linelist.tsv")

rows = []
rows_label = []
rows_hw = []
# current offset (lineidx)
tsv_idx = 0
label_idx = 0
hw_idx = 0

mkdir(op.dirname(tsv_file))

cnt = 0
for img_id in coco.getImgIds():
    cnt = cnt + 1 # starting from 1
    # train.tsv
    img_meta = coco.loadImgs(ids=img_id)[0] # img_id: id in the dataset
    img_key = img_meta["id"]
    raw_img_path = img_meta["file_name"] # images/v1/patch2/objects365_v1_index.jpg
    [img_dir, img_p] = os.path.split(raw_img_path)
    img_path = op.join(data_path, img_p) # data_path/objects365_v1_index.jpg
    img = cv2.imread(img_path)
    # skip empty image (as a single-img loss to the train dataset)
    if img is None:
        continue
    img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img)[1])

    # train.label.tsv.
    # need to be recorded: iscrowd:bool, id:int, area:float, class:str, rect:float tuple
    # load original anno_file to see the original json label names
    labels = []
    anno_ids = coco.getAnnIds(imgIds=img_id)
    for anno_id in anno_ids:
        anno_meta = coco.loadAnns(ids=anno_id)[0]
        labels.append({"iscrowd": anno_meta["iscrowd"],
                       "id": anno_meta["id"],
                       "area": anno_meta["area"],
                       "class": coco.loadCats(ids=anno_meta["category_id"])[0]["name"],
                       "rect": anno_meta["bbox"]})

    row = [img_key, img_encoded_str]
    rows.append(row)

    row_label = [img_key, json.dumps(labels)]
    rows_label.append(row_label)

    height = img.shape[0]
    width = img.shape[1]
    row_hw = [img_key, json.dumps([{"height":height, "width":width}])]
    rows_hw.append(row_hw)

    print(f"{cnt}/1742292")

    if cnt % 1000 == 0 or cnt == 1742292:
        print("writing to tsv files")
        # tsv_file
        sep='\t'
        values = rows
        lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
        with open(tsv_file, 'a') as fp, open(lineidx_file, 'a') as fpidx:
            assert values is not None
            for value in values:
                assert value is not None
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
                v = '{0}\n'.format(sep.join(map(str, value)))
                fp.write(v)
                fpidx.write(str(tsv_idx) + '\n')
                tsv_idx = tsv_idx + len(v)
        # label_file
        values = rows_label
        lineidx_file = op.splitext(label_file)[0] + '.lineidx'
        with open(label_file, 'a') as fp, open(lineidx_file, 'a') as fpidx:
            assert values is not None
            for value in values:
                assert value is not None
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
                v = '{0}\n'.format(sep.join(map(str, value)))
                fp.write(v)
                fpidx.write(str(label_idx) + '\n')
                label_idx = label_idx + len(v)
        # hw_file
        values = rows_hw
        lineidx_file = op.splitext(hw_file)[0] + '.lineidx'
        with open(hw_file, 'a') as fp, open(lineidx_file, 'a') as fpidx:
            assert values is not None
            for value in values:
                assert value is not None
                value = [v if type(v)!=bytes else v.decode('utf-8') for v in value]
                v = '{0}\n'.format(sep.join(map(str, value)))
                fp.write(v)
                fpidx.write(str(hw_idx) + '\n')
                hw_idx = hw_idx + len(v)

        # reset buffer
        rows=[]
        rows_label=[]
        rows_hw=[]
# generate linelist file
generate_linelist_file(label_file, save_file=linelist_file)