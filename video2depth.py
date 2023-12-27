import os
import argparse
import time
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth

import matplotlib as mpl
import matplotlib.cm as cm
import cv2

# using cpu
ctx = mx.cpu(0)


data_path = os.path.expanduser("~/Replica/office0/results")

files = os.listdir(data_path)
files.sort()

count = 1
raw_img_sequences = []
for file in files:
    
    file = os.path.join(data_path, file)
    img = pil.open(file).convert('RGB')
    raw_img_sequences.append(img)
    if (count > 1000):
        break
    count += 1 
    

original_width, original_height = raw_img_sequences[0].size

model_zoo = 'monodepth2_resnet18_kitti_mono_stereo_640x192'
model = gluoncv.model_zoo.get_model(model_zoo, pretrained_base=False, ctx=ctx, pretrained=True)

min_depth = 0.1
max_depth = 100

# while use stereo or mono+stereo model, we could get real depth value
scale_factor = 6 # 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 80

feed_height = 192
feed_width = 640

count = 1
pred_depth_sequences = []
pred_disp_sequences = []
for img in raw_img_sequences:
    img = img.resize((feed_width, feed_height), pil.LANCZOS)
    img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

    outputs = model.predict(img)
    mx.nd.waitall()
    pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
    t = time.time()
    pred_disp = pred_disp.squeeze().as_in_context(mx.cpu()).asnumpy()
    pred_disp = cv2.resize(src=pred_disp, dsize=(original_width, original_height))
    pred_disp_sequences.append(pred_disp)

    pred_depth = 1 / pred_disp
    pred_depth *= scale_factor
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    pred_depth_sequences.append(pred_depth)
    if (count % 100 == 0):
        print("Good news!", count)
    count += 1


# Output
output_path = os.path.join(os.path.expanduser("."), "tmp")

pred_path = os.path.join(output_path, 'pred')
if not os.path.exists(pred_path):
    os.makedirs(pred_path)


for pred, file in zip(pred_depth_sequences, files):
    file_name = os.path.basename(file)
    file_list = ["depth", file_name[5:11], ".png"]
    # file_list = ["depth", file_name[5:7], "1", file_name[8:11], ".png"]
    pred_out_file = os.path.join(pred_path, "".join(file_list))
    cv2.imwrite(pred_out_file, pred)

rgb_path = os.path.join(output_path, 'rgb')
if not os.path.exists(rgb_path):
    os.makedirs(rgb_path)


output_sequences = []
for raw_img, pred, file in zip(raw_img_sequences, pred_disp_sequences, files):
    vmax = np.percentile(pred, 95)
    normalizer = mpl.colors.Normalize(vmin=pred.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(pred)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    raw_img = np.array(raw_img)
    pred = np.array(im)
    output = np.concatenate((raw_img, pred), axis=0)
    output_sequences.append(output)

    pred_out_file = os.path.join(rgb_path, file)
    cv2.imwrite(pred_out_file, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

width = int(output_sequences[0].shape[1] + 0.5)
height = int(output_sequences[0].shape[0] + 0.5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    os.path.join(output_path, 'demo.mp4'), fourcc, 20.0, (width, height))

for frame in output_sequences:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    out.write(frame)
    # uncomment to display the frames
    # cv2.imshow('demo', frame)

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #    break
