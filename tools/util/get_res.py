import sys
sys.path.append('tools/util')

import cv2
import os
import librosa
import json
import matplotlib.pyplot as plt
from cut_image import configs, get_audio_specs, get_img_scale, get_figname
from const_values import _COLORS
import numpy as np
import argparse

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_path', type=str)
    parser.add_argument('--res_path', type=str)
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--ext', type=str, default='.wav')

    parser.add_argument('--is_save_fig', type=bool, default=False)
    parser.add_argument('--prefix', action='store_true', default=False)
    args = parser.parse_args()
    return args

def cal_intersection(box1, box2):
    x1, y1, x2, y2 = box1
    m1, n1, m2, n2 = box2
    box_s1 = (x2-x1)*(y2-y1)
    box_s2 = (m2-m1)*(n2-n1)
    top = max(y1, n1)
    bottom = min(y2, n2)
    left = max(x1, m1)
    right = min(x2, m2)
    height = bottom-top
    width = right-left
    inner=0
    if height>0 and width>0:
        inner = height*width
    return inner, box_s1, box_s2


def process_box(bboxs, jsonpath):
    inner_thrd = 0.8
    is_valid = [True] * len(bboxs)
    for i in range(len(bboxs)):
        for j in range(len(bboxs)):
            if j==i or not is_valid[j]: continue
            inner, box_s1, box_s2 = cal_intersection(bboxs[i],bboxs[j])
            x1, y1, x2, y2 = bboxs[i]
            m1, n1, m2, n2 = bboxs[j]
            box = [min(x1, m1), min(y1, n1), max(x2, m2), max(y2, n2)]
            if box_s2==0:
                raise ValueError(jsonpath, i, j, bboxs[j])
            if inner/box_s2>=inner_thrd:
                bboxs[i]=box
                is_valid[j]=False

    boxs = []
    for i in range(len(is_valid)):
        if is_valid[i]:
            boxs.append(bboxs[i])
    bboxs = boxs

    for i in range(len(bboxs)):
        for j in range(len(bboxs)):
            if j==i: continue
            x1, y1, x2, y2 = bboxs[i]
            m1, n1, m2, n2 = bboxs[j]
            if x1<m1<x2 and m2>x2:
                origin = bboxs[j]
                bboxs[j] = [x2, n1, m2, n2]
                assert bboxs[j][2]>bboxs[j][0], ('error box!',i, j, bboxs[i], origin, bboxs[j])
    return bboxs

def save_img(spec, boxs, figpath):
    height, width = spec.shape
    plt.matshow(spec)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    fig = plt.gcf()
    fig.set_size_inches(width / 20, height / 40)
    plt.margins(0, 0)
    plt.axis('off')
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
    plt.close()
    spec_img = cv2.imread(figpath)
    os.remove(figpath)

    color = (_COLORS[0] * 255).astype(np.uint8).tolist()

    for x1, y1, x2, y2 in boxs:
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        spec_img = cv2.rectangle(spec_img, (x1, y1), (x2, y2), color, 2)

    # spec_img = np.array(spec_img)
    labefig_path = os.path.join(labelfig_dir, os.path.basename(figpath))
    label_img = cv2.imread(labefig_path)
    # label_img = np.array(label_img)

    # padded_img = np.ones(label_img.shape) * 114.0
    # spec_img_height, spec_img_width, _ = spec_img.shape
    # padded_img[: spec_img_height, : spec_img_width] = spec_img

    # img = np.concatenate([padded_img, label_img], axis=0)
    img = cv2.vconcat([spec_img, label_img])
    cv2.imwrite(figpath, img)

def choose_offset(audio, notes):
    win_len=1024
    blocks = int(np.ceil(audio.shape[0] / hop_length))
    pre_padding = win_len // 2
    sub_padding = blocks * hop_length - audio.shape[0]
    y_ = np.pad(audio, ((pre_padding, sub_padding)))
    y_=  100*y_/y_.max()
    energy=np.zeros([blocks,], np.float32)
    for i in range(blocks):
        block_energy=0
        for j in range(i*hop_length, i*hop_length+win_len):
            block_energy+=y_[j]**2
        energy[i]=block_energy

    energy=energy/energy.max()

    thrd=0.1
    for i in range(len(notes)-1):
        onset,offset=notes[i]
        onset_next, offset_next=notes[i+1]
        pos1=(onset+offset)/2
        pos2=(offset+onset_next)/2
        frame_idx1=int(np.floor(sr*pos1/hop_length))
        frame_idx2=int(np.ceil(sr*pos2/hop_length))
        idx=0
        for idx in range(frame_idx1, frame_idx2+1):
            if energy[idx]<=thrd:
                break
        if idx<frame_idx2+1:
            time_offset=idx*hop_length/sr
            notes[i][1]=time_offset
            # notes[i][1]=(notes[i][1]+time_offset)/2
    return notes

def generate_res(audiodir, save_dir, ext, res_fig_path, prefix, is_save_fig=None):

    sr = configs.sr
    mono = configs.mono
    hop_length = configs.hop_length

    filelist=dict()
    for imgname in os.listdir(res_fig_path):
        if imgname.endswith('.json'):

            *id, order = imgname.split('_')
            id = '_'.join(id)
            order = int(order.replace('.json', ''))
            if id not in filelist.keys():
                filelist[id]=[order]
            else:
                filelist[id].append(order)

    for id in filelist:
        filelist[id].sort()

    for id in filelist:
        if prefix:
            audiopath = os.path.join(audiodir, id, id+ext)
        else:
            audiopath = os.path.join(audiodir, id+ext)

        audio, sr_ = librosa.load(audiopath, sr=sr, mono=mono)
        duration = audio.shape[0]/sr_

        audio, specs = get_audio_specs(audio=audio, sr_=sr_)
        figname = get_figname(audiopath)
        # os.makedirs(save_dir, exist_ok=True)
        _, [img_height, img_width] = get_img_scale(specs, figname, save_dir, delete_fig=True, save_dir=save_dir)

        width_offset = 0
        total_boxs = []
        
        for order in filelist[id]:
            jsonpath=os.path.join(res_fig_path, id + '_%03d.json' % order)
            with open(jsonpath) as f:
                data=json.load(f)
            boxs = data['boxs'] if 'boxs' in data.keys() else data['bboxs']
            newboxs = []
            if 'bboxs' in data.keys():
                for x1, y1, x2, y2, score in boxs:
                    if score>=0.3:
                        newboxs.append([x1, y1, x2, y2])
            else:
                for x1, y1, x2, y2 in boxs:
                    newboxs.append([x1, y1, x2, y2])
            boxs = newboxs

            for x1, y1, x2, y2 in boxs:
                total_boxs.append([x1+width_offset, y1, x2+width_offset, y2])
            height, width = data['img_size']
            width_offset += width

        print(id)
        boxs = process_box(total_boxs, jsonpath)
        notes = []
        for x1, y1, x2, y2 in boxs:
            onset = duration*x1/img_width
            offset = duration*x2/img_width
            notes.append([onset, offset, img_height-y2])

        notes.sort(key=lambda x:x[0])
        newnotes = [notes[0]] if len(notes)>0 else []
        for i in range(1, len(notes)):
            if notes[i][0]<notes[i-1][1]:
                newnotes[i-1][1]=notes[i][0]
            newnotes.append(notes[i])

        # notes = choose_offset(audio, notes)

        savepath = os.path.join(save_dir, id+'.txt')
        with open(savepath, 'wt') as f:
            for onset, offset, candidate in notes:
                f.write('%.6f\t%.6f\t%.6f\n'%(onset, offset, candidate) )

        if is_save_fig:
            figpath = os.path.join(save_dir, id+'.png')
            save_img(specs, boxs, figpath)

if __name__=='__main__':

    args=parse_argument()

    is_save_fig = args.is_save_fig
    res_fig_path = args.bbox_path
    save_dir = args.res_path
    audiodir =  args.audio_path
    prefix = args.prefix
    os.makedirs(save_dir, exist_ok=True)

    generate_res(audiodir, save_dir, args.ext, res_fig_path, prefix, is_save_fig)