import os
import random

import librosa
from librosa.core.convert import cqt_frequencies, mel_frequencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
from tqdm import tqdm
import argparse
import json

common_config = {'sr': 44100, 'hop_length': 512, 'mono': True, 'fmin': 27.5}
split_audio_config = {'top_db': 20, 'frame_length': 1024, 'merge_thrd': 0.2}
cqt_config = {'bins_per_octave':24, 'n_bins':178}
mel_config = {'fmax': 8000, 'n_mels': 178, 'mel_n_fft': 2048}
img_config = {'max_ratio': 0.65, 'best_ratio':0.2, 'img_width_factor': 20, 'img_height_factor': 40}

configs = dict(**common_config, **cqt_config, **mel_config,
               **img_config, **split_audio_config)
configs = EasyDict(configs)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audiodir', type=str)
    parser.add_argument('--imgdir', type=str)

    parser.add_argument('--type', type=str, default='cqt')
    parser.add_argument('--minwidth', type=int, default=7)
    parser.add_argument('--plotbox', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--refaudiodir', type=str, default=None)
    args = parser.parse_args()
    return args

def get_spectrogram(audio, type='cqt'):
    if type=='mel':
        specs = librosa.feature.melspectrogram(audio, sr=configs.sr, n_fft=configs.mel_n_fft,
                hop_length=configs.hop_length, fmin=configs.fmin, fmax=configs.fmax, htk=True, n_mels=configs.n_mels)
        specs = librosa.power_to_db(specs)
    elif type=='cqt':
        specs = librosa.cqt(audio, sr=configs.sr, hop_length=configs.hop_length, fmin=configs.fmin, bins_per_octave=
                            configs.bins_per_octave, n_bins=configs.n_bins)
        specs = librosa.amplitude_to_db(np.abs(specs))
    else:
        raise ValueError("error spectrogram type!")
    return specs

def get_figname(audiopath):
    return os.path.splitext(os.path.split(audiopath)[1])[0]+'.png'

def get_img_scale(specs, figpath):
    height, width = specs.shape
    plt.matshow(specs)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()

    fig = plt.gcf()
    fig.set_size_inches(width / configs.img_width_factor, height / configs.img_height_factor)
    plt.margins(0, 0)
    plt.axis('off')
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0, dpi=100.0)
    plt.close()

    img = cv2.imread(figpath)
    img_height, img_width, channel = img.shape
    scale_height = img_height/height
    scale_width = img_width/width
    return [scale_height, scale_width], [img_height, img_width]

def get_split_line(audio):
    '''
    :param audio:
    :return new_splits: the idx is the raw point index
    '''
    splits = librosa.effects.split(audio, top_db=configs.top_db, frame_length=configs.frame_length,
                                   hop_length=configs.hop_length)
    # audio onset, end merge
    new_splits = [splits[0]]
    for idx in range(1, len(splits)):
        if (splits[idx][1] - splits[idx][0]) / configs.sr < configs.merge_thrd:
            new_splits[-1][1] = splits[idx][1]
        else:
            new_splits.append(splits[idx])
    return new_splits

def split_long_silence(start, end, new_splits, scale_width, scale_height):
    height =  np.round(scale_height*configs.n_bins)
    width = np.round((end-start)*scale_width)
    ratio = width/height-1
    if ratio<0:
        raise ValueError('cannot use this function')

    if ratio<=configs.best_ratio:
        new_splits.append(end)
        return new_splits

    best_hop_len = int(np.round(height*(1+configs.best_ratio) / scale_width))

    split = start + best_hop_len
    new_splits.append(split)
    width = np.round((end-split)*scale_width)
    ratio = width/height - 1

    while ratio>=configs.best_ratio:
        split = split+best_hop_len
        new_splits.append(split)
        width = np.round((end-split)*scale_width)
        ratio = width/height - 1
    if ratio>=0:
        new_splits.append(end)
    return new_splits

def get_s_ratio(new_splits, splits, seg_idx, height, scale_width):
    split_idx = max(new_splits[-1], int(np.round(splits[seg_idx + 1][0] - 2*configs.hop_length)))
    width = np.round((split_idx - new_splits[-1]) * scale_width)
    s_ratio = width / height - 1
    return split_idx, s_ratio

def get_next_ratio(new_splits, splits, seg_idx, height, scale_width):
    begin_split_next = max(new_splits[-1], np.round(splits[seg_idx + 1][0] - 2*configs.hop_length))
    end_split_next = np.round(splits[seg_idx + 1][1] + 2*configs.hop_length)
    width = np.round((end_split_next - begin_split_next) * scale_width)
    next_ratio = width / height - 1
    return next_ratio

def get_best_splits(splits, scale_height, scale_width, audio_length):
    '''
    :param splits: (start, end) tuples
    :return new_splits: tuple, each two successive complete a segment
    the long silence at the begin or the inner can't be ignored, but it can be
    ignored if it is at the end of the audio.
    '''
    new_splits=[0]

    # the half spec frame
    split_idx = int(max(0, np.round(splits[0][0]-2*configs.hop_length)))
    width = np.round(scale_width*split_idx)
    height = np.round(scale_height*configs.n_bins)
    ratio = width/height - 1
    if abs(ratio)<=configs.best_ratio:
        new_splits.append(split_idx)
    # the silence is to long
    elif ratio>=configs.best_ratio:
        new_splits=split_long_silence(0, split_idx, new_splits, scale_width, scale_height)
    # too short(ratio<-best_ratio don't don anything)

    seg_idx = 0
    while seg_idx<len(splits)-1:

        split_idx = np.round(min(splits[seg_idx][1] + 2*configs.hop_length,splits[seg_idx+1][0] - 2*configs.hop_length))
        width = (split_idx - new_splits[-1])*scale_width
        ratio = width / height - 1
        old_split_idx = split_idx

        if ratio>configs.max_ratio:
            new_splits.append(split_idx)
        elif -configs.best_ratio<=ratio<=0:
            split_idx, s_ratio=get_s_ratio(new_splits, splits, seg_idx, height, scale_width)
            if s_ratio>configs.max_ratio:
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            else:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > configs.max_ratio:
                    new_splits.append(split_idx)
                elif s_ratio>configs.best_ratio:
                    new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
                elif s_ratio>=-configs.best_ratio:
                    new_splits.append(split_idx)
            seg_idx+=1
        elif 0<ratio<=configs.best_ratio:

            split_idx, s_ratio=get_s_ratio(new_splits, splits, seg_idx, height, scale_width)

            if s_ratio>configs.max_ratio:
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            else:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > configs.max_ratio:
                    new_splits.append(split_idx)
                else:
                    new_splits.append(old_split_idx)
            seg_idx+=1
        elif configs.best_ratio<ratio<=configs.max_ratio:
            split_idx, s_ratio=get_s_ratio(new_splits, splits, seg_idx, height, scale_width)
            if s_ratio<=configs.max_ratio:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > configs.max_ratio:
                    new_splits.append(split_idx)
                else:
                    new_splits.append(old_split_idx)
            elif s_ratio<1+configs.max_ratio:
                length = np.round(height*(1+configs.max_ratio)/scale_width)
                split_idx = int(new_splits[-1]+length)
                new_splits.append(split_idx)
            else:
                new_splits.append(old_split_idx)
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            seg_idx+=1

        elif -configs.max_ratio<=ratio<-configs.best_ratio:

            split_idx, s_ratio = get_s_ratio(new_splits, splits, seg_idx, height, scale_width)

            if s_ratio>configs.max_ratio:
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            elif s_ratio>=0:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > configs.max_ratio:
                    new_splits.append(split_idx)
                elif s_ratio<=configs.best_ratio:
                    new_splits.append(split_idx)
                else:
                    length = np.round(height * (1 + configs.best_ratio) / scale_width)
                    split_idx = int(new_splits[-1] + length)
                    new_splits.append(split_idx)
            elif s_ratio>=-configs.best_ratio:
                new_splits.append(split_idx)
            seg_idx+=1
        else:
            seg_idx+=1

    split_idx = min(int(np.round(splits[seg_idx][1] + 2*configs.hop_length)), audio_length-1)
    width = np.round((split_idx - new_splits[-1]) * scale_width)
    ratio = width / height - 1
    if ratio>=-configs.best_ratio:
        new_splits.append(split_idx)
    else:
        split_idx = audio_length-1
        width = np.round((split_idx-new_splits[-1])*scale_width)
        s_ratio = width / height -1
        if s_ratio<=configs.best_ratio:
            new_splits.append(split_idx)
        else:
            length = np.round(height * (1 + configs.best_ratio) / scale_width)
            split_idx = int(new_splits[-1] + length)
            new_splits.append(split_idx)

    return new_splits

def show_cut_line(specs, splits, figname, savedir):
    plt.matshow(specs)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    for idx, split_idx in enumerate(splits):
        plt.vlines(split_idx, 0, configs.n_bins - 1, color='r', linestyle='dashed', linewidth=2)

    fig = plt.gcf()
    height, width = specs.shape
    fig.set_size_inches(width / configs.img_width_factor, height / configs.img_height_factor)
    plt.margins(0, 0)
    plt.axis('off')

    figpath = os.path.join(savedir, figname)
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
    plt.close()

def cut_image(specs, splits, scale_width, figpath, bboxs, plotbox):
    img=cv2.imread(figpath)
    os.remove(figpath)
    img_height, img_width, channels = img.shape
    height, width  = specs.shape
    if splits[-1]==width-1:
        splits[-1]=width
    imgs = []
    for i in range(len(splits)-1):
        start = int(np.round(splits[i]*scale_width))
        end = int(min(np.round(splits[i+1]*scale_width), img_width))
        imgs.append(img[:, start:end])
    savedir, figname=os.path.split(figpath)
    figname_prefix = os.path.splitext(figname)[0]
    assert len(imgs)==len(bboxs), 'the imgs length != bbox length'
    for idx,(img, boxs) in enumerate(zip(imgs, bboxs)):
        figname = figname_prefix+'_%03d.png'%idx
        img_height, img_width, channels = img.shape
        boxlabel={'flags':{}, 'imagePath': figname, 'imageHeight':img_height, 'imageWidth':img_width}
        shapes=[]
        for onset_loc, offset_loc, pitch_loc in boxs:
            onset_loc = min(max(1, onset_loc), img_width)
            offset_loc = min(max(1, offset_loc), img_width)
            pitch_loc = min(max(1, pitch_loc), img_height)
            note={'label': 'note', 'group_id': None, 'shape_type': 'rectangle', 'flags': {},
                  'points':[[onset_loc, 1], [offset_loc, pitch_loc]]}
            shapes.append(note)
            if plotbox:
                img = cv2.rectangle(img, (int(onset_loc), 1), (int(offset_loc), int(pitch_loc)), (0, 255, 0), 2)

        if len(shapes)==0:
            continue
        boxlabel['shapes']=shapes
        boxpath = os.path.join(savedir, figname_prefix+'_%03d.json'%idx)
        data = json.dumps(boxlabel, indent=4, separators=(',', ':'))
        with open(boxpath, 'w') as f:
            f.write(data)

        imgpath = os.path.join(savedir, figname_prefix+'_%03d.png'%idx)
        cv2.imwrite(imgpath, img)

def find_best_pitch_loc(midi, freqs):
    freq = librosa.midi_to_hz(midi)
    best_bin = 0
    for i in range(len(freqs)):
        if abs(freq-freqs[i])<abs(freq-freqs[best_bin]):
            best_bin=i
    return best_bin

def process_labels_splits(labels, splits, scale, type, min_width=5):
    [scale_height, scale_width] = scale
    if type=='mel':
        freqs=mel_frequencies(configs.n_mels + 2, fmin=configs.fmin, fmax=configs.fmax, htk=True)[1:-1]
    elif type=='cqt':
        freqs=cqt_frequencies(configs.n_bins, configs.fmin, configs.bins_per_octave)
    newlabels = []
    for onset, offset, pitch in labels:
        onset_loc = int(onset*configs.sr/configs.hop_length+0.5)
        offset_loc = int(offset*configs.sr/configs.hop_length+0.5)
        pitch_loc = find_best_pitch_loc(pitch, freqs)
        newlabels.append([onset_loc, offset_loc, pitch_loc])
    bboxs_len = len(splits)-1
    bboxs=[[] for i in range(bboxs_len)]
    if type=='cqt':
        n_bins = configs.n_bins
    elif type=='mel':
        n_bins = configs.n_mels
    else:
        raise ValueError("error spectrogram type")
    idx, label_idx=0, 0
    while idx<bboxs_len and label_idx<len(newlabels):
        begin, end=splits[idx], splits[idx+1]

        if newlabels[label_idx][0]>=begin and newlabels[label_idx][1]<=end:
            onset_loc, offset_loc, pitch_loc = newlabels[label_idx]
            onset_loc = (onset_loc - begin)*scale_width
            offset_loc = (offset_loc - begin)*scale_width
            pitch_loc = (n_bins - pitch_loc + 1.5) * scale_height
            bboxs[idx].append([onset_loc, offset_loc, pitch_loc])
            label_idx+=1

        elif newlabels[label_idx][0]>=end:
            idx+=1

        elif newlabels[label_idx][0]<end and newlabels[label_idx][1]>end:
            onset_loc, offset_loc, pitch_loc = newlabels[label_idx]
            onset_loc1 = (onset_loc - begin) * scale_width
            offset_loc1 = (end - begin) * scale_width
            onset_loc2 =  1
            offset_loc2 = (offset_loc - end) * scale_width
            pitch_loc = (configs.n_bins - pitch_loc + 1) * scale_height
            if offset_loc1-onset_loc1>min_width:
                bboxs[idx].append([onset_loc1, offset_loc1, pitch_loc])
            if idx+1<bboxs_len:
                if offset_loc2 - onset_loc2 > min_width:
                    bboxs[idx+1].append([onset_loc2, offset_loc2, pitch_loc])
            label_idx += 1
            idx += 1
    return bboxs


def load_audio_info(audiopath, sr, mono):
    audio, sr = librosa.load(audiopath, sr=sr, mono=mono)
    audiopath_prefix, filename = os.path.split(audiopath)
    filename = os.path.splitext(filename)[0]+'.txt'
    labels = np.loadtxt(os.path.join(audiopath_prefix, filename)).reshape([-1, 3])
    return audio, sr, labels


def get_cut_images_labels(audiopath, refpath, savedir, type, plotbox, minwidth=5):
    audio, sr, labels = load_audio_info(audiopath, configs.sr, configs.mono)
    if refpath == audiopath:
        ref_audio = audio
    else:
        ref_audio, _, _ = load_audio_info(refpath, configs.sr, configs.mono)
    specs = get_spectrogram(audio, type)

    height, width = specs.shape
    figname = get_figname(audiopath)
    # this scale_height and scale_width is for specs
    [scale_height, scale_width], _ = get_img_scale(specs, os.path.join(savedir, figname))

    splits = get_split_line(ref_audio)
    splits = get_best_splits(splits, scale_height, scale_width / configs.hop_length, audio.shape[0])
    # splits from audio points to frame unit
    for i in range(len(splits)):
        splits[i]=min(int(np.round((splits[i]/configs.hop_length))), width-1)

    bboxs=process_labels_splits(labels, splits, [scale_height, scale_width], type, minwidth)

    # scale_width is for specs
    cut_image(specs, splits, scale_width, os.path.join(savedir, figname), bboxs, plotbox)

def get_alread_process(imgdir):
    except_files= set()
    for file in os.listdir(imgdir):
        filename, ext = os.path.splitext(file)
        if ext!='.png': continue
        *id, number = filename.split('_')
        id = '_'.join(id)
        except_files.add(id)
    return list(except_files)


def main():
    args = parse_args()
    audiodir = args.audiodir
    refdir = args.refaudiodir if args.refaudiodir is not None else audiodir
    imgdir = args.imgdir
    type = args.type
    plotbox = args.plotbox
    minwidth = args.minwidth
    sample = args.sample
    except_files = get_alread_process(imgdir)
    filenames = [filename for filename in os.listdir(audiodir) if filename not in except_files]
    if sample:
        for i in range(20):
            random.shuffle(filenames)
        filenames=filenames[:500]
    os.makedirs(imgdir, exist_ok=True)
    for filename in tqdm(filenames):
        if filename in except_files:
            continue
        filepath = os.path.join(audiodir, filename, filename+'.wav')
        refpath = os.path.join(refdir, filename, filename+'.wav')
        if not os.path.exists(filepath):
            filepath = os.path.join(audiodir, filename, filename+'.mp3')
            refpath = os.path.join(refdir, filename, filename+'.mp3')
        print(filepath)
        get_cut_images_labels(filepath, refpath, imgdir, type, plotbox, minwidth)


if __name__=='__main__':
    main()