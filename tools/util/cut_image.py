import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import cv2
from easydict import EasyDict
import argparse

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
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--ext', type=str, default='.flac')

    parser.add_argument('--prefix', action='store_true', default=False)
    
    # only for our paper experiment, not needed for inference
    parser.add_argument('--refaudiodir', type=str, default=None)
    parser.add_argument('--cutline', action='store_true', default=False)
    parser.add_argument('--type', type=str, default='cqt')

    args = parser.parse_args()
    return args

def plot_specs(specs, plt):
    plt.matshow(specs)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()

def set_img_size(width, height, plt):
    fig = plt.gcf()
    fig.set_size_inches(width / 20, height / 40)
    plt.margins(0, 0)
    plt.axis('off')

def get_spectrogram(audiopath=None, audio=None, type='cqt'):
    if audio==None:
        audio, _ = librosa.load(audiopath, sr=configs.sr, mono=True)
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
    return audio, specs

def get_audio_specs(audiopath=None, audio=None, sr_=None):
    if audiopath:
        assert audio is None and sr_ is None, 'audiopath should be None'
        audio, sr_ = librosa.load(audiopath, sr=configs.sr, mono=configs.mono)
    specs = librosa.cqt(audio, sr=configs.sr, hop_length=configs.hop_length, fmin=configs.fmin,
                        n_bins=configs.n_bins, bins_per_octave=configs.bins_per_octave)
    specs = np.abs(specs)
    specs = librosa.amplitude_to_db(specs)

    return audio, specs

def get_figname(audiopath, prefix=''):
    figname = os.path.splitext(os.path.basename(audiopath))[0]
    return prefix+figname+'.png'

def get_img_scale(specs, figname, savedir, delete_fig=False, save_dir=None):
    height, width = specs.shape
    plt.matshow(specs)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()

    # set_img_size(height, width, plt)
    fig = plt.gcf()
    # raise ValueError(fig.get_dpi())
    fig.set_size_inches(width / 20, height / 40)
    plt.margins(0, 0)
    plt.axis('off')

    figpath = os.path.join(savedir, figname)
    plt.savefig(figpath, bbox_inches='tight', pad_inches=0, dpi=100.0)
    plt.close()

    img = cv2.imread(figpath)
    img_height, img_width, channel = img.shape
    scale_height = img_height/height
    scale_width = img_width/width
    if delete_fig:
        os.remove(figpath)
    return [scale_height, scale_width], [img_height, img_width]

def get_split_line(audio):
    '''
    :param audio:
    :return new_splits: the idx is the raw point index
    '''
    splits = librosa.effects.split(audio, top_db=20, frame_length=1024, hop_length=512)
    # audio onset, end merge
    merge_thrd = 0.2
    new_splits = [splits[0]]
    for idx in range(1, len(splits)):
        if (splits[idx][1] - splits[idx][0]) / configs.sr < merge_thrd:
            new_splits[-1][1] = splits[idx][1]
        else:
            new_splits.append(splits[idx])
    return new_splits

def split_long_silence(start, end, new_splits, scale_width, scale_height):
    height =  np.round(scale_height*configs.n_bins)
    width = np.round((end-start)*scale_width)
    ratio = width/height-1
    best_ratio = configs.best_ratio
    if ratio<0:
        raise ValueError('cannot use this function')

    if ratio<=best_ratio:
        new_splits.append(end)
        return new_splits

    best_hop_len = int(np.round(height*(1+best_ratio) / scale_width))
    split = start + best_hop_len
    new_splits.append(split)
    width = np.round((end-split)*scale_width)
    ratio = width/height - 1

    while ratio>=best_ratio:
        split = split+best_hop_len
        new_splits.append(split)
        width = np.round((end-split)*scale_width)
        ratio = width/height - 1
    if ratio>=0:
        new_splits.append(end)
    return new_splits

def get_s_ratio(new_splits, splits, seg_idx, height, scale_width):
    hop_length = configs.hop_length
    split_idx = max(new_splits[-1], int(np.round(splits[seg_idx + 1][0] - 2*hop_length)))
    width = np.round((split_idx - new_splits[-1]) * scale_width)
    s_ratio = width / height - 1
    return split_idx, s_ratio

def get_next_ratio(new_splits, splits, seg_idx, height, scale_width):
    hop_length = configs.hop_length
    begin_split_next = max(new_splits[-1], np.round(splits[seg_idx + 1][0] - 2*hop_length))
    end_split_next = np.round(splits[seg_idx + 1][1] + 2*hop_length)
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
    hop_length = configs.hop_length
    best_ratio = configs.best_ratio
    max_ratio = configs.max_ratio
    new_splits=[0]
    # the half spec frame
    split_idx = int(max(0, np.round(splits[0][0]-2*hop_length)))
    width = np.round(scale_width*split_idx)
    height = np.round(scale_height*configs.n_bins)
    ratio = width/height - 1
    if abs(ratio)<=best_ratio:
        new_splits.append(split_idx)
    # the silence is to long
    elif ratio>=best_ratio:
        new_splits=split_long_silence(0, split_idx, new_splits, scale_width, scale_height)
    # too short(ratio<-best_ratio don't don anything)

    seg_idx = 0
    while seg_idx<len(splits)-1:

        split_idx = np.round(min(splits[seg_idx][1] + 2*hop_length,splits[seg_idx+1][0] - 2*hop_length))
        width = (split_idx - new_splits[-1])*scale_width
        ratio = width / height - 1
        old_split_idx = split_idx

        if ratio>max_ratio:
            new_splits.append(split_idx)
        elif -best_ratio<=ratio<=0:
            split_idx, s_ratio=get_s_ratio(new_splits, splits, seg_idx, height, scale_width)
            if s_ratio>max_ratio:
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            else:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > max_ratio:
                    new_splits.append(split_idx)
                elif s_ratio>best_ratio:
                    new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
                elif s_ratio>=-best_ratio:
                    new_splits.append(split_idx)
            seg_idx+=1
        elif 0<ratio<=best_ratio:

            split_idx, s_ratio=get_s_ratio(new_splits, splits, seg_idx, height, scale_width)

            if s_ratio>max_ratio:
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            else:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > max_ratio:
                    new_splits.append(split_idx)
                else:
                    new_splits.append(old_split_idx)
            seg_idx+=1
        elif best_ratio<ratio<=max_ratio:
            split_idx, s_ratio=get_s_ratio(new_splits, splits, seg_idx, height, scale_width)
            if s_ratio<=max_ratio:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > max_ratio:
                    new_splits.append(split_idx)
                else:
                    new_splits.append(old_split_idx)
            elif s_ratio<1+max_ratio:
                length = np.round(height*(1+max_ratio)/scale_width)
                split_idx = int(new_splits[-1]+length)
                new_splits.append(split_idx)
            else:
                new_splits.append(old_split_idx)
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            seg_idx+=1

        elif -max_ratio<=ratio<-best_ratio:

            split_idx, s_ratio = get_s_ratio(new_splits, splits, seg_idx, height, scale_width)

            if s_ratio>max_ratio:
                new_splits = split_long_silence(new_splits[-1], split_idx, new_splits, scale_width, scale_height)
            elif s_ratio>=0:
                next_ratio = get_next_ratio(new_splits, splits, seg_idx, height, scale_width)
                if next_ratio > max_ratio:
                    new_splits.append(split_idx)
                elif s_ratio<=best_ratio:
                    new_splits.append(split_idx)
                else:
                    length = np.round(height * (1 + best_ratio) / scale_width)
                    split_idx = int(new_splits[-1] + length)
                    new_splits.append(split_idx)
            elif s_ratio>=-best_ratio:
                new_splits.append(split_idx)
            seg_idx+=1
        else:
            seg_idx+=1

    # process the last note
    split_idx = min(int(np.round(splits[seg_idx][1] + 2*hop_length)), audio_length-1)
    width = np.round((split_idx - new_splits[-1]) * scale_width)
    ratio = width / height - 1
    if ratio>=-best_ratio:
        new_splits.append(split_idx)
    else:
        split_idx = audio_length-1
        width = np.round((split_idx-new_splits[-1])*scale_width)
        s_ratio = width / height -1
        if s_ratio<=best_ratio:
            new_splits.append(split_idx)
        else:
            length = np.round(height * (1 + best_ratio) / scale_width)
            split_idx = int(new_splits[-1] + length)
            new_splits.append(split_idx)

    return new_splits

def show_cut_line(specs, splits, figpath):
    plt.matshow(specs)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    for idx, split_idx in enumerate(splits):
        plt.vlines(split_idx, 0, configs.n_bins - 1, color='r', linestyle='dashed', linewidth=2)

    fig = plt.gcf()
    height, width = specs.shape
    fig.set_size_inches(width / 20, height / 40)
    plt.margins(0, 0)
    plt.axis('off')

    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_label_line(specs, labelpath, figpath):
    plt.matshow(specs)
    ax = plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    labels = np.loadtxt(labelpath).reshape([-1, 3])
    for onset, offset, pitch in labels:
        onset_idx = onset*configs.sr/configs.hop_length
        offset_idx = offset*configs.sr/configs.hop_length
        plt.vlines(onset_idx, 0, configs.n_bins - 1, color='r', linestyle='dashed', linewidth=2)
        plt.vlines(offset_idx, 0, configs.n_bins - 1, color='g', linestyle='dashed', linewidth=2)
    fig = plt.gcf()
    height, width = specs.shape
    fig.set_size_inches(width / 20, height / 40)
    plt.margins(0, 0)
    plt.axis('off')

    plt.savefig(figpath, bbox_inches='tight', pad_inches=0)
    plt.close()

def show_comparable_fig(cutlinepath, labelfigpath, concat_fig_path):
    cutline_img = cv2.imread(cutlinepath)
    label_img = cv2.imread(labelfigpath)
    img = cv2.vconcat([cutline_img, label_img])
    cv2.imwrite(concat_fig_path, img)
    os.remove(cutlinepath)
    os.remove(labelfigpath)

def cut_image(specs, splits, scale_width, savedir, figname):
    figpath = os.path.join(savedir, figname)
    img=cv2.imread(figpath)
    img_height, img_width, channels = img.shape
    height, width  = specs.shape
    if splits[-1]==width-1:
        splits[-1]=width
    imgs = []
    for i in range(len(splits)-1):
        start = int(np.round(splits[i]*scale_width))
        end = int(min(np.round(splits[i+1]*scale_width), img_width))
        imgs.append(img[:, start:end])
    figname = figname.replace('.png', '')
    for idx,img in enumerate(imgs):
        imgpath = os.path.join(savedir, figname+'_%03d.png'%idx)
        cv2.imwrite(imgpath, img)
    os.remove(figpath)


def plot_cut_image(audiopath, savedir, cutline, type, refaudiodir=None, ext=None):

    hop_length = configs.hop_length

    if refaudiodir is not None:
        basename = os.path.basename(audiopath)
        refaudiopath = os.path.join(refaudiodir,basename.replace(ext, ''), basename)
        refaudio, _ = get_spectrogram(audiopath=refaudiopath, type=type)
    audio, specs = get_spectrogram(audiopath=audiopath, type=type)
    height, width = specs.shape
    if refaudiodir is not None:
        splits = get_split_line(refaudio)
    else:
        splits = get_split_line(audio)

    figname = get_figname(audiopath)
    # this scale_height and scale_width is for specs
    [scale_height, scale_width], _ = get_img_scale(specs, figname, savedir, delete_fig=cutline)
    # this scale_width is for raw audio points
    scale_width = scale_width / hop_length

    splits = get_best_splits(splits, scale_height, scale_width, audio.shape[0])
    # splits from audio points to frame unit
    for i in range(len(splits)):
        splits[i]=min(int(np.round((splits[i]/hop_length))), width-1)

    if cutline:
        labelpath = os.path.splitext(audiopath)[0]+'.txt'
        cutline_dir = os.path.join(savedir, 'cutline')
        os.makedirs(cutline_dir,exist_ok=True)
        label_dir = os.path.join(savedir, 'label')
        os.makedirs(label_dir, exist_ok=True)
        cutline_fig_path = os.path.join(cutline_dir, figname)
        label_fig_path = os.path.join(label_dir, figname)
        show_cut_line(specs, splits, cutline_fig_path)
        show_label_line(specs, labelpath, label_fig_path)
        # concat_fig_path = os.path.join(savedir, figname)
        # show_comparable_fig(cutline_fig_path, label_fig_path, concat_fig_path)
    else:
        # scale_width is for specs
        scale_width = scale_width*hop_length
        cut_image(specs, splits, scale_width, savedir, figname)


def generate_slices(audiodir, savedir, prefix, ext, atype='cqt', refaudiodir=None, cutline=False):
    
    filename_sets = set([filename.split('.')[0] for filename in os.listdir(audiodir)])
    for filename in filename_sets:
        if prefix:
            audiopath = os.path.join(audiodir, filename, filename+ext)
        else:
            audiopath = os.path.join(audiodir, filename+ext)
        print(audiopath)
        plot_cut_image(audiopath, savedir, cutline, atype, refaudiodir, ext)


if __name__=='__main__':
    args=parse_args()
    audiodir = args.audiodir
    refaudiodir = args.refaudiodir
    savedir = args.savedir
    prefix = args.prefix
    cutline = args.cutline
    os.makedirs(savedir, exist_ok=True)

    generate_slices(audiodir, savedir, prefix, args.ext, args.type, refaudiodir, cutline)