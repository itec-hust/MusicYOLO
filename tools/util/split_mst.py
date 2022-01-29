import os
import argparse
import numpy as np
import librosa
import soundfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mst_path', type=str)
    parser.add_argument('--dest_dir', type=str)
    return parser.parse_args()

def main(args):
    vocal_path = args.mst_path
    dest_dir = args.dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(vocal_path):
        labelpath = os.path.join(vocal_path, filename, filename+'.txt')
        label = np.loadtxt(labelpath)
        splits=[0]
        offset=0
        for i in range(label.shape[0]-1):
            if label[i][1]-offset>25 and label[i+1][0]-label[i][1]>1:
                splits.append((label[i][1]+label[i+1][0])/2)
                offset=splits[-1]
            elif label[i][1]-offset>35 and label[i+1][0]-label[i][1]>0.5:
                splits.append((label[i][1]+label[i+1][0])/2)
                offset=splits[-1]
            elif label[i][1] - offset > 45:
                splits.append((label[i][1] + label[i + 1][0]) / 2)
                offset = splits[-1]

        if label[-1][1]+1-offset<10:
            splits.pop()
        audiopath = os.path.join(vocal_path, filename, 'Mixture.mp3')
        audio, sr = librosa.load(audiopath, sr=44100, mono=True)
        splits.append(min(label[-1][1]+1, audio.shape[0]/sr))
        # raise ValueError(splits)
        split_labels = [[] for i in range(len(splits)-1)]
        label_idx = 0
        for i in range(len(split_labels)):
            begin, end = splits[i:i+2]
            while label_idx<label.shape[0] and label[label_idx][0]>=begin and label[label_idx][1]<=end:
                onset, offset, pitch = label[label_idx]
                onset, offset = onset-begin, offset-begin
                split_labels[i].append([onset, offset, pitch])
                label_idx+=1

        for i in range(len(split_labels)):
            dirpath = os.path.join(dest_dir, filename + '_%02d' % (i))
            os.makedirs(dirpath, exist_ok=True)
            begin, end = splits[i:i + 2]
            begin_idx, end_idx = int(begin*sr), int(end*sr)+1
            split_audio = audio[begin_idx: end_idx]
            file = os.path.join(dirpath, filename + '_%02d.wav' % (i))
            soundfile.write(file, split_audio, sr)
            split_label_path = os.path.join(dirpath, filename + '_%02d.txt' % (i))
            with open(split_label_path, 'w') as f:
                for onset, offset, pitch in split_labels[i]:
                    f.write('%.6f\t%.6f\t%.1f\n'%(onset, offset, pitch))

if __name__=='__main__':
    main(parse_args())