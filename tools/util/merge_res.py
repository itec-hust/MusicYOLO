import os
import librosa
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str)
    parser.add_argument('--origin_dir', type=str)
    parser.add_argument('--final_dir', type=str)
    return parser.parse_args()

def save_file(notes, filepath):
    with open(filepath, 'wt') as f:
        for onset, offset, position in notes:
            if position < 0:
                print(filepath)
            f.write('%.6f\t%.6f\t%.6f\n'%(onset, offset, position))

def process_onefile(origin_path, final_path, filename, maxn, audiodir):
    total_res = []
    offset = 0
    for n in range(maxn+1):
        name = filename+'_%02d'%(n)
        part_res = np.loadtxt(os.path.join(origin_path, name+'.txt'))
        if part_res.shape[0]==0: continue
        part_res = part_res.reshape([-1, 3])
        part_res[:, :2]+=offset
        total_res.append(part_res)
        audiopath = os.path.join(audiodir, name, name+'.wav')
        audio, sr = librosa.load(audiopath)
        offset+=audio.shape[0]/sr
    total_res = np.concatenate(total_res, axis=0)
    save_file(total_res, os.path.join(final_path, filename + '.txt'))


if __name__=='__main__':
    args = parse_args()
    audio_dir = args.audio_dir
    origin_res_path = args.origin_dir
    final_res_path = args.final_dir
    os.makedirs(final_res_path, exist_ok=True)

    filenames = dict()
    for name in os.listdir(origin_res_path):
        name = os.path.splitext(name)[0]
        if name.split('_')[0] not in filenames.keys():
            filenames[name.split('_')[0]]=int(name.split('_')[1])
        else:
            filenames[name.split('_')[0]] = max(int(name.split('_')[1]), filenames[name.split('_')[0]])

    for name in filenames.keys():
        print(name)
        process_onefile(origin_res_path, final_res_path, name, filenames[name], audio_dir)