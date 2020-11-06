import os
import pickle
import threading

import numpy as np
from torch.utils import data
from tqdm import tqdm

from load_audio import load_wav
from load_video import load_mp4
from utils import colorize


class DemoDataset(data.Dataset):

    def __init__(self, video_path, resize, fps, sample_rate):

        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate

        self.data_path = 'media' 

        self.all_vids = [video_path]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_vids)

    def __getitem__(self, index):
        'Generates one sample of data'
        assert index < self.__len__()

        vid_path = self.all_vids[index]

        vid_name, vid_ext = os.path.splitext(vid_path)

        # -- load video
        vid_path_orig = os.path.join(self.data_path, vid_path)
        vid_path_25fps = os.path.join(self.data_path, vid_name + '_25fps.mp4')

        # -- reencode video to 25 fps
        command = (
            "ffmpeg -threads 1 -loglevel error -y -i {} -an -r 25 {}".format(
                vid_path_orig, vid_path_25fps))
        from subprocess import call
        cmd = command.split(' ')
        print('Resampling {} to 25 fps'.format(vid_path_orig))
        call(cmd)

        video = self.__load_video__(vid_path_25fps, resize=self.resize)

        aud_path = os.path.join(self.data_path, vid_name + '.wav')

        if not os.path.exists(aud_path):  # -- extract wav from mp4
            command = (
                ("ffmpeg -threads 1 -loglevel error -y -i {} "
                 "-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}")
                .format(vid_path_orig, aud_path))
            from subprocess import call
            cmd = command.split(' ')
            call(cmd)

        audio = load_wav(aud_path).astype('float32')

        fps = self.fps  # TODO: get as param?
        aud_fact = int(np.round(self.sample_rate / fps))
        audio, video = self.trunkate_audio_and_video(video, audio, aud_fact)
        assert aud_fact * video.shape[0] == audio.shape[0]

        audio = np.array(audio)

        video = video.transpose([3, 0, 1, 2])  # t c h w -> c t h w

        out_dict = {
            'video': video,
            'audio': audio,
            'sample': vid_path,
        }

        return out_dict

    def __load_video__(self, vid_path, resize=None):

        frames = load_mp4(vid_path)

        if resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in frames]
            ims = [
                torchvision.transforms.functional.resize(im,
                                                         resize,
                                                         interpolation=2)
                for im in ims
            ]
            frames = np.array([np.array(im) for im in ims])

        return frames.astype('float32')

    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):

        aud_in_frames = aud_feats.shape[0] // aud_fact

        # make audio exactly devisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]
        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            import ipdb
            ipdb.set_trace(context=20)

        return aud_feats, video
