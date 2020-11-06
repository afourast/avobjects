# -*- coding: utf-8 -*-

import os
import subprocess
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from load_audio import reconstruct_wav_from_mag_phase, wav2filterbanks
from utils import (colorize, detect_peaks, extract_attended_features,
                   extract_face_crops, full_to_map, load_model_params,
                   logsoftmax_2d, my_unfold, run_func_in_parts,
                   calc_flow_on_vid_wrapper)
from viz_utils import VideoSaver, viz_avobjects, viz_source_separation


class DemoEvalTrainer():

    def __init__(self, model, opts):
        self.model = model
        self.opts = opts

        self.device = torch.device('cuda:0')
        self.model.to(self.device)

        # make log directories
        self.step = 0
        opts.output_dir = os.path.join(opts.output_dir)

        if os.path.exists(opts.output_dir):
            # Clean up old logs
            command = 'rm %s/* -rf' % (opts.output_dir)
            print(command)
            subprocess.call(command, shell=True, stdout=None)

        self.checkpoints_path = opts.output_dir + "/checkpoints"

        # set up tb saver
        os.makedirs(opts.output_dir, exist_ok=True)
        self.video_saver = VideoSaver(opts.output_dir)

        from warp_video import Warper
        self.warper = Warper(device=self.device)

    def eval(self, dataloader):

        bs = self.opts.batch_size

        for batch_sample in dataloader:

            self.model.zero_grad()

            video = batch_sample['video']  # b x T_v x H x W x 3
            audio = batch_sample['audio']  # b x T_a

            # extract log-mel filterbanks  -  # b*41 x T_a x C
            mel, _, _, _ = wav2filterbanks(audio.to(self.device))

            # -- 1. Forward audio and video through the model to get embeddings

            # pad so that we get an attention map over the whole video
            pad_len = int(self.model.start_offset)
            video = torch.nn.ConstantPad3d([pad_len, pad_len, pad_len, pad_len],
                                           0)(video)

            # For handling large videos: split the forward pass into parts
            max_len = 50
            vid_parts = my_unfold(video,
                                  size=max_len,
                                  step=max_len - 4,
                                  dimension=2,
                                  chunk_at_least=5)  # bs x n_chunks x 5 x tchw

            vid_embs = []
            vid_feats = []

            for vid_ch in tqdm(vid_parts, desc='Forward vid pass with parts'):
                emb, feat = self.model.forward_vid(vid_ch.to(self.device),
                                                   return_feats=True)
                vid_embs.append(emb.detach().cpu())
                vid_feats.append(feat.detach().cpu())

            vid_emb = torch.cat(vid_embs, 2)
            vid_feat = torch.cat(vid_feats, 2)

            aud_emb = self.model.forward_aud(
                mel.permute([0, 2, 1])[:, None].to(self.device))
            # this is for simulating a the dimension of posneg samples
            aud_emb = aud_emb[:, None]

            # L2 normalize embeddings
            vid_emb = torch.nn.functional.normalize(vid_emb, p=2, dim=1)
            aud_emb = torch.nn.functional.normalize(aud_emb, p=2, dim=1)

            # -- 2. Synchronize the inputs online by calculating optimal offset
            #       then shifting modalities accordingly
            video, audio, vid_emb, aud_emb, vid_feat = self.online_sync(
                vid_emb, aud_emb, video, audio, vid_feat)

            # recalculate spectrograms on shifted (syncrhonized) audio version
            mel, mag, phase, _ = wav2filterbanks(audio.to(self.device))
            mel = mel.permute([0, 2, 1])[:, None]  # b*41 x 1 x F x T_a
            mag /= 32768.

            # -- 3. Get the avobject trajectories by aggregating over flow + NMS
            att_map = self.calc_att_map(vid_emb, aud_emb)
            att_map = logsoftmax_2d(att_map)

            h_map, w_map = att_map.shape[-2:]
            avobject_traj, agg_att_map, att_map_smooth = \
                    self.avobject_trajectories_from_flow(att_map, video)

            # need for reconstruction
            audio = audio.reshape((bs, -1) + audio.shape[1:])[:, -1]

            # -- 4. Attend on the features at the peaks locations
            # convert pixel peaks to att map coords
            h_full, w_full = video.shape[-2:]
            avobject_traj_map = np.array([
                full_to_map(traj, h_full, w_full, h_map, w_map,
                             self.model.start_offset) for traj in avobject_traj
            ])
            # b x C x T -- x100 is for getting the feature at max location
            attended_v_feat = extract_attended_features(att_map[:, 0:1] * 100,
                                                        vid_feat,
                                                        avobject_traj_map)

            # -- 5. Use the attended visual features for source separation
            attended_v_feat = attended_v_feat.reshape(
                (-1,) + attended_v_feat.shape[2:])  # fold into batch dim

            # enhance magnitudes
            # remove the +-2 frames from edges we lose from convolution
            t_slice = slice(4 * 2, -4 * 2)  # (4 is the audio multiplier)
            mag, phase = mag[..., t_slice], phase[..., t_slice]
            t_slice = slice(mag.shape[-1])
            mel_tiled = mel
            # tile the mix audio embeddings to apply to all proposals
            mel_tiled = mel_tiled.repeat([1, self.opts.n_peaks, 1, 1])
            mel_tiled = mel_tiled.reshape((-1,) + mel_tiled.shape[2:])
            mel_tiled = mel_tiled.permute([0, 2, 1])  # b x t x c --> b x c x t
            sep_map = self.model.sepnet(
                [attended_v_feat.to(self.device),
                 mel_tiled.to(self.device)])  # ( bs * n_spks) x F x T

            # enhance phase
            mag_tiled = mag[:, None, ..., t_slice]
            enh_mag = sep_map.reshape(
                (bs, self.opts.n_peaks) +
                mag_tiled.shape[2:]) * mag_tiled  # bs x n_spks x F x T
            enh_mag_folded = enh_mag.reshape(
                (-1,) + enh_mag.shape[2:])  # fold into batch dim
            phase_folded = phase[:, None].repeat([1, self.opts.n_peaks, 1, 1])
            phase_folded = phase_folded.reshape(
                (-1,) + phase_folded.shape[2:])  # fold into batch dim
            assert enh_mag_folded.shape == phase_folded.shape
            phasenet_inp = [
                enh_mag_folded.to(self.device),
                phase_folded.to(self.device)
            ]
            enh_phase_as_2d, enh_phase_angle = self.model.phasenet(
                phasenet_inp)  # ( bs * n_spks) x F x T
            enh_phase_as_2d = enh_phase_as_2d.reshape(enh_mag.shape + (2,))
            enh_phase = enh_phase_angle.reshape(enh_mag.shape)
            enh_phase = enh_phase[..., t_slice]

            # -- 6. visualization + reconstruction

            b_id = 0 # first batch sample  

            # visualize the attention map + avobject trajectories 
            viz_avobjects(
                video[b_id],
                audio[b_id],
                att_map=att_map_smooth[b_id],
                avobject_traj=avobject_traj[b_id],
                model_start_offset=int(self.model.start_offset),
                video_saver=self.video_saver,
                const_box_size=self.opts.const_box_size,
                step=self.step,
            )

            enhanced_audio = reconstruct_wav_from_mag_phase(enh_mag[b_id],
                                                            enh_phase[b_id])

            viz_source_separation(
                video=video[b_id],
                enh_audio=enhanced_audio,
                avobject_traj=avobject_traj[b_id],
                model_start_offset=int(self.model.start_offset),
                video_saver=self.video_saver,
                const_box_size=self.opts.const_box_size,
                step=self.step,
                )

            import aolib_p3.imtable as imtable
            avobj_vid_name = os.path.join(
                self.video_saver.savedir,
                'avobject_viz/{}.mp4'.format(self.step))
            enh_vid_name = os.path.join(self.video_saver.savedir,
                                          'sep_vid/{}/enh_'.format(self.step))
            imtable.show(
                [ [ 'AV attention + tracked obects', imtable.Video(video_fname=avobj_vid_name) ], ] 
              + [ ['Speaker {}'.format(ii),
                    imtable.Video(video_fname=enh_vid_name + '{}.mp4'.format(ii)) ]
                        for ii in range(self.opts.n_peaks)],
                output_path=os.path.join(self.video_saver.savedir),
                        )

            self.step += 1

        return "", 0

    # ============ att maps & av scores ============

    def calc_av_scores(self, vid_emb, aud_emb):
        """
        :return: aggregated scores over T, h, w
        """

        scores = self.calc_att_map(vid_emb, aud_emb)
        att_map = logsoftmax_2d(scores)

        scores = torch.nn.MaxPool3d(kernel_size=(1, scores.shape[-2],
                                                 scores.shape[-1]))(scores)
        scores = scores.squeeze(-1).squeeze(-1).mean(2)

        return scores, att_map

    def calc_att_map(self, vid_emb, aud_emb):
        """
        :param vid_emb: b x C x T x h x w
        :param aud_emb: b x num_neg x C x T
        """

        # b x C x   1   x T x h x w
        vid_emb = vid_emb[:, :, None]
        # b x C x n_neg x T x 1 x 1
        aud_emb = aud_emb.transpose(1, 2)[..., None, None]

        scores = run_func_in_parts(lambda x, y: (x * y).sum(1),
                                   vid_emb,
                                   aud_emb,
                                   part_len=10,
                                   dim=3,
                                   device=self.device)

        # this is the learned logits scaling to move the input to the softmax
        # out of the [-1,1] (e.g what SimCLR sets to 0.07)
        scores = self.model.logits_scale(scores[..., None]).squeeze(-1)

        return scores

    # ============ online AV sync  ============

    def online_sync(self, vid_emb, aud_emb, video, audio, vid_feat):

        sync_offset = self.calc_optimal_av_offset(vid_emb, aud_emb)

        vid_emb_sync, aud_emb_sync = self.sync_av_with_offset(vid_emb,
                                                              aud_emb,
                                                              sync_offset,
                                                              dim_v=2,
                                                              dim_a=3)

        vid_feat_sync, _ = self.sync_av_with_offset(vid_feat,
                                                    aud_emb,
                                                    sync_offset,
                                                    dim_v=2,
                                                    dim_a=3)

        # e.g. a_mult = 640 for 16khz and 25 fps
        a_mult = int(self.opts.sample_rate / self.opts.fps)
        video, audio = self.sync_av_with_offset(video,
                                                audio,
                                                sync_offset,
                                                dim_v=2,
                                                dim_a=1,
                                                a_mult=a_mult)
        return video, audio, vid_emb_sync, aud_emb_sync, vid_feat_sync

    def create_online_sync_negatives(self, vid_emb, aud_emb):
        assert self.opts.n_negative_samples % 2 == 0
        ww = self.opts.n_negative_samples // 2

        fr_trunc, to_trunc = ww, aud_emb.shape[-1] - ww
        vid_emb_pos = vid_emb[:, :, fr_trunc:to_trunc]
        slice_size = to_trunc - fr_trunc

        aud_emb_posneg = aud_emb.squeeze(1).unfold(-1, slice_size, 1)
        aud_emb_posneg = aud_emb_posneg.permute([0, 2, 1, 3])

        # this is the index of the positive samples within the posneg bundle
        pos_idx = self.opts.n_negative_samples // 2
        aud_emb_pos = aud_emb[:, 0, :, fr_trunc:to_trunc]

        # make sure that we got the indices correctly
        assert torch.all(aud_emb_posneg[:, pos_idx] == aud_emb_pos)

        return vid_emb_pos, aud_emb_posneg, pos_idx

    def calc_optimal_av_offset(self, vid_emb, aud_emb):
        vid_emb, aud_emb, pos_idx = self.create_online_sync_negatives(
            vid_emb, aud_emb)
        scores, _ = self.calc_av_scores(vid_emb, aud_emb)
        offset = scores.argmax() - pos_idx
        return offset.item()

    def sync_av_with_offset(self,
                            vid_emb,
                            aud_emb,
                            offset,
                            dim_v,
                            dim_a,
                            a_mult=1):
        if vid_emb is not None:
            init_dim = vid_emb.shape[dim_v]
        else:
            init_dim = aud_emb.shape[dim_a] // a_mult

        length = init_dim - int(np.abs(offset))

        if vid_emb is not None:
            if offset < 0:
                vid_emb = vid_emb.narrow(dim_v, -offset, length)
            else:
                vid_emb = vid_emb.narrow(dim_v, 0, length)

            assert vid_emb.shape[dim_v] == init_dim - np.abs(offset)

        if aud_emb is not None:
            if offset < 0:
                aud_emb = aud_emb.narrow(dim_a, 0, length * a_mult)
            else:
                aud_emb = aud_emb.narrow(dim_a, offset * a_mult,
                                         length * a_mult)
            assert aud_emb.shape[dim_a] // a_mult == init_dim - np.abs(offset)

        return vid_emb, aud_emb

    # ============ avobjects tracking ============

    def avobject_trajectories_from_flow(self, att_map, video):
        """
        Use the av attention map to aggregate vid
        features corresponding to peaks
        """

        # - flow not provided, we need to calculate it on the fly
        flows = []
        for b_id in range(len(video)):
            flow_inp = video.permute([0, 2, 3, 4, 1])

            # NOTE: This is a workaround to the GPL license of PWCnet wrapper
            # We call it as an executable: The network is therefore initialized
            # again in each call and the input images and output flow are passed
            # by copying into shared memory (/dev/shm)
            # This is very suboptimal - not to be used for training
            flow = calc_flow_on_vid_wrapper(flow_inp[b_id].detach().cpu().numpy(), gpu_id=self.opts.gpu_id)
            flow = torch.from_numpy(flow).permute([0, 2, 3, 1])  # tchw -> thwc
            flows.append(flow)
        flow = np.stack(flows)
        flow = torch.from_numpy(flow)
        flow = torch.nn.ConstantPad3d([0, 0, 0, 0, 0, 0, 0, 1], 0)(flow)

        def smoothen_and_pad_att_map(av_att, pad_resid=2):
            map_t_avg = torch.nn.AvgPool3d(kernel_size=(5, 1, 3),
                                            stride=1,
                                            padding=(2, 0, 1),
                                            count_include_pad=False)(av_att)
            map_t_avg = torch.nn.ReplicationPad3d(
                (0, 0, 0, 0, pad_resid, pad_resid))(map_t_avg[None]).squeeze()
            return map_t_avg

        att_map_smooth = smoothen_and_pad_att_map(att_map[:, 0])

        if self.opts.batch_size == 1:
            att_map_smooth = att_map_smooth[None]

        map_for_peaks = att_map_smooth

        # -- 1. aggregate (sum) the attention map values over every pixel trajectory
        flow_inp_device = 'cpu'  # for handling large inputs
        agg_att_map, pixel_trajectories, _ = \
            self.warper.integrate_att_map_over_flow_trajectories(flow.to(flow_inp_device),
                                                                    map_for_peaks,
                                                                    int(self.model.start_offset),
                                                                    )
        agg_att_map = agg_att_map.detach().cpu().numpy()  # bs x T x h x w

        avobject_trajectories = []

        bs = len(pixel_trajectories)
        for b_id in range(bs):

            # -- 2. detect peaks on the aggregated attention map with NMS
            peaks, peak_coords = detect_peaks(
                agg_att_map[b_id], overlap_thresh=self.opts.nms_thresh)
            peak_sort_ids = np.argsort(-agg_att_map[b_id][peak_coords.T[0],
                                                          peak_coords.T[1]])

            selected_peaks = peak_coords[peak_sort_ids[:self.opts.n_peaks]]
            top_traj_map = np.zeros_like(agg_att_map[b_id])
            for peak_y, peak_x in selected_peaks:
                pw = 5 // 2
                top_traj_map[max(0, peak_y - pw):peak_y + pw + 1,
                             max(0, peak_x - pw):peak_x + pw + 1] = 1

            # -- 3. Select only the trajectories of the peaks
            peak_pixel_traj = torch.stack([
                pixel_trajectories[b_id][..., peak[0], peak[1]]
                for peak in selected_peaks
            ])
            peak_pixel_traj = peak_pixel_traj.detach().cpu().numpy(
            )  # bs x T x 2 x h x w

            avobject_trajectories.append(
                peak_pixel_traj[..., [1, 0]])  # x -> y and y -> x

        avobject_trajectories = np.stack(avobject_trajectories)

        time_dim = att_map_smooth.shape[1]
        agg_att_map = np.tile(agg_att_map[:, None], [1, time_dim, 1, 1])

        return avobject_trajectories, agg_att_map, att_map_smooth

