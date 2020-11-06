import numpy as np
import torch
from tqdm import tqdm

from utils import map_to_full_torch


class Warper():

    def __init__(self, device='cuda:0'):
        self.grid_offset = None
        self.base = None
        self.map_full = None
        self.device = device

    def warp(self, im, flow, padding_mode='border'):
        # im : b x C x H x W
        # flow : b x 2 x H x W, such that flow[dst_y, dst_x] = (src_x, src_y),
        #     where (src_x, src_y) is the pixel location we want to sample from.

        ft = torch.FloatTensor
        imtype = im.dtype

        im = im.float()

        if self.grid_offset is None:
            self.grid_offset = ft(
                np.array([im.shape[-1], im.shape[-2]],
                         np.float32)).to(self.device)
            self.grid_offset = (-1 + self.grid_offset)[:, None,
                                                       None].contiguous()

        # -- slow, ~60% of time
        grid = -1. + 2. * flow / self.grid_offset
        grid = grid.permute((0, 2, 3, 1))

        inp = im

        # inp:  bs x C x h x w
        # grid: bs x h x w x 2

        warped = torch.nn.functional.grid_sample(inp,
                                                 grid,
                                                 mode='bilinear',
                                                 padding_mode=padding_mode)
        # warped: bs x C x h x w

        return warped

    def integrate_att_map_over_flow_trajectories(self, flow, att_map,
                                                 att_map_offset):
        """
        :param flow: t x h x w x 2
        :param att_map: t x h_map x w_map
        """

        bs, T, h_full, w_full, _ = flow.shape

        if self.base is None:
            self.base = np.mgrid[:h_full, :w_full].astype(
                np.float32)[::-1].copy()
            self.base = torch.from_numpy(self.base).to(flow.device)

        tracks = self.base[None].repeat([bs, 1, 1, 1]).to(
            self.device)  # tile over batch to init
        flow_abs = flow.permute([0, 1, 4, 2, 3]) + self.base

        track_scores = []
        pixel_trajectories = []
        assert att_map.shape[1] == T

        for t_id in tqdm(range(T), desc='Averaging scores over flow tracks'):
            map_full_mid, offset = map_to_full_torch(att_map[:, t_id], w_full,
                                                     h_full, att_map_offset)

            # ---- pad the edges with the min value
            map_min = att_map[:, t_id].detach().min(-1)[0].min(-1)[0] - 1
            map_min = map_min[:, None, None].double()

            if self.map_full is None:  # the dynamic time dim changes the batch size of the combined tensor
                self.map_full = (torch.ones(
                    (bs, h_full, w_full), dtype=torch.float64)).to(self.device)
            self.map_full[:] = map_min
            self.map_full[:, offset:h_full - offset,
                          offset:w_full - offset] = map_full_mid

            track_sc = self.warp(self.map_full[:, None],
                                 tracks,
                                 padding_mode='border').squeeze(1)

            track_scores.append(track_sc.detach().cpu())
            pixel_trajectories.append(tracks.detach().cpu())

            tracks = self.warp(flow_abs[:, t_id].to(tracks.device), tracks)

        track_scores = torch.stack(track_scores, 1)
        track_scores = track_scores.double()
        # average the scores over the trajectories
        track_score_map = track_scores.mean(1)
        pixel_trajectories = torch.stack(pixel_trajectories, 1)

        return track_score_map, pixel_trajectories, track_scores
