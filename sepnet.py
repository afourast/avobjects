import numpy as np
import torch
import torch.nn as nn

from load_audio import (torch_mag_phase_2_complex_as_2d,
                        torch_phase_from_normalized_complex)
from utils import DebugModule


class Conv1dBnRelu(DebugModule):

    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, init_std=None):
        super().__init__()
        self.conv = nn.Conv1d(in_dim,
                              out_dim,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=kernel_size // 2)

        if init_std is not None:
            torch.nn.init.normal_(self.conv.weight, mean=0.0, std=init_std)
            torch.nn.init.normal_(self.conv.bias, mean=0.0, std=init_std)

        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, inp):
        out = inp
        out = self.conv(out)
        out = self.bn(out)
        out = nn.ReLU(inplace=True)(out)
        self.debug_line(self.conv, out, final_call=True)
        return out


class Conv1dStack(DebugModule):

    def __init__(self,
                 inp_dim,
                 hidden_dim,
                 nlayers,
                 strides,
                 kernel_size,
                 init_std=None,
                 project_first=1,
                 shortcut_every=1,
                 resid_preactivation=0):
        super().__init__()

        self.project_first = project_first
        self.shortcut_every = shortcut_every
        self.resid_preactivation = resid_preactivation

        if project_first:
            self.fc_inp = nn.Conv1d(inp_dim, hidden_dim, kernel_size=1)
        self.nlayers = nlayers
        self.strides = strides

        for l_id in range(self.nlayers):
            stride = strides[l_id] if not strides is None else 1
            # this to acommodate upsampling for fractional strides
            stride = int(np.ceil(stride))

            # -- depthwise separable
            self.__setattr__(
                'conv{:d}_1'.format(l_id),
                nn.Conv1d(hidden_dim,
                          hidden_dim,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=kernel_size // 2,
                          groups=hidden_dim))
            self.__setattr__('conv{:d}_2'.format(l_id),
                             nn.Conv1d(
                                 hidden_dim,
                                 hidden_dim,
                                 kernel_size=1,
                             ))

            if init_std is not None:
                torch.nn.init.normal_(self.__getattr__(
                    'conv{:d}_1'.format(l_id)).weight,
                                      mean=0.0,
                                      std=init_std)
                torch.nn.init.normal_(self.__getattr__(
                    'conv{:d}_1'.format(l_id)).bias,
                                      mean=0.0,
                                      std=init_std)
                torch.nn.init.normal_(self.__getattr__(
                    'conv{:d}_2'.format(l_id)).weight,
                                      mean=0.0,
                                      std=init_std)
                torch.nn.init.normal_(self.__getattr__(
                    'conv{:d}_2'.format(l_id)).bias,
                                      mean=0.0,
                                      std=init_std)

            self.__setattr__('bn{:d}'.format(l_id), nn.BatchNorm1d(hidden_dim))

    def forward(self, inp):

        out = inp
        if self.project_first:
            out = self.fc_inp(inp)
        shortcut = out
        for l_id in range(self.nlayers):
            stride = self.strides[l_id] if not self.strides is None else 1

            out = self.__getattr__('conv{:d}_1'.format(l_id))(out)
            out = self.__getattr__('conv{:d}_2'.format(l_id))(out)

            self.debug_line(self.__getattr__('conv{:d}_1'.format(l_id)), out)

            if stride < 1:  # we need to upsample
                stride = int(np.round(1 / stride))
                out = torch.nn.Upsample(scale_factor=stride)(out)
                shortcut = torch.nn.Upsample(scale_factor=stride)(shortcut)
                self.debug_line('Upsample', out)
            elif stride == 2:
                shortcut = nn.AvgPool1d(kernel_size=2, stride=2)(shortcut)

            if not self.resid_preactivation:
                if ((l_id + 1) % self.shortcut_every) == 0:
                    out = out + shortcut

            out = self.__getattr__('bn{:d}'.format(l_id))(out)
            out = nn.ReLU(inplace=True)(out)

            if ((l_id + 1) % self.shortcut_every) == 0:
                if self.resid_preactivation:
                    out = out + shortcut
                shortcut = out

        self.debug_line('Out', out, final_call=True)

        return out


class SepNetConversation(DebugModule):

    def __init__(self, upsample_vid=True, use_rnn=True):
        super().__init__()

        self.upsample_vid = upsample_vid
        self.use_rnn = use_rnn

        nl_vid = 10
        nl_aud = 5
        nlayers_fuse = 15
        n_channels = 1536
        kernel_size = 5

        if self.upsample_vid:
            strides_v = calc_up_down_strides(nl_vid, stride=0.5)
            strides_a = calc_up_down_strides(nl_aud, stride=1)
            strides_av = calc_up_down_strides(nlayers_fuse, stride=1)
        else:
            strides_v = calc_up_down_strides(nl_vid, stride=1)
            strides_a = calc_up_down_strides(nl_aud, stride=2)
            strides_av = calc_up_down_strides(nlayers_fuse, stride=0.5)

        self.vid_net = Conv1dStack(512, n_channels, nl_vid, strides_v,
                                   kernel_size)
        self.aud_net = Conv1dStack(80, n_channels, nl_aud, strides_a,
                                   kernel_size)

        # -----------------------------

        n_channels_vid = n_channels
        self.conv_vid = nn.Conv1d(n_channels_vid, 256, kernel_size=5, padding=2)
        self.bn_vid = nn.BatchNorm1d(256)

        self.conv_aud = nn.Conv1d(n_channels, 256, kernel_size=5, padding=2)
        self.bn_aud = nn.BatchNorm1d(256)

        if self.use_rnn:

            rnn_dim = 400

            self.rnn = nn.LSTM(512,
                               rnn_dim,
                               batch_first=True,
                               bidirectional=True)
            fuse_out_dim = rnn_dim * 2
        else:

            self.av_fuse_convnet = Conv1dStack(512, n_channels, nlayers_fuse,
                                               strides_av, kernel_size)
            fuse_out_dim = n_channels

        fc_dim = 600

        if not self.upsample_vid and self.use_rnn:
            self.fc1_up = nn.ConvTranspose1d(fuse_out_dim,
                                             fc_dim,
                                             kernel_size=2,
                                             stride=2)
            self.fc2_up = nn.ConvTranspose1d(fc_dim,
                                             fc_dim,
                                             kernel_size=2,
                                             stride=2)
            self.bn1_up = nn.BatchNorm1d(fc_dim)
            self.bn2_up = nn.BatchNorm1d(fc_dim)

            self.fc1 = nn.Conv1d(fc_dim, fc_dim, kernel_size=1)
        else:
            self.fc1 = nn.Conv1d(fuse_out_dim, fc_dim, kernel_size=1)

        self.fc2 = nn.Conv1d(fc_dim, fc_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)

        mask_out_dim = 257
        self.fc_mask = nn.Conv1d(fc_dim, mask_out_dim, kernel_size=1)

    def forward(self, inp):

        vid_inp, aud_inp = inp

        self.debug_line('SepNet: Video Input', vid_inp)
        out_vid = self.vid_net(vid_inp)

        aud_inp = aud_inp.permute([0, 2, 1])
        self.debug_line('SepNet: Audio Input', aud_inp)
        out_aud = self.aud_net(aud_inp)

        out_vid = self.conv_vid(out_vid)
        out_vid = self.bn_vid(out_vid)
        out_vid = nn.ReLU(inplace=True)(out_vid)
        self.debug_line(self.conv_vid, out_vid)

        out_aud = self.conv_aud(out_aud)
        out_aud = self.bn_aud(out_aud)
        out_aud = nn.ReLU(inplace=True)(out_aud)
        self.debug_line(self.conv_aud, out_aud)

        # The video features come from lwtnet (missing 2 each side)
        # and audio has been downsampled
        t_slice = slice(2 * 4, -2 * 4) if self.upsample_vid else slice(2, -2)
        out_aud = out_aud[..., t_slice]

        vid_aud_concat = torch.cat([out_vid, out_aud], 1)
        self.debug_line('AV concat', vid_aud_concat)

        if self.use_rnn:
            rnn_out, _ = self.rnn(vid_aud_concat.permute([0, 2, 1]))
            rnn_out = rnn_out.permute([0, 2, 1])
            self.debug_line(self.rnn, rnn_out)
            out = rnn_out

        else:
            out = self.av_fuse_convnet(vid_aud_concat)

        if not self.upsample_vid and self.use_rnn:
            out = self.fc1_up(out)
            self.debug_line(self.fc1_up, out)
            out = self.bn1_up(out)
            out = nn.ReLU(inplace=True)(out)

        out = self.fc1(out)
        self.debug_line(self.fc1, out)
        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)

        if not self.upsample_vid and self.use_rnn:
            out = self.fc2_up(out)
            self.debug_line(self.fc2_up, out)
            out = self.bn2_up(out)
            out = nn.ReLU(inplace=True)(out)

        out = self.fc2(out)
        self.debug_line(self.fc2, out)
        out = self.bn2(out)
        out = nn.ReLU(inplace=True)(out)

        out = self.fc_mask(out)
        self.debug_line('Mask', out, final_call=True)

        out = nn.Sigmoid()(out)

        return out


class PhaseNetConversation(DebugModule):

    def __init__(self):
        super().__init__()

        freq_dim = 257
        n_channels_base = 1024
        lin_pred_kern_size = 5
        n_layers = 6
        freq_dim = 257

        init_std = 0.1 / np.sqrt((freq_dim * lin_pred_kern_size))

        self.mag_prefuse_conv = Conv1dBnRelu(in_dim=freq_dim,
                                             out_dim=n_channels_base // 2,
                                             kernel_size=lin_pred_kern_size,
                                             init_std=init_std)

        self.signal_est_prefuse_conv = Conv1dBnRelu(
            in_dim=2 * freq_dim,
            out_dim=n_channels_base // 2,
            kernel_size=lin_pred_kern_size,
            init_std=init_std)

        self.phase_resid_input_conv = Conv1dBnRelu(
            in_dim=n_channels_base,
            out_dim=n_channels_base,
            kernel_size=lin_pred_kern_size,
            init_std=init_std)

        self.phase_residual_net = Conv1dStack(
            n_channels_base,
            n_channels_base,
            n_layers,
            kernel_size=lin_pred_kern_size,
            strides=None,
            init_std=init_std,
            project_first=0,
            shortcut_every=2,
            resid_preactivation=1,
        )

        self.phase_resid_out_conv = Conv1dBnRelu(in_dim=n_channels_base,
                                                 out_dim=freq_dim * 2,
                                                 kernel_size=1,
                                                 init_std=init_std)

    def forward(self, inp):

        enh_mag, mix_phase = inp  # (b x t x f) , (b x t x f x 2)

        signal_est = torch_mag_phase_2_complex_as_2d(
            enh_mag, mix_phase)  # bs x f x t x 2
        bs, freq_dim, t_dim = enh_mag.shape
        signal_est = signal_est.permute([0, 1, 3, 2])  # b x f x 2 x t
        signal_est = signal_est.reshape([bs, freq_dim * 2, t_dim])

        signal_est_prefuse = self.signal_est_prefuse_conv(signal_est)
        mag_prefuse = self.mag_prefuse_conv(enh_mag)

        signal_mag_concat = torch.cat([signal_est_prefuse, mag_prefuse], 1)

        phase_resid_input = self.phase_resid_input_conv(signal_mag_concat)
        phase_residual = self.phase_residual_net(phase_resid_input)
        phase_residual = self.phase_resid_out_conv(phase_residual)

        phase_residual = phase_residual.reshape(
            (bs, freq_dim, 2, t_dim)).permute([0, 1, 3, 2])

        mix_phase_as_2d = torch_mag_phase_2_complex_as_2d(
            torch.ones_like(enh_mag), mix_phase)  # bs x f x t x 2
        phase_pred_unnormalized = mix_phase_as_2d + phase_residual

        enh_phase_as_2d = torch.nn.functional.normalize(phase_pred_unnormalized,
                                                        p=2,
                                                        dim=-1)  # b x t x f x 2
        enh_phase_angle = torch_phase_from_normalized_complex(enh_phase_as_2d)

        return enh_phase_as_2d, enh_phase_angle


def calc_up_down_strides(num_layers, stride):
    """
    Calculates where to place downsampling (stride=2) or upsampling (stride=0.5) strides.
    The method is hardcoded for a total x4 sub(up)sampling, i.e. 2 layers.
    The strides are placed at the 2/5 and 4/5 of the total number of layers.
    :param num_layers: Total number of layers over which the sub(up)sumpling will take place
    :param stride: The stride - should be 2 to denote subsampling and 0.5 for upsampling
    :return: integer array with #num_layers elements, all but 2 equal to 1
    Example: calc_up_down_strides(5, 2) =  [ 1, 2, 1, 2, 1 ]
    """

    def round_up_to_even(f):
        return int(np.ceil(f / 2.) * 2)

    strides = [1] * num_layers
    # round to even so that we can do shortcut every 2
    stride_inds = round_up_to_even(2 * num_layers / 5.), round_up_to_even(
        4 * num_layers / 5.)
    for ind in stride_inds:
        strides[ind - 1] = stride
    return strides
