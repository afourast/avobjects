import matplotlib.colors as colors
import os
import tempfile
import numpy as np
import torch
import math
from torch import __init__
from torch import nn
from tqdm import tqdm

# ----------------- model utils -----------------------------------------------------


class DebugModule(nn.Module):
    """
    Wrapper class for printing the activation dimensions 
    """

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.debug_log = True

    def debug_line(self, layer_str, output, memuse=1, final_call=False):
        if self.debug_log:
            namestr = '{}: '.format(self.name) if self.name is not None else ''
            print('{}{:80s}: dims {}'.format(namestr, repr(layer_str),
                                             output.shape))

            if final_call:
                self.debug_log = False
                print()


def load_checkpoint(chkpt, model):
    load_model_params(model, chkpt)
    print(colorize("Checkpoint {} loaded!".format(chkpt), 'green'))


def load_model_params(model, path):

    loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
    self_state = model.state_dict()

    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")

            if name not in self_state:
                print(colorize("%s is not in the model." % origname, 'red'))
                continue

        if self_state[name].size() != param.size():
            if np.prod(param.shape) == np.prod(self_state[name].shape):
                print(
                    colorize(
                        "Caution! Parameter length: {}, model: {}, loaded: {}, Reshaping"
                        .format(origname, self_state[name].shape,
                                loaded_state[origname].shape), 'red'))
                param = param.reshape(self_state[name].shape)
            else:
                print(
                    colorize(
                        "Wrong parameter length: {}, model: {}, loaded: {}".
                        format(origname, self_state[name].shape,
                               loaded_state[origname].shape), 'red'))
                continue

        self_state[name].copy_(param)


def calc_receptive_field(layers, imsize, layer_names=None):
    if layer_names is not None:
        print("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]

    for l_id, layer in enumerate(layers):
        conv = [
            layer[key][-1] if type(layer[key]) in [list, tuple] else layer[key]
            for key in ['kernel_size', 'stride', 'padding']
        ]
        currentLayer = outFromIn(conv, currentLayer)
        if 'maxpool' in layer:
            conv = [
                (layer['maxpool'][key][-1] if type(layer['maxpool'][key])
                 in [list, tuple] else layer['maxpool'][key]) if
                (not key == 'padding' or 'padding' in layer['maxpool']) else 0
                for key in ['kernel_size', 'stride', 'padding']
            ]
            currentLayer = outFromIn(conv, currentLayer, ceil_mode=False)
    return currentLayer


def outFromIn(conv, layerIn, ceil_mode=True):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2 * p) / s) + 1
    actualP = (n_out - 1) * s - n_in + k
    pR = math.ceil(actualP / 2)
    pL = math.floor(actualP / 2)

    j_out = j_in * s
    r_out = r_in + (k - 1) * j_in
    start_out = start_in + ((k - 1) / 2 - pL) * j_in
    return n_out, j_out, r_out, start_out


# ----------------------------------------------------------------------


def gpu_initializer(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    return device


def my_unfold(tens, size, step, dimension, chunk_at_least):
    """
    Unfolds into list allowing uneven last chunk, which is appending to the penultimate one
    """
    fr = 0
    out = []
    done = 0
    while fr < tens.shape[dimension]:
        length = min(size, tens.shape[dimension] - fr)
        if tens.shape[dimension] - (fr + length) < chunk_at_least:
            # permit last chunk to be that longer so that it takes all the sequence
            length = tens.shape[dimension] - fr
            done = 1
        out.append(tens.narrow(dimension, fr, length))
        if done:
            break
        fr += step
    return out


# ---------------------- peaks + NMS -------------------------------------


def detect_peaks(image, overlap_thresh=10):
    """
    https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array

    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background,
                                       structure=neighborhood,
                                       border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    peak_coords = np.array(np.where(detected_peaks)).T

    detected_peaks, peak_coords = non_max_suppression_fast(
        detected_peaks, image, overlap_thresh=overlap_thresh)

    return detected_peaks, peak_coords


# Malisiewicz et al.


def non_max_suppression_fast(peaks_map, values_map, overlap_thresh):
    """
    adapted from 
    https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    # if there are no boxes, return an empty list
    if len(peaks_map) == 0:
        return []

    ww = peaks_map.shape[-1]
    peak_coords = np.array(np.where(peaks_map)).T.astype(int)

    values = values_map[np.where(peaks_map)]

    # initialize the list of picked indexes
    pick = []

    idxs = np.argsort(values)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        last = len(idxs) - 1
        ii = idxs[last]
        pick.append(ii)
        dif = np.abs(peak_coords[ii] - peak_coords[idxs]).sum(
            -1)  # manhattan distance
        idxs = np.delete(idxs, np.where(dif <= overlap_thresh))

    peak_coords_out = peak_coords[pick].T
    out_map = np.zeros_like(peaks_map)
    out_map[(peak_coords_out[0], peak_coords_out[1])] = 1

    debug = 0
    if debug:
        import matplotlib
        matplotlib.use('GTK3Agg')
        import matplotlib.pyplot as plt

        plt.imshow(out_map)
        plt.title('output binary')

        plt.figure()
        in_map = np.ones_like(values_map) * values_map.min()
        in_map[(peak_coords.T[0],
                peak_coords.T[1])] = values_map[(peak_coords.T[0],
                                                 peak_coords.T[1])]
        plt.imshow(in_map)

        plt.figure()
        out_map_val = np.ones_like(values_map) * values_map.min()
        out_map_val[(peak_coords_out[0],
                     peak_coords_out[1])] = values_map[(peak_coords_out[0],
                                                        peak_coords_out[1])]
        plt.imshow(out_map_val)
        plt.show()

    return out_map, peak_coords_out.T


# ----------------- utils for mapping from original image coordinates (full) to feature map / attention map coordinates ----------------

def map_to_full_torch(map_in, w_frame, h_frame, offset):
    offset, h_map, w_map, h_att, w_att = calc_map_offset(
        offset, h_frame, w_frame, map_in.shape[-1])
    import torch
    interp = 'area'
    map_full = torch.nn.functional.interpolate(map_in[None],
                                              size=(h_map, w_map),
                                              mode=interp).squeeze()
    return map_full, offset


def map_to_full(map_in, w_frame, h_frame, offset, w_map=None):
    w_map = w_map or map_in.shape[-1]
    from PIL import Image
    hm_im = Image.fromarray(map_in)
    offset, h_map, w_map, h_att, w_att = calc_map_offset(
        offset, h_frame, w_frame, w_map)
    hm_im = hm_im.resize((w_map, h_map))
    map_full = np.array(hm_im)
    return map_full, offset


def calc_map_offset(offset, h_frame, w_frame, w_map):
    # this is without the edge for going between map coords and original image pixels
    w_att, h_att = w_frame - 2 * offset, h_frame - 2 * offset
    edge = int(np.round((w_frame - 2 * offset) / (w_map - 1) / 2))
    offset -= edge
    w_map, h_map = w_frame - 2 * offset, h_frame - 2 * offset
    return offset, h_map, w_map, h_att, w_att


def full_to_map(coords_full, h_full, w_full, h_map, w_map, start_offset):
    # this is without the edge for going between map coords and original image pixels
    w_att, h_att = w_full - 2 * start_offset, h_full - 2 * start_offset
    map_ratio = np.array((h_att / h_map, w_att / w_map))
    if isinstance(coords_full, torch.Tensor):
        coords_map = torch.round((coords_full - start_offset) /
                                  torch.from_numpy(map_ratio).float()).int()

        coords_map[..., 0] = coords_map[..., 0].clamp(0, h_map - 1)
        coords_map[..., 1] = coords_map[..., 1].clamp(0, w_map - 1)

        assert (coords_map[...] < 0).sum() == 0
        assert (coords_map[..., 0] > h_map).sum() == 0
        assert (coords_map[..., 1] > w_map).sum() == 0

    else:
        coords_map = np.round(
            (coords_full - start_offset) / map_ratio).astype(int)

        coords_map[..., 0] = coords_map[..., 0].clip(0, h_map - 1)
        coords_map[..., 1] = coords_map[..., 1].clip(0, w_map - 1)

    return coords_map


# ----------------------------------------------------------------------------------------------------

# ----------------------  cropping utils  ------------------------------------------------------------


def extract_attended_features(att_map, vid_emb, peak_traj_map):

    weighted_feats = []

    for b_id in range(len(vid_emb)):

        # w_feats = torch.stack( [(fp * atw).sum(-1).sum(-1) for fp, atw in zip(feat_patches, att_weights)], 1)
        width = 3
        b_id_central_peak = peak_traj_map[b_id, :, slice(2, -2)]
        feat_patches = torch.stack([
            crop_feat_patch(vid_emb[b_id:b_id + 1], peak, width)
            for peak in b_id_central_peak
        ], 1)  # n_peaks x T x 1 x ff x  h x w
        att_weights = torch.stack([
            crop_feat_patch(att_map[b_id:b_id + 1], peak, width, softmax=1)
            for peak in b_id_central_peak
        ], 1)  # n_peaks x T x 1 x 1 x  h x w
        w_feats = (feat_patches.to(att_weights.device) * att_weights).sum(
            [-2, -1])

        weighted_feats.append(w_feats)

    w_feats = torch.cat(weighted_feats, 0)
    # w_feats = w_feats.reshape((-1,) + w_feats.shape[2:])

    return w_feats


def extract_face_crops(vid_frames_for_crop, avobject_traj, crop_size):
    face_crop_vids = []
    bs = len(vid_frames_for_crop)
    for b_id in range(bs):
        h_full, w_full = vid_frames_for_crop.shape[-2:]
        face_crops = torch.stack([
            crop_feat_patch(vid_frames_for_crop[b_id:b_id + 1],
                            peak.clip((0, 0), (h_full, w_full)), crop_size)
            for peak in avobject_traj[b_id].astype(int)
        ], 1)
        face_crop_vids.append(face_crops)
    face_crop_vids = torch.cat(face_crop_vids, 0)
    return face_crop_vids


def crop_feat_patch(feat_map, peak, ww, softmax=0):
    # peak: T x 2
    # feat_map: 1 x d x T x h x w

    peak = peak.copy()
    NEG_INF = -1e10
    pad_val = NEG_INF if softmax else 0

    padlen = ww // 2
    feat_map = nn.ConstantPad2d([padlen, padlen, padlen, padlen],
                                pad_val)(feat_map)
    peak += padlen

    if len(peak.shape) == 1:  # Same map over all time steps
        c_y, c_x = peak
        feat_patch = feat_map[...,
                              max(c_y - ww // 2, 0):c_y + ww // 2 + 1,
                              max(c_x - ww // 2, 0):c_x + ww // 2 + 1]
    else:
        feat_patch = []
        # check that  time dims are the same
        assert peak.shape[0] == feat_map.shape[-3], 'Time dims are different'
        for t_i, (c_y, c_x) in enumerate(peak):

            fp = feat_map[..., t_i,
                          max(c_y - ww // 2, 0):c_y + ww // 2 + 1,
                          max(c_x - ww // 2, 0):c_x + ww // 2 + 1]
            feat_patch.append(fp)

        feat_patch = np.stack(feat_patch, 2) if isinstance(
            feat_map, np.ndarray) else torch.stack(feat_patch, 2)

    if softmax:
        feat_patch = logsoftmax_2d(feat_patch).exp()

    return feat_patch


# ----------------------


def logsoftmax_2d(logits):
    # Log softmax on last 2 dims because torch won't allow multiple dims
    orig_shape = logits.shape
    logprobs = torch.nn.LogSoftmax(dim=-1)(
        logits.reshape(list(logits.shape[:-2]) + [-1])).reshape(orig_shape)
    return logprobs


def run_func_in_parts(func, vid_emb, aud_emb, part_len, dim, device):
    """
    Run given function in parts, spliting the inputs on dimension dim 
    This is used to save memory when inputs too large to compute on gpu 
    """
    dist_chunk = []
    for v_spl, a_spl in tqdm(list(
            zip(vid_emb.split(part_len, dim=dim),
                aud_emb.split(part_len, dim=dim))),
                             desc='Calculating pairwise scores'):
        dist_chunk.append(func(v_spl.to(device), a_spl.to(device)))
    dist = torch.cat(dist_chunk, dim - 1)
    return dist

# ---------------------- flow wrapper -------------------------

def calc_flow_on_vid_wrapper(ims, tmp_dir='/dev/shm', gpu_id=0):
    """
    Wrapper for calling PWC-net through a separate process 
    """

    # Free GPU memory before running flow in another process.
    torch.cuda.empty_cache()
    
    input_ims_path = tempfile.NamedTemporaryFile(suffix='.npy', dir=tmp_dir).name 
    output_flow_path = tempfile.NamedTemporaryFile(suffix='.npy', dir=tmp_dir).name  

    np.save(input_ims_path, ims)

    command = "python flow/pwcnet.py {} {} {}".format(input_ims_path, output_flow_path, gpu_id) 
    from subprocess import call
    cmd = command.split(' ')
    call(cmd)

    flow = np.load(output_flow_path)

    os.remove(input_ims_path)
    os.remove(output_flow_path)

    return flow 

# -------------------------- colorize utils -----------------------------------------------
"""
Borrowed from Tom Jakab & Ankush Gupta 
https://github.com/tomasjakab/imm/blob/3f34424b853c9ead980a9b7f116d47b56d476b58/imm/utils/colorize.py

A set of common utilities used within the environments. These are
not intended as API functions, and will not remain stable over time.
"""

color2num = dict(gray=30,
                 red=31,
                 green=32,
                 yellow=33,
                 blue=34,
                 magenta=35,
                 cyan=36,
                 white=37,
                 crimson=38)


def colorize(string, color, bold=False, highlight=False):
    """Return string surrounded by appropriate terminal color codes to
    print colorized text.  Valid colors: gray, red, green, yellow,
    blue, magenta, cyan, white, crimson
    """

    # Import six here so that `utils` has no import-time dependencies.
    # We want this since we use `utils` during our import-time sanity checks
    # that verify that our dependencies (including six) are actually present.
    import six

    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(six.u(str(num)))
    if bold:
        attr.append(six.u('1'))
    attrs = six.u(';').join(attr)
    return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)


def green(s):
    return colorize(s, 'green', bold=True)


def blue(s):
    return colorize(s, 'blue', bold=True)


def red(s):
    return colorize(s, 'red', bold=True)


def magenta(s):
    return colorize(s, 'magenta', bold=True)
