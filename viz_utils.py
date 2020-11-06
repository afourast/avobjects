import os
import numpy as np

from utils import map_to_full


class VideoSaver:

    def __init__(self, savedir):
        try:
            os.makedirs(savedir)
        except:
            pass
        self.savedir = savedir
        self.id = 0

    def save_mp4_from_vid_and_audio(self,
                                    video_tensor,
                                    audio_wav=None,
                                    fps=25,
                                    sr=16000,
                                    outname=None,
                                    extract_frames_hop=None):
        """
        :param video_tensor: tchw
        :param sr:
        :return:
        """

        from moviepy.audio.AudioClip import AudioArrayClip
        from moviepy.video.VideoClip import VideoClip

        video_tensor = video_tensor.transpose([0, 2, 3, 1])  # thwc
        # that's to avoid error due to float precision
        vid_dur = len(video_tensor) * (1. / fps) - 1e-6
        v_clip = VideoClip(lambda t: video_tensor[int(np.round(t * 25))],
                           duration=vid_dur)

        import tempfile

        if outname:
            outfile = os.path.join(self.savedir, outname)
            if not outfile.endswith('.mp4'):
                outfile += '.mp4'
        else:
            outfile = os.path.join(self.savedir, '%03d.mp4' % self.id)

        if audio_wav is not None:
            _, temp_audiofile = tempfile.mkstemp(dir='/dev/shm', suffix='.wav')
            import torch
            if isinstance(audio_wav, torch.Tensor):
                audio_wav = audio_wav.numpy()

            import scipy.io
            scipy.io.wavfile.write(temp_audiofile, 16000, audio_wav)

        self.id += 1
        try:
            os.makedirs(os.path.dirname(outfile))
        except:
            pass
        _, temp_videofile = tempfile.mkstemp(dir='/dev/shm', suffix='.mp4')

        v_clip.write_videofile(temp_videofile, fps=25, verbose=False)

        if audio_wav is not None:
            command = ("ffmpeg -threads 1 -loglevel error -y -i {} -i {} " 
                      "-c:v copy -map 0:v:0 -map 1:a:0 -pix_fmt yuv420p "
                      "-shortest {}").format(temp_videofile, temp_audiofile, outfile)
            from subprocess import call
            cmd = command.split(' ')
            call(cmd)
        else:
            import shutil
            shutil.move(temp_videofile, outfile)

        v_clip.close()
        import imageio
        if extract_frames_hop:  # extract the video as frames for paper
            frames_dir = os.path.join(
                os.path.dirname(outfile),
                'frames_' + os.path.basename(outfile).replace('.mp4', ''))
            os.makedirs(frames_dir, exist_ok=True)
            import scipy.misc
            for fr_id, frame in enumerate(video_tensor[::extract_frames_hop]):
                scipy.misc.imsave(frames_dir + '/%04d.png' % fr_id,
                                  frame[:, :-5, :])
            pass


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


# @profile
def show_cam_on_vid(vid, cam, offset=0):
    """
    :param vid: t x h x w x c
    :param cam: h_att x w_att
    :return:
    """

    assert len(cam) == len(vid)

    vids = {}
    vmin = cam.min()
    vmax = cam.max()
    vmin = vmax = None
    vid_with_cam = np.array([
        show_cam_on_image(frame, msk, offset, vmin, vmax)
        for frame, msk in zip(vid, cam)
    ])
    return vid_with_cam


def viz_boxes_with_scores(video,
                          box_centers,
                          scores=None,
                          const_box_size=None,
                          colors = None,
                          asd_thresh=None):
    """
    video: np array -> t h w c
    """
    import aolib_p3.util as ut
    import aolib_p3.img as ig
    if colors is None:
        colors = ut.distinct_colors(len(box_centers))
    peaks_on_vid_viz = []

    def add_cont_bb_size_to_traj(box_centers, const_box_size):
        const_box_size = np.array([const_box_size, const_box_size])
        const_box_size = np.tile(const_box_size[None, None],
                                box_centers.shape[:2] + (1,))
        box_centers = np.concatenate( [box_centers, const_box_size], -1)
        return box_centers

    if box_centers.shape[-1] == 2:  # no box size, need to pad it
        box_centers = add_cont_bb_size_to_traj(box_centers,
                                               const_box_size)
    bb_sizes = box_centers[..., 2:]
    box_centers = box_centers[..., :2]

    if scores is not None:
        padlen = box_centers.shape[1] - scores.shape[-1]
        scores = np.pad(scores,
                                     [[0, 0], [padlen // 2, padlen // 2]],
                                     mode='edge')

    for tt in range(len(video)):

        border_width = 3
        track_vis = video[tt]

        def make_text(track_vis,
                      scores,
                      const_off=40,
                      relative_off=1,
                      fmt='{:.2f}',
                      font_size=30):
            texts = list(map(lambda xx: fmt.format(xx), scores))
            if relative_off:
                txt_off = const_off + border_width
                text_loc = box_centers[:, tt] + \
                    np.array([-txt_off, txt_off])
            else:
                text_loc = np.array([const_off, const_off
                                    ])[None].repeat(box_centers.shape[0], 0)
            track_vis = ig.draw_text(track_vis,
                                     texts,
                                     text_loc,
                                     colors,
                                     font_size=font_size)
            return track_vis

        if scores is not None:
            asd_scores = scores[:, tt]
            track_vis = make_text(track_vis, asd_scores)

            pnt_locs = []
            cols = []
            wds = int(bb_sizes.mean())

            for ii, asd_sc in enumerate(asd_scores):
                if asd_sc > asd_thresh:
                    pnt_locs.append(box_centers[ii, tt])
                    cols.append(colors[ii])

            track_vis = draw_hollow_rects(track_vis,
                                          np.array(pnt_locs),
                                          cols,
                                          width=wds,
                                          border_width=border_width)

        else:
            track_vis = draw_hollow_rects(track_vis,
                                          box_centers[:, tt],
                                          colors,
                                          width=bb_sizes[:, tt],
                                          border_width=border_width)

        peaks_on_vid_viz.append(track_vis)

    peaks_on_vid_viz = np.array(peaks_on_vid_viz)
    vid_top_trajectories_viz = peaks_on_vid_viz.transpose([0, 3, 1, 2])

    return vid_top_trajectories_viz


def draw_hollow_rects(im,
                      points,
                      colors=None,
                      width=1,
                      border_width=None,
                      texts=None):
    import aolib_p3.img as ig
    points = list(points)
    colors = ig.colors_from_input(colors, (255, 0, 0), len(points))
    if isinstance(width, int):
        heights = widths = [width] * len(points)
    else:
        assert len(width) == len(points)
        widths, heights = np.array(width).T

    rects = [(p[0] - width / 2, p[1] - height / 2, width, height)
             for p, width, height in zip(points, widths, heights)]
    line_widths = None
    if border_width is not None:
        line_widths = [border_width] * len(points)

    return ig.draw_rects(im,
                         rects,
                         fills=[None] * len(points),
                         outlines=colors,
                         texts=texts,
                         line_widths=line_widths)


def show_cam_on_image(frame, cam, offset, vmin=None, vmax=None):
    """
    :param frame: c x h x w
    :param cam: h_att x w_att
    :return:
    """

    # frame = frame.transpose([1, 2, 0])  # chw --> hwc
    frame = np.float32(frame) / 255

    import cv2

    if vmin is not None:
        vmax = -vmin
        vmin = -vmax

    cam = normalize_img(-cam, vmin=vmin, vmax=vmax)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    h_frame, w_frame = frame.shape[:2]
    heatmap, offset = map_to_full(heatmap,
                                  w_frame,
                                  h_frame,
                                  offset,
                                  w_map=heatmap.shape[1])
    heatmap = np.float32(heatmap) / 255
    heatmap_frame = np.zeros_like(frame)
    heatmap_frame[offset:h_frame - offset, offset:w_frame - offset] = heatmap

    cam = heatmap_frame + frame
    cam = cam / np.max(cam)

    new_img = np.uint8(255 * cam)

    new_img = new_img.transpose([2, 0, 1])  # hwc --> chw

    return new_img


def viz_avobjects(
    video,
    audio,
    att_map,
    avobject_traj,
    model_start_offset,
    video_saver,
    const_box_size,
    step,
    asd_thresh=None,
    vids_name='avobject_viz'):
    """
    video: c T H W 
    att_map: t h w 
    """
    print('Vizualizaing av att and avobject trajectories')

    video = video.permute([1,2,3,0]).numpy().astype('uint8') # C T H W -> T H W C

    # ----------- make cam_vid showing AV-att map and peaks  ---------------

    vid_with_cam = show_cam_on_vid(video,
                                    att_map.detach().cpu(),
                                    offset=model_start_offset)

    vid_avobject = viz_boxes_with_scores(
        video,
        avobject_traj[..., [1, 0]], # switch x and y coords
        const_box_size=const_box_size
    )

    # remove padding equal to the model's conv offset
    pad_len = model_start_offset
    vid_with_cam = vid_with_cam[..., pad_len:-pad_len, pad_len:-pad_len]
    vid_avobject = vid_avobject[..., pad_len:-pad_len, pad_len:-pad_len]

    video_saver.save_mp4_from_vid_and_audio(
        np.concatenate([vid_with_cam, vid_avobject], axis=3),
        audio / 32768,
        outname='{}/{}'.format(vids_name, step),
    )

def viz_source_separation(video,
                          enh_audio,
                          avobject_traj,
                          model_start_offset,
                          const_box_size,
                          video_saver,
                          step):

    video = video.permute([1,2,3,0]).numpy().astype('uint8') # C T H W -> T H W C

    assert avobject_traj.shape[0] == enh_audio.shape[0]
    n_objects = avobject_traj.shape[0]

    import aolib_p3.util as ut
    colors = ut.distinct_colors(n_objects)

    for ii in range(n_objects):

        vid_avobject = viz_boxes_with_scores(
            video,
            avobject_traj[ ii:ii+1, :, [1, 0]], # switch x and y coords
            const_box_size=const_box_size,
            colors = [colors[ii]]
        )

        # remove padding equal to the model's conv offset
        pad_len = model_start_offset
        vid_avobject = vid_avobject[..., pad_len:-pad_len, pad_len:-pad_len]

        # vid_sep = video[0:1, ii].numpy().astype('uint8')
        # vid_sep = vid_sep.transpose([0, 2, 1, 3, 4])
        video_saver.save_mp4_from_vid_and_audio(
            vid_avobject,
            enh_audio[ii],
            outname='sep_vid/{}/enh_{}'.format(step, ii))

