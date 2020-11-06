import numpy as np


def load_mp4(vid_path):

    import av

    container = av.open(vid_path)

    ims = [frame.to_image() for frame in container.decode(video=0)]

    ims_c = np.array([np.array(im) for im in ims])

    return ims_c


def load_mp4_ffmpeg(vid_path, grey=1, resolution=None):

    import ffmpeg

    probe = ffmpeg.probe(vid_path)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    out, _ = (
        ffmpeg
        .input(vid_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .global_args('-loglevel', 'error')
        .run(capture_stdout=True)
    )
    ims = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )

    if resolution is not None or grey:
        from PIL import Image
        ims = [Image.fromarray(im) for im in ims]

        if resolution:
            ims = [im.resize(resolution) for im in ims]

        if grey:
            ims = [im.convert('L') for im in ims]

        ims_c = np.array([np.array(im) for im in ims])
    else:
        ims_c = ims

    if grey:
        ims_c = np.expand_dims(ims_c, axis=3)

    return ims_c
