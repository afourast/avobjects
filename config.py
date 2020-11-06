import configargparse


def save_opts(args, fn):
    with open(fn, 'w') as fw:
        for items in vars(args):
            fw.write('%s %s\n' % (items, vars(args)[items]))


def load_opts():
    parser = configargparse.ArgumentParser(description="main")

    parser.add('-c', '--config', is_config_file=True, help='config file path')

    # --- general
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0,
                        help='-1: all, 0-7: GPU index')

    parser.add_argument('--output_dir',
                        type=str,
                        default="./save",
                        help='Path for saving results')

    parser.add_argument('--n_workers',
                        type=int,
                        default=0,
                        help='Num data workers')

    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')

    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Checkpoints to load model weights from')

    parser.add_argument('--input_video',
                        type=str,
                        default="./save",
                        help='Input video path')

    # --- video
    parser.add_argument('--resize',
                        default=540,
                        type=int,
                        help='Scale input video to that resolution')
    parser.add_argument('--fps', type=int, default=25, help='Video input fps')

    # --- audio
    parser.add_argument('--sample_rate', type=int, default=16000, help='')

    # -- avobjects
    parser.add_argument( '--n_negative_samples',
                        type=int,
                        default=30,
                        help='Shift range used for synchronization.'
                        'E.g. set to 30 from -15 to +15 frame shifts'
    )
    parser.add_argument('--n_peaks',
                        default=4,
                        type=int,
                        help='Number of peaks to use for separation')

    parser.add_argument('--nms_thresh',
                        type=int,
                        default=100,
                        help='Area for thresholding nms in pixels')

    # -- viz
    parser.add_argument('--const_box_size',
                        type=int,
                        default=80,
                        help='Size of bounding box in visualization')

    args = parser.parse_args()

    return args
