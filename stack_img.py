"""
Usage:
Does not support video file input. Images to be stacked must be placed
into a folder, the folder is passed to the script.

Single image stack
python -m stack_img stack <FOLDER>

Animate the stacking
python -m stack_img animate <FOLDER>

Stack every Nth image
python -m stack_img animate <FOLDER> --nframe 2

Limit the number of a frames in a stack (animated tail)
python -m stack_img animate <FOLDER> --frange 60

Combininations are supported
python -m stack_img animate <FOLDER> --frange 5 --nframe 2
"""
import sys
import os
import tempfile
import shutil
import argparse
import numpy
from scipy import stats
from PIL import Image
from subprocess import Popen

### TODO ###
# missing operators
#entropy - scipy.stats.entropy has weird results
#midrange? https://stackoverflow.com/questions/23855976/middle-point-of-each-pair-of-an-numpy-array
#outlier? https://blog.finxter.com/how-to-find-outliers-in-python-easily/

### TODO ###
# -crf is for 8 bit, may need -qp for 10 bit (yuv420p10le) later on
# but ffmpeg builds may not support it

FFMPEG = 'ffmpeg -r {fps} -f image2 -i '\
         '"{input}" -vcodec {codec} '\
         '-crf {crf} -pix_fmt {pix_fmt} -y "{output}"'
NFRAME = 1
FRANGE = -1
VCODEC = 'libx264'
PIX_FMT = 'yuv420p'
CONSTANT_RATE = 17
MOVIE_EXTENSIONS = ('MP4', 'MOV', 'AVI')
IMG_EXTENSIONS = ('JPG', 'JPEG', 'PNG', 'TIF', 'TIFF')
FPS = 30


RENDERING = {
    'kurtosis': stats.kurtosis,
    'maximum': numpy.amax,
    'mean': numpy.mean,
    'median': numpy.median,
    'minimum': numpy.amin,
    'skewness': stats.skew,
    'standard-deviation': numpy.std,
    'summation': numpy.sum,
    'variance': numpy.var,
    'range': numpy.ptp
}


def animate(images, output,
            rendering='median',
            framerate=FPS,
            codec=VCODEC,
            pix_fmt=PIX_FMT,
            constant_rate=CONSTANT_RATE,
            nframe=NFRAME,
            frange=FRANGE,
            preserve_length=False):
    prefix = os.path.basename(os.path.splitext(__file__)[0])
    tmpdir = tempfile.mkdtemp(prefix='{}-'.format(prefix))
    ext = _ext(images[0])
    total = len(images)
    img_seq = 1
    frange = total if frange == FRANGE else frange
    img_groups = _img_groups(total, nframe, frange)
    total = sum([len(x) for x in img_groups])

    progress = 0
    for grp_index, img_group in enumerate(img_groups):
        subtemp = tempfile.mkdtemp(dir=tmpdir)
        tmpfiles = []

        for img_index, image in enumerate(img_group):

            if img_index == 0:
                x = images[image]
            else:
                x = tmpfiles[-1]

            try:
                y = images[img_group[img_index+1]]
            except IndexError:
                break

            progress += 1
            name = 'tmp{}-{}{}'.format(grp_index, img_index, ext)
            tmpfiles.append(os.path.join(subtemp, name))
            _stack([x, y], tmpfiles[-1], rendering)

            print("Stacking progress: {0:.1f}%".format((progress / total) * 100))

        tmpfiles = tmpfiles if grp_index == 0 else tmpfiles[-1:]

        for tmpfile in tmpfiles:
            name = 'stacked.{:05d}{}'.format(img_seq, ext)
            dest = os.path.join(tmpdir, name)
            shutil.move(tmpfile, dest)
            img_seq += 1

            #@TODO: would need a flag for this?
            if not preserve_length:
                continue

            for _ in range(1, nframe):
                name = 'stacked.{:05d}{}'.format(img_seq, ext)
                shutil.copy(dest, os.path.join(tmpdir, name))
                img_seq += 1
        
        shutil.rmtree(subtemp)
    
    input_file = 'stacked.%05d{}'.format(ext)
    input_file = os.path.join(tmpdir, input_file)

    cmd = FFMPEG.format(input=input_file, 
                        output=output,
                        fps=framerate,
                        crf=constant_rate,
                        pix_fmt=pix_fmt,
                        codec=codec)
    proc = Popen(cmd, shell=True)
    proc.wait()
    shutil.rmtree(tmpdir)


def stack(images, output, 
          rendering='median',
          nframe=NFRAME):
    total = len(images)
    indices = _img_groups(total, nframe, total)[0]
    images = [images[i] for i in indices]
    _stack(images, output, rendering)


def _stack(images, output, rendering):
    ext = _ext(images[0])[1:].lower()
    try:
        func = RENDERING[rendering]
    except KeyError:
        error = "Unsupported rendering '{}'".format(rendering)
        raise RuntimeError(error)
    images = [numpy.array(Image.open(i)) for i in images]
    stacked = numpy.uint8(func(images, axis=0))
    newarray = Image.fromarray(stacked)
    formats = {
        'jpg': 'JPEG',
        'tif': 'TIFF'
    }
    newarray.save(output, format=formats.get(ext, ext.upper()))


def _img_groups(total, nframe, frange):
    img_groups = [[]]

    for index in range(total):

        if index % nframe:
            continue

        img_groups[-1].append(index)
        if len(img_groups[-1]) == frange:
            img_groups.append(img_groups[-1][1:])

    return img_groups


def _ext(filepath):
    return os.path.splitext(filepath)[-1]


def _find_input_images(input_path):
    images = []

    for each in os.listdir(input_path):

        if each.startswith('.'):
            continue

        if _ext(each)[1:].upper() not in IMG_EXTENSIONS:
            continue

        fp = os.path.join(input_path, each)

        if not os.path.isfile(fp):
            continue

        images.append(fp)

    images.sort()
    return images


def _default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default=os.getcwd())
    parser.add_argument('-r', '--rendering', type=str, default='minimum')
    parser.add_argument('-o', '--output', type=str)
    return parser


def _stack_parser(parser):
    parser.add_argument('-n', '--nframe', type=int, default=NFRAME)
    return parser


def _animate_parser(parser):
    parser.add_argument('--framerate', type=int, default=FPS)
    parser.add_argument('--codec', type=str, default=VCODEC)
    parser.add_argument('--pix-fmt', type=str, default=PIX_FMT)
    parser.add_argument('--constant-rate', type=int, default=CONSTANT_RATE)
    parser.add_argument('-n', '--nframe', type=int, default=NFRAME)
    parser.add_argument('-f', '--frange', type=int, default=FRANGE)
    return parser


def _mode_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    args = parser.parse_args(sys.argv[1:2])

    if args.mode not in ('stack', 'animate'):
        error = "Unkown mode: '{}'\nValid modes: {}".format(
            args.mode, ', '.join(RENDERING)
        )
        raise RuntimeError(error)
    
    return args.mode


def _main():
    mode = _mode_parser()
    glo = globals()
    parser = glo['_{}_parser'.format(mode)](_default_parser())
    kwargs = vars(parser.parse_args(sys.argv[2:]))
    input_path = os.path.abspath(kwargs['input'])
    kwargs.pop('input')
    output_path = kwargs['output']
    kwargs.pop('output')
    images = _find_input_images(os.path.abspath(input_path))

    if output_path is None:
        ext = '.mp4' if mode == 'animate' else _ext(images[0])
        prefix = '{}-{}-'.format(os.path.join(os.path.dirname(input_path),
                                             os.path.basename(input_path)),
                                kwargs['rendering'])
        output_path = tempfile.mktemp(prefix=prefix, suffix=ext)

    glo[mode](images, output_path, **kwargs)


if __name__ == '__main__':
    _main()