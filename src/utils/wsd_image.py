from typing import List, Tuple

import shutil

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from wholeslidedata.interoperability.asap.imagewriter import WholeSlideMaskWriter, WholeSlideImageWriter, \
    WholeSlideImageWriterBase
from wholeslidedata.image.spacings import take_closest_level
from wholeslidedata.image.wholeslideimage import WholeSlideImage


def path_exists(path):
    return Path(path).exists()


def ensure_dir_exists(path):
    if not path_exists(path):
        path = Path(path)
        return Path(path).mkdir(parents=True, exist_ok=True)
    return True


def mkdir(path):
    return ensure_dir_exists(path)


def write_array_with_writer(array, writer, tile_size=512, close=True):
    if array.dtype==np.float16:
        array = array.astype(np.float32)

    shape = array.shape
    if len(shape)==2:
        n_channels = 1
    else:
        n_channels = shape[-1]

    for row in range(0, shape[0]+tile_size, tile_size): #+tile_size if array not divisible by tile_size
        if row >= shape[0]: continue
        for col in range(0, shape[1]+tile_size, tile_size):
            if col >= shape[1]: continue
            tile = array[row:row+tile_size, col:col+tile_size]
            wtile = np.zeros((tile_size, tile_size, n_channels), dtype=array.dtype)
            # wtile = wtile.squeeze()
            if len(wtile.shape)!=len(tile.shape):
                wtile = wtile.squeeze()
                tile = tile.squeeze()
            wtile[:tile.shape[0],:tile.shape[1]] = tile.copy()
            # print('writing tile from row', row, 'col', col, tile.shape, 'wtile', wtile.shape)
            writer.write(tile=wtile, row=row, col=col)
            # print('done writing tile')
    if close:
        writer.close()


def take_closest_smallest_numer(l, number):
    l = sorted(l)
    l = l+[l[-1]] #so that the loop doesnt have to handle the last number in list
    for i,val in enumerate(l):
        if val > number:
            if i==0:
                raise ValueError('no value in %s is smaller then %s' % (str(l), str(number)))
            break
    return l[i-1]


class PixelSpacingLevelError(Exception):
    """Raise when there is no level for the spacing within the tolerance."""
    def __init__(self, path, spacing, tolerance, *args, **kwargs):
        super().__init__('no level for spacing %.3f with tolerance %.2f, path: %s, %s' % \
                         (spacing, tolerance, str(path), str(kwargs)))

class ImageReader(WholeSlideImage):
    def __init__(self, path, backend='asap', cache_path=None, verbose=False, spacing_tolerance=0.3):
        if cache_path is not None and str(cache_path)!=str(path) and Path(cache_path)!=Path(path).parent:
            path = Path(path)
            cache_path = Path(cache_path)
            if cache_path.suffix!=path.suffix:
                mkdir(cache_path)
                cache_path = cache_path/path.name
            if not cache_path.exists():
                shutil.copyfile(src=str(path), dst=str(cache_path))
            cache_path = str(cache_path)
            path = cache_path
        super().__init__(str(path), backend=backend)
        # self._calc_spacing_ranges()
        self.cache_path = cache_path
        self.verbose = verbose
        self.spacing_tolerance = spacing_tolerance
        if self.verbose:
            print('initialized reader for %s, backend=%s' % (self.path, str(self._backend)))
            print('%d channels, shapes %s, downsamplings %s' % (self.channels, str(self.shapes), str(self.downsamplings)))

    @property
    def channels(self):
        return self._backend.getSamplesPerPixel()

    @property
    def path(self):
        return self._path

    @property
    def shapes(self) -> List[Tuple[int, int]]:
        """ (h,w) """
        shapes_ = super().shapes
        #convert from (w,h) to (h,w)
        shapes = [(w,h) for (h,w) in shapes_]
        return shapes

    def content(self, spacing):
        content = self.get_slide(spacing)
        # content = np.swapaxes(content, 0, 1)
        # content = self._mask_convert(content)
        return content

    def shape(self, spacing):
        level = self.get_level_from_spacing(spacing)
        shape = self.shapes[level]
        return shape

    def get_slide(self, spacing):
        shape = self.shape(spacing)
        if self.verbose:
            print('reading at spacing %.2f (%d,%d)' % (spacing, shape[0], shape[1]))
        content = self.read(spacing, 0, 0, shape[0], shape[1])
        if self.verbose:
            print('read slide with shape %s' % str(content.shape))
        return content

    def _mask_convert(self, img):
        if 'openslide' in str(self._backend.name).lower() \
                and (img[:,:,0]==img[:,:,1]).all() and (img[:,:,0]==img[:,:,2]).all():
            if self.verbose:
                print('openslide _mask_convert')
            img = img[:,:,0] #openslide returns wxhx3 for masks, asap wxh.1
            img = img[:,:,None]
        return img

    def read(self, spacing, row, col, height, width):
        patch = self.get_patch(col, row, width, height, spacing, center=False, relative=True)
        # patch = self._mask_convert(patch)
        # return patch.transpose([1,0,2])
        return patch

    def refine(self, spacing):
        self.level(spacing) #check for missing spacing
        return self.get_real_spacing(spacing)

    def level(self, spacing: float) -> int:
        level = take_closest_level(self._spacings, spacing)
        spacing_margin = spacing * self.spacing_tolerance

        if abs(self.spacings[level] - spacing) > spacing_margin:
            raise PixelSpacingLevelError(self.path, spacing, self.SPACING_MARGIN, spacings=self.spacings)

        return level

    def close(self, clear=True):
        if self.cache_path is not None and Path(self.cache_path).exists():
            if self.verbose: print('deleting cached image %s' % self.cache_path)
            Path(self.cache_path).unlink()
            # Path(self.cache_path).unlink(missing_ok=True) only python 3.8
        super().close()

    def closest_spacing(self, spacing):
        try:
            closest = self.refine(spacing)
            return closest
        except PixelSpacingLevelError as ex:
            pos = take_closest_level(self.spacings, spacing)
            return self.spacings[pos]

class ArrayImageWriter(object):
    def __init__(self, cache_dir=None, tile_size=512, suppress_mir_stdout=True, skip_empty=False, jpeg_quality=80):
        self.cache_dir = cache_dir
        self.tile_size = tile_size
        self.tile_shape = (tile_size, tile_size)
        self.suppress_mir_stdout = suppress_mir_stdout
        self.skip_empty = skip_empty
        self.jpeg_quality = jpeg_quality

    def write_array(self, arr, path, spacing, verbose=False):
        tile_size = self.tile_size
        tile_shape = self.tile_shape
        if min(arr.shape[:2]) < tile_size:
            tile_size = take_closest_smallest_numer([8, 16, 32, 64, 128, 256], min(arr.shape[:2]))
            tile_shape = (tile_size, tile_size)
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)

        jpeg_quality = None
        # f = f.transpose(1, 2, 0)
        kwargs = {}
        if 'int' in str(arr.dtype):
            if len(arr.shape)==3 and arr.shape[-1]==3:
                writer = WholeSlideImageWriter()
                kwargs['jpeg_quality'] = self.jpeg_quality
                kwargs['interpolation'] = 'linear'
            else:
                writer = WholeSlideMaskWriter()
        else:
            writer = WholeSlideMaskWriter()

        if len(arr.shape) == 2:
            arr = arr[:, :, None]

        mkdir(Path(path).parent)
        writer.write(path, dimensions=arr.shape[:2][::-1], spacing=spacing, tile_shape=tile_shape, **kwargs)

        diaglike = WsdWriterWrapper(writer)

        write_array_with_writer(arr, diaglike, tile_size=tile_size)

class WsdWriterWrapper(object):
    def __init__(self, writer:WholeSlideImageWriterBase):
        self.writer = writer

    def write(self, tile, row, col):
        self.writer.write_tile(tile, (col, row))

    def close(self):
        self.writer.finishImage()

def write_array(array, path, out_spacing, cache_dir=None):
    writer = ArrayImageWriter(cache_dir)
    writer.write_array(array, path, out_spacing)

