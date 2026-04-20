import numpy as np
from skimage.io import imread

from gSLICrPy import (CUDA_gSLICr,
                      CUDA_gSLICr_segment,
                      __get_CUDA_gSLICr__,
                      __get_CUDA_gSLICr_segment__,
                      labels_to_contours)


def main():
    image = imread('./example.jpg')
    img_size_y, img_size_x = image.shape[0:2]
    image_bgr_flat = image[:, :, ::-1].flatten().astype('uint8')

    __CUDA_gSLICr__ = __get_CUDA_gSLICr__()
    CUDA_gSLICr(__CUDA_gSLICr__,
                image_bgr_flat,
                img_size_x=img_size_x,
                img_size_y=img_size_y,
                n_segs=10,
                spixel_size=20,
                coh_weight=0.6,
                n_iters=50,
                color_space=2,
                segment_color_space=2,
                segment_by_size=True,
                enforce_connectivity=True,
                out_name='example_results')

    __CUDA_gSLICr_segment__ = __get_CUDA_gSLICr_segment__()
    labels = CUDA_gSLICr_segment(__CUDA_gSLICr_segment__,
                                 image_bgr_flat,
                                 img_size_x=img_size_x,
                                 img_size_y=img_size_y,
                                 n_segs=10,
                                 spixel_size=20,
                                 coh_weight=0.6,
                                 n_iters=50,
                                 segment_color_space=2,
                                 segment_by_size=True,
                                 enforce_connectivity=True)

    print('\nlabel map shape:', labels.shape, 'dtype:', labels.dtype)
    print('superpixel count:', int(labels.max()) + 1)

    np.save('example_results_labels.npy', labels)

    try:
        contours_by_label = labels_to_contours(labels)
        sample = next(iter(contours_by_label.items()))
        print('contours retrieved for', len(contours_by_label),
              'superpixels (sample id', sample[0], 'has',
              len(sample[1]), 'contour blob(s))')
    except ImportError:
        print('opencv-python not installed; skipped contour extraction')


if __name__ == '__main__':
    main()
