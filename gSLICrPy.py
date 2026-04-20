import ctypes
from ctypes import POINTER

import numpy as np


def __get_CUDA_gSLICr__(path_to_shared='./build/libDEMO.so'):
    """
    :return: Callable
    """
    dll = ctypes.CDLL(path_to_shared, mode=ctypes.RTLD_GLOBAL)
    func = dll.CUDA_gSLICr
    """
    void CUDA_gSLICr(unsigned char* image,
                     int img_size_x,
                     int img_size_y,
                     int n_segs,
                     int spixel_size,
                     float coh_weight,
                     int n_iters,
                     int color_space,
                     int segment_color_space,
                     bool segment_by_size,
                     bool enforce_connectivity,
                     char* out_name)
    """
    func.argtypes = [POINTER(ctypes.c_uint8),
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_float,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_bool,
                     ctypes.c_bool,
                     ctypes.c_char_p]
    func.restype = None

    return func


def __get_CUDA_gSLICr_segment__(path_to_shared='./build/libDEMO.so'):
    """
    Resolve the contour-friendly entry point that fills a caller-provided
    int32 label buffer. Use this instead of __get_CUDA_gSLICr__ when you want
    the labels in memory rather than written to disk.
    """
    dll = ctypes.CDLL(path_to_shared, mode=ctypes.RTLD_GLOBAL)
    func = dll.CUDA_gSLICr_segment
    """
    void CUDA_gSLICr_segment(unsigned char* image,
                             int img_size_x,
                             int img_size_y,
                             int n_segs,
                             int spixel_size,
                             float coh_weight,
                             int n_iters,
                             int segment_color_space,
                             bool segment_by_size,
                             bool enforce_connectivity,
                             int* out_labels)
    """
    func.argtypes = [POINTER(ctypes.c_uint8),
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_float,
                     ctypes.c_int,
                     ctypes.c_int,
                     ctypes.c_bool,
                     ctypes.c_bool,
                     POINTER(ctypes.c_int32)]
    func.restype = None

    return func


def CUDA_gSLICr(__get_CUDA_gSLICr__,
                image,
                img_size_x,
                img_size_y,
                n_segs,
                spixel_size,
                coh_weight,
                n_iters,
                color_space,
                segment_color_space,
                segment_by_size,
                enforce_connectivity,
                out_name):
    """

    :param __get_CUDA_gSLICrm__:
    :param image:
    :param img_size_x:
    :param img_size_y:
    :param n_segs:
    :param spixel_size:
    :param coh_weight:
    :param n_iters:
    :param color_space:
    :param segment_color_space:
    :param segment_by_size:
    :param enforce_connectivity:
    :param out_name:
    :return:
    """
    image = image.ctypes.data_as(POINTER(ctypes.c_uint8))
    out_name = out_name.encode('utf-8')

    return __get_CUDA_gSLICr__(image,
                               img_size_x,
                               img_size_y,
                               n_segs,
                               spixel_size,
                               coh_weight,
                               n_iters,
                               color_space,
                               segment_color_space,
                               segment_by_size,
                               enforce_connectivity,
                               out_name)


def CUDA_gSLICr_segment(__CUDA_gSLICr_segment__,
                        image,
                        img_size_x,
                        img_size_y,
                        n_segs,
                        spixel_size,
                        coh_weight,
                        n_iters,
                        segment_color_space,
                        segment_by_size,
                        enforce_connectivity):
    """
    Run gSLICr and return the per-pixel superpixel label map.

    :param image: flattened uint8 image (BGR, length = H*W*3)
    :return: numpy.ndarray of shape (img_size_y, img_size_x), dtype int32,
             where each entry is the integer id of the superpixel that owns
             that pixel. Convert to per-superpixel binary masks or contours
             with e.g. ``labels_to_contours`` below.
    """
    out_labels = np.zeros((img_size_y, img_size_x), dtype=np.int32)

    image_ptr = image.ctypes.data_as(POINTER(ctypes.c_uint8))
    labels_ptr = out_labels.ctypes.data_as(POINTER(ctypes.c_int32))

    __CUDA_gSLICr_segment__(image_ptr,
                            img_size_x,
                            img_size_y,
                            n_segs,
                            spixel_size,
                            coh_weight,
                            n_iters,
                            segment_color_space,
                            segment_by_size,
                            enforce_connectivity,
                            labels_ptr)

    return out_labels


def labels_to_contours(labels):
    """
    Convert a label map produced by ``CUDA_gSLICr_segment`` into a dict of
    OpenCV contours, keyed by superpixel id. Each value is a list of contours
    (the same structure ``cv2.findContours`` returns), so a single superpixel
    that gets split into disconnected blobs by ``enforce_connectivity=False``
    still yields one entry per blob.

    Requires ``opencv-python``. The label map dtype must be int32.
    """
    import cv2

    contours_by_label = {}
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours_by_label[int(lab)] = list(contours)

    return contours_by_label
