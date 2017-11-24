import ctypes
from ctypes import POINTER

def __get_CUDA_gSLICr__(path_to_shared='./build/libDEMO.so'):
    """
    :return: Callable
    """
    dll = ctypes.CDLL(path_to_shared, mode=ctypes.RTLD_GLOBAL)
    func = dll.CUDA_gSLICr
    """
    int* CUDA_gSLICr(unsigned char* image,
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
    # POINTER(c_char) or ctypes.c_char_p ?

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
