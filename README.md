# gSLICrPy : Python3 Wrapper for "gSLICr: SLIC superpixels at over 250Hz"

This is the simple python3 wrapper for gSLICr.
For more information, please refer [the original work's repository](https://github.com/carlren/gSLICr).

### Licenses

I am totally not related with research group of the original work.
This repository is for personal research purposes.
Public release of this repository is based on the following items in the license specification of the original repository:

```
Isis’s permission is not required \
if the said reproduction, modification, transmission or transference \
is done without financial return, \
the conditions of this Licence are imposed upon the receiver of the product, \
and all original and amended source code is included in any transmitted product.
```
All license terms in this repository are governed by the original repository, and the author of this repository also waives all rights and liabilities for this work.


### Requirements
```
  - python3 (>= .5)
  - cmake (>= 2.8.10.2)
  - CUDA (>= 6.0)  
  - OpenCV (>= 3.0)
```
### Build Process
```
mkdir build
cd build
cmake ../
make
python3 example.py
```

### Docker

A `Dockerfile` is provided so you can build and run without installing CUDA,
OpenCV, or Python deps on the host. The image is based on
`nvidia/cuda:11.8.0-devel-ubuntu22.04` and installs everything the demo needs
(`numpy`, `scikit-image`, `opencv-python-headless`).

Running requires the host to have an NVIDIA GPU plus the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
(so `--gpus all` works).

Build:

```
docker build -t gslicrpy .
```

CUDA 11+ no longer supports `compute_30` (Kepler), so the Dockerfile rewrites
the arch in `CMakeLists.txt` to `compute_75` (Turing) by default. PTX is
forward-compatible, so the resulting binary still runs on Ampere/Ada/Hopper.
For best perf on a specific arch, override:

```
docker build --build-arg CUDA_ARCH=compute_86 -t gslicrpy .   # RTX 30xx (Ampere)
docker build --build-arg CUDA_ARCH=compute_89 -t gslicrpy .   # RTX 40xx (Ada)
docker build --build-arg CUDA_ARCH=compute_90 -t gslicrpy .   # H100 (Hopper)
```

Run the bundled example:

```
docker run --rm --gpus all gslicrpy
```

Mount your own working directory to consume your images and persist the label
map / segmented outputs back to the host:

```
docker run --rm --gpus all -v "$PWD":/work -w /work gslicrpy python3 example.py
```

### Example

![original](https://github.com/mikigom/SLICrPy/blob/master/example.jpg?raw=true)
![segmented](https://github.com/mikigom/SLICrPy/blob/master/seg_example_results.jpg?raw=true)

When n_iters=50, above example (2880 x 1800) is processed in 220ms on 1080Ti.

### Retrieving superpixels as contours

`CUDA_gSLICr` writes its result to disk. To work with each superpixel
individually (resolves [#1](https://github.com/mikigom/gSLICrPy/issues/1)),
use `CUDA_gSLICr_segment`, which returns the per-pixel label map as a NumPy
`int32` array of shape `(H, W)`:

```python
from gSLICrPy import CUDA_gSLICr_segment, __get_CUDA_gSLICr_segment__, labels_to_contours

labels = CUDA_gSLICr_segment(__get_CUDA_gSLICr_segment__(),
                             image_bgr_flat,
                             img_size_x=W, img_size_y=H,
                             n_segs=10, spixel_size=20,
                             coh_weight=0.6, n_iters=50,
                             segment_color_space=2,
                             segment_by_size=True,
                             enforce_connectivity=True)

# {label_id: [cv2 contour, ...]} — one entry per superpixel
contours_by_label = labels_to_contours(labels)
```

`labels_to_contours` requires `opencv-python`. If you only need binary masks
you can skip it and just do `mask = (labels == sp_id)`.


### Wrapper Writer

Junghoon Seo (sjh@satreci.com)
