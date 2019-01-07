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

### Example

![original](https://github.com/mikigom/SLICrPy/blob/master/example.jpg?raw=true)
![segmented](https://github.com/mikigom/SLICrPy/blob/master/seg_example_results.jpg?raw=true)

When n_iters=50, above example (2880 x 1800) is processed in 220ms on 1080Ti.


### Wrapper Writer

Junghoon Seo (sjh@satreci.com)
