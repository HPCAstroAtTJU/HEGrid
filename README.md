# HEGrid

HEGrid is a High Eﬀicient Multi-Channel Radio Astronomical Data Gridding Framework in Heterogeneous Computing Environments.

### Dependencies

- cfitsio-3.47 or later
- wcslib-5.16 or later
- HDF5
- boost library
- CUDA Toolkit
- ROCm Toolkit 4.0 or later

All of these packages can be found in "Dependencies" directory or get from follow address:

- cfitsio: https://heasarc.gsfc.nasa.gov/fitsio/
- wcslib: https://www.atnf.csiro.au/people/Mark.Calabretta/WCS/
- HDF5: https://www.hdfgroup.org/downloads/hdf5
- boost: https://www.boost.org/
- CUDA: https://developer.nvidia.com/cuda-toolkit-archive
- ROCm: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html#rocm-installation-guide-v5-0

## Build

```shell
git clone git@github.com:HWang-Summit/HEGrid.git
cd HEGrid
```
First, before compiling and installing HEGrid, please install the relevant dependencies. Second, update the dependencies paths of the Makefiles under "cuda_version" and "rocm_version". Then:

**1. Build HEGrid of CUDA Version:**

```shell
cd cuda_version 
make
```
**2. Build HEGrid of ROCm Version:**

```shell
cd rocm_version 
make
```

## Usage

Parameters:

```shell
--file_path   # absolute path of file, includ input,target,output
--input_file  # input file name
--target_file # target file name
--output_file # output file name
--file_id	  # file id of input and output, such as input100, output100, id is 100
--beam_size	  # beam size
--order_arg   # default 1
--block_num   # thread block size
```

Example:

```shell
./HCGrid --fits_path /my_file_path/ --input_file input --target_file target --output_file output --file_id 100 --beam_size 180 --order_arg 1 --block_num 352
```

***Note:*** One can use the "create_target_map.py" in cuda_version or rocm_version to create the target map based on the related parameters of the actual observations, such as beam size, map center and map size, etc.

## Community Contribution and Advice

All bug reports, comments and suggestions are welcome.

Feel free to open a new issue.

