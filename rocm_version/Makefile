HIP_PATH    ?=/opt/rocm
HIPCC       :=$(HIP_PATH)/bin/hipcc
INCLUDE     :=-I/home/summit/.local/cfitsio/include\
	      	-I/home/summit/.local/wcslib/include/wcslib\
			-I/usr/local/hdf5/include\
	      	-I./
LIBRARIES   :=-L/home/summit/.local/cfitsio/lib\
              -L/home/summit/.local/wcslib/lib\
			  -L/usr/local/hdf5/lib\

HIPCC_FLAGS := -fgpu-rdc -Wno-deprecated-register
HIPCC_LINK_FLAGS := -fgpu-rdc --hip-link
CXX_FLAGS := -std=c++11 
FITS_FLAGS := -lm -fopenmp -lcfitsio -lwcs -lhdf5 

HEGrid: helpers.o healpix.o gmap.o gridding.o main.o
	$(HIPCC) -O3 $(HIPCC_LINK_FLAGS) -o $@ $+ $(LIBRARIES) $(FITS_FLAGS)

helpers.o: helpers.hip
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

healpix.o: healpix.hip
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

gmap.o: gmap.cpp
	$(HIPCC) -O3 $(HIPCC_FLAGS) $(INCLUDE) -o $@ -c $<

gridding.o: gridding-omp.hip
	$(HIPCC) -O3 -fopenmp $(HIPCC_FLAGS) $(INCLUDE) -o $@ -c $<

main.o: HEGrid.cpp
	$(HIPCC) -O3 $(HIPCC_FLAGS) -o $@ -c $<

clean:
	rm -rf *.o HEGrid
