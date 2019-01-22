#ifndef HDF5UTILS_HEADER
#define HDF5UTILS_HEADER

#include <hdf5.h>
#include <mpi.h>

int new_step(int step, char* filename, MPI_Comm comm);
int new_file(char* filename);
int save_in_hdf5(char* filename, char* dataset_name, hsize_t dsize, hsize_t local_size, hsize_t offset, double* buf, MPI_Comm comm);

#endif
