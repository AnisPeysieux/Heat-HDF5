#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>

int main()
{

  char filename[] = "heat.h5";
  hid_t last, previous;
  

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  
  hid_t last_dset = H5Dopen (file_id, "/last", H5P_DEFAULT);
  hid_t last_dspace = H5Dget_space(last_dset);
  const int last_ndims = H5Sget_simple_extent_ndims(last_dspace);
  
  printf("last dims: %d\n", last_ndims);
  hsize_t* last_dims;
  last_dims = malloc(last_ndims * sizeof(hsize_t));
  H5Sget_simple_extent_dims(last_dspace, last_dims, NULL);
  for(size_t i = 0; i < last_ndims; ++i)
  {
    printf("\tdim %lu = %llu\n", i, last_dims[i]);
  }

	double(*last_buf)[last_dims[1]]  = malloc(sizeof(double)*last_dims[1]*last_dims[0]);
  H5Dread(last_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  last_buf);
//  for(size_t line = 0; line < last_dims[0]; ++line)
//  {
//    for(size_t col = 0; col < last_dims[1]; ++col)
//    {
//      printf("%f ", last_buf[line][col]);
//    }
//    printf("\n");
//  }
  H5Fclose(file_id);
  H5Dclose(last_dset);
  return 0;
//  H5Dread (last, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  &readBuf[0]);
}
