#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <string.h>
#include <assert.h>

int save_in_hdf5(hid_t file_id, char* filename, char* dataset_name, hsize_t dsize,double* buf)
{
  assert(filename != NULL);
  assert(dataset_name != NULL);
  assert(buf != NULL);
  
  //Create file

  //Create dataspaces
  hid_t dataspace = H5Screate_simple(1, &dsize, NULL); if(dataspace < 0) { return -2; }

  //Creating datasets
  hid_t dataset = H5Dcreate(file_id, dataset_name, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset < 0) { return -3; }

  //Writing in datasets
  if(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, dataspace, dataspace, H5P_DEFAULT, buf) < 0) { return -4; }

  //Closing ressources
  if(H5Sclose(dataspace) < 0) { return -5; }
  if(H5Dclose(dataset) < 0) { return -6; }
  //if(H5Fclose(file_id) < 0) { return -1; }

  return 0;
}

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
//    printf("\tdim %lu = %llu\n", i, last_dims[i]);
  }

	double(*last_buf)[last_dims[1]]  = malloc(sizeof(double)*last_dims[1]*last_dims[0]);
  H5Dread(last_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  last_buf);
//  for(size_t row = 0; row < last_dims[0]; ++row)
//  {
//    for(size_t col = 0; col < last_dims[1]; ++col)
//    {
//      printf("%f ", last_buf[row][col]);
//    }
//    printf("\n");
//  }

  double last_mean = 0;
  double* last_mean_rows = malloc(last_dims[0] * sizeof(double));
  memset(last_mean_rows, 0, last_dims[0] * sizeof(double));
  double* last_mean_cols = malloc(last_dims[1] * sizeof(double));
  memset(last_mean_cols, 0, last_dims[1] * sizeof(double));
  for(size_t row = 0; row < last_dims[0]; ++row)
  {
    
    for(size_t col = 0; col < last_dims[1]; ++col)
    {
      last_mean_rows[row] += last_buf[row][col];
      last_mean_cols[col] += last_buf[row][col];
      last_mean += last_buf[row][col];
    }
  }
  for(size_t row = 0; row < last_dims[0]; ++row)
  {
    last_mean_rows[row] /= last_dims[1];
  }
  for(size_t col = 0; col < last_dims[1]; ++col)
  {
    last_mean_cols[col] /= last_dims[0];
  }
  last_mean /= last_dims[0] * last_dims[1];

  printf("mean row:\n");
  for(size_t row = 0; row < last_dims[0]; ++row)
  {
    printf("%f ", last_mean_rows[row]);
  }
  printf("\n");
  printf("mean col:\n");
  for(size_t col = 0; col < last_dims[1]; ++col)
  {
    printf("%f ", last_mean_cols[col]);
  }
  printf("\n");
  printf("mean: %f\n", last_mean);


  hid_t file_id_2 = H5Fcreate("diag.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) { return -1; }
  int ret_save_1 = save_in_hdf5 (file_id_2, "diag.h5", "/x_mean", last_dims[0], last_mean_rows);
  int ret_save_2 = save_in_hdf5 (file_id_2, "diag.h5", "/y_mean", last_dims[1], last_mean_cols);
  int ret_save_3 = save_in_hdf5 (file_id_2, "diag.h5", "/mean", 1, &last_mean);
  printf("%d\n", ret_save_1);
  printf("%d\n", ret_save_2);
  printf("%d\n", ret_save_3);

  
  free(last_mean_rows);
  free(last_mean_cols);
  free(last_buf);
  H5Dclose(last_dset);
  H5Fclose(file_id);
  H5Fclose(file_id_2);
  return 0;
//  H5Dread (last, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  &readBuf[0]);
}
