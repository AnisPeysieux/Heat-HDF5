#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <string.h>
#include <assert.h>

struct loaded_s
{
  int ndims;
  hsize_t* dims;
  void* buf;
};
typedef struct loaded_s loaded_t;

void free_loaded(loaded_t* load)
{
  if(load -> dims != NULL)
  {  
    free(load -> dims);
  }
  if(load -> buf != NULL)
  {
    free(load -> buf);
  }
}

int save_in_hdf5(hid_t file_id, char* dataset_name, hsize_t dsize,double* buf)
{
  assert(dataset_name != NULL);
  assert(buf != NULL);
  
  //Create dataspace
  hid_t dataspace;
  if(dsize == 1)
  {
    dataspace = H5Screate(H5S_SCALAR); if(dataspace < 0) { return -2; }
  }
  else
  {
    dataspace = H5Screate_simple(1, &dsize, NULL); if(dataspace < 0) { return -2; }
  }

  //Creating dataset
  hid_t dataset = H5Dcreate(file_id, dataset_name, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if(dataset < 0) { return -3; }

  //Writing in dataset
  if(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, dataspace, dataspace, H5P_DEFAULT, buf) < 0) { return -4; }

  //Closing ressources
  if(H5Sclose(dataspace) < 0) { return -5; }
  if(H5Dclose(dataset) < 0) { return -6; }

  return 0;
}

int read_dataset_hdf5(char* filename, char* dataset_name, loaded_t* load)
{
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); if(file_id < 0) { return -1; }
  hid_t dset = H5Dopen (file_id, dataset_name, H5P_DEFAULT); if(dset < 0) { return -2; }
  hid_t dspace = H5Dget_space(dset); if(dspace < 0) { return -3; }
  load -> ndims = H5Sget_simple_extent_ndims(dspace); if(load -> ndims < 0) { return -4; }
  load -> dims = malloc(load -> ndims * sizeof(hsize_t)); if(load -> dims == NULL) { return -5; }
  if(H5Sget_simple_extent_dims(dspace, load -> dims, NULL) < 0) { return -6; }
	load -> buf  = malloc(sizeof(double)*load->dims[1]*load->dims[0]); if(load -> buf == NULL) { return -7; }
  if(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, load -> buf) < 0) {return -8; }
  if(H5Dclose(dset) < 0) { return -9; }
  if(H5Fclose(file_id) < 0) { return -10; }

  return 0;
}

int main()
{

  char input_filename[] = "heat.h5";
  char output_filename[] = "diags.h5";

  loaded_t load;
  int err_read = read_dataset_hdf5(input_filename, "/last", &load);
  if(err_read != 0)
  {
    fprintf(stderr, "Error in read_dataset_hdf5: return %d instead of 0\n", err_read);
    exit(EXIT_FAILURE);
  }
  
  if(load.ndims != 2) exit(EXIT_FAILURE);
  
  double(*buf)[load.dims[load.ndims - 1]] = load.buf;

  double mean = 0;
  double* x_mean = malloc(load.dims[0] * sizeof(double));
  memset(x_mean, 0, load.dims[0] * sizeof(double));
  double* y_mean = malloc(load.dims[1] * sizeof(double));
  memset(y_mean, 0, load.dims[1] * sizeof(double));

  for(size_t row = 0; row < load.dims[0]; ++row)
  {
    
    for(size_t col = 0; col < load.dims[1]; ++col)
    {
      x_mean[row] += buf[row][col];
      y_mean[col] += buf[row][col];
      mean += buf[row][col];
    }
    x_mean[row] /= load.dims[1];
  }
  for(size_t col = 0; col < load.dims[1]; ++col)
  {
    y_mean[col] /= load.dims[0];
  }
  mean /= load.dims[0] * load.dims[1];

  hid_t file_id = H5Fcreate(output_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) { return -1; }
  int ret_save_1 = save_in_hdf5 (file_id, "/x_mean", load.dims[0], x_mean);
  if(ret_save_1 != 0)
  {
    fprintf(stderr, "Error in save_in_hdf5: return %d instead of 0\n", ret_save_1);
    exit(EXIT_FAILURE);
  }
  int ret_save_2 = save_in_hdf5 (file_id, "/y_mean", load.dims[1], y_mean);
  if(ret_save_2 != 0)
  {
    fprintf(stderr, "Error in save_in_hdf5: return %d instead of 0\n", ret_save_2);
    exit(EXIT_FAILURE);
  }
  int ret_save_3 = save_in_hdf5 (file_id, "/mean", 1, &mean);
  if(ret_save_3 != 0)
  {
    fprintf(stderr, "Error in save_in_hdf5: return %d instead of 0\n", ret_save_3);
    exit(EXIT_FAILURE);
  }

  free_loaded(&load);
  free(x_mean);
  free(y_mean);
  if(H5Fclose(file_id) < 0)
  {
    fprintf(stderr, "Error when closing output\n");
  }

  return 0;
}
