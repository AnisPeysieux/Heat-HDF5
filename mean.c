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

int save_in_hdf5(hid_t file_id, char* filename, char* dataset_name, hsize_t dsize,double* buf)
{
  assert(filename != NULL);
  assert(dataset_name != NULL);
  assert(buf != NULL);
  
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

int read_dataset_hdf5(char* filename, char* dataset_name, loaded_t* load)
{
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t dset = H5Dopen (file_id, dataset_name, H5P_DEFAULT);
  hid_t dspace = H5Dget_space(dset);
  load -> ndims = H5Sget_simple_extent_ndims(dspace);
  load -> dims = malloc(load -> ndims * sizeof(hsize_t));
  H5Sget_simple_extent_dims(dspace, load -> dims, NULL);
	load -> buf  = malloc(sizeof(double)*load->dims[1]*load->dims[0]);
  H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, load -> buf);
  H5Dclose(dset);
  H5Fclose(file_id);
}

int main()
{

  char input_filename[] = "heat.h5";
  char output_filename[] = "diag.h5";

  loaded_t load;
  read_dataset_hdf5(input_filename, "/last", &load);
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

  printf("x_mean:\n");
  for(size_t row = 0; row < load.dims[0]; ++row)
  {
    printf("%f ", x_mean[row]);
  }
  printf("\n");
  printf("y_mean:\n");
  for(size_t col = 0; col < load.dims[1]; ++col)
  {
    printf("%f ", y_mean[col]);
  }
  printf("\n");
  printf("mean: %f\n", mean);


  hid_t file_id = H5Fcreate(input_filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if(file_id < 0) { return -1; }
  int ret_save_1 = save_in_hdf5 (file_id, output_filename, "/x_mean", load.dims[0], x_mean);
  int ret_save_2 = save_in_hdf5 (file_id, output_filename, "/y_mean", load.dims[1], y_mean);
  int ret_save_3 = save_in_hdf5 (file_id, output_filename, "/mean", 1, &mean);
  printf("%d\n", ret_save_1);
  printf("%d\n", ret_save_2);
  printf("%d\n", ret_save_3);

  free_loaded(&load);
  free(x_mean);
  free(y_mean);
  H5Fclose(file_id_2);

  return 0;
}
