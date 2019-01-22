#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <math.h>

#define STEPS_REALLOCATIONS_FACTOR 2

struct heatdatas_s
{
  int ndims;
  hsize_t local_dims[2];
  hsize_t global_dims[2];
  hsize_t offsets[2];
  int ranks_dims[2];
  MPI_Comm comm_dims[2];
  void* buf;
};
typedef struct heatdatas_s heatdatas_t;
typedef hsize_t hdsize_t;

int heatdatas_n_local_rows(heatdatas_t* heatdatas, hdsize_t* nrows)
{
  *nrows =  heatdatas -> local_dims[0];
  return 0;
}

int heatdatas_n_local_cols(heatdatas_t* heatdatas, hdsize_t* ncols)
{
  *ncols =  heatdatas -> local_dims[1];
  return 0;
}

int heatdatas_n_global_rows(heatdatas_t* heatdatas, hdsize_t* nrows)
{
  *nrows =  heatdatas -> global_dims[0];
  return 0;
}

int heatdatas_n_global_cols(heatdatas_t* heatdatas, hdsize_t* ncols)
{
  *ncols =  heatdatas -> global_dims[1];
  return 0;
}

int heatdatas_offset_rows(heatdatas_t* heatdatas, hdsize_t* offset_rows)
{
  *offset_rows =  heatdatas -> offsets[0];
  return 0;
}

int heatdatas_offset_cols(heatdatas_t* heatdatas, hdsize_t* offset_cols)
{
  *offset_cols =  heatdatas -> offsets[1];
  return 0;
}

int heatdatas_comm_rows(heatdatas_t* heatdatas, MPI_Comm* comm_rows)
{
  *comm_rows =  heatdatas -> comm_dims[0];
  return 0;
}

int heatdatas_comm_cols(heatdatas_t* heatdatas, MPI_Comm* comm_cols)
{
  *comm_cols =  heatdatas -> comm_dims[1];
  return 0;
}

int heatdatas_rank_row(heatdatas_t* heatdatas, int* rank_row)
{
  *rank_row =  heatdatas -> ranks_dims[0];
  return 0;
}

int heatdatas_rank_col(heatdatas_t* heatdatas, int* rank_col)
{
  *rank_col =  heatdatas -> ranks_dims[1];
  return 0;
}

struct steps_s
{
  int capacity;
  int size;
  int* steps;
};
typedef struct steps_s steps_t;

void init_steps(steps_t* steps)
{
  steps -> capacity = 0;
  steps -> size = 0;
  steps -> steps = NULL;
}

int steps_get(steps_t* steps, int n, int* out)
{
  *out = steps->steps[n];
  return 0;
}

void free_heatdatas(heatdatas_t* heatdatas)
{
  if(heatdatas -> local_dims != NULL)
  {  
    free(heatdatas -> local_dims);
  }
  if(heatdatas -> buf != NULL)
  {
    free(heatdatas -> buf);
  }
}

int save_in_hdf5(char* filename, char* dataset_name, hsize_t dsize, hsize_t local_size, hsize_t offset, double* buf, MPI_Comm comm)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(dataset_name != NULL);
  assert(buf != NULL);
 
  MPI_Info info;
  MPI_Info_create(&info);
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, comm, info);
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id);

 
  //Create dataspace
  hid_t dataspace;
  hid_t dataspace_mem;
  if(dsize == 1)
  {
    dataspace = H5Screate(H5S_SCALAR); if(dataspace < 0) { return -20; }
    dataspace_mem = H5Screate(H5S_SCALAR); if(dataspace_mem < 0) { return -21; }
  }
  else
  {
    dataspace = H5Screate_simple(1, &dsize, NULL); if(dataspace < 0) { return -2; }
    dataspace_mem = H5Screate_simple(1, &local_size, NULL); if(dataspace_mem < 0) { return -20; }
  }
  hid_t plist_id2 = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(plist_id2, H5FD_MPIO_COLLECTIVE);
 //Creating dataset
  //fprintf(stderr, "DATASET_NAME: %s\n", dataset_name);
  hid_t dataset = H5Dcreate(file_id, dataset_name, H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  //hid_t dataset = H5Dopen(file_id, dataset_name, H5P_DEFAULT) ;
  if(dataset < 0) { return -3; }
  if(dsize != 1)
    if(H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset, NULL, &local_size, NULL) < 0) { return -4; }

  //Writing in dataset
  if(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, dataspace_mem, dataspace, plist_id2, buf) < 0) { return -5; }

  //Closing ressources
  if(H5Sclose(dataspace) < 0) { return -6; }
  if(H5Dclose(dataset) < 0) { return -7; }
  H5Fclose(file_id);

  return 0;
}

int read_dataset_known_dims_hdf5(heatdatas_t* heatdatas, char* filename, char* dataset_name)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); if(file_id < 0) { return -1; }
  hid_t dset = H5Dopen (file_id, dataset_name, H5P_DEFAULT); if(dset < 0) { return -2; }
  hid_t dspace = H5Dget_space(dset); if(dspace < 0) { return -3; }
  if(H5Sselect_hyperslab(dspace, H5S_SELECT_SET, heatdatas->offsets, NULL, heatdatas->local_dims, NULL) < 0) { return -4; }
  //if(H5Sget_simple_extent_dims(dspace, heatdatas -> dims, NULL) < 0) { return -6; }
	heatdatas -> buf  = malloc(sizeof(double)*heatdatas->local_dims[1]*heatdatas->local_dims[0]); if(heatdatas -> buf == NULL) { return -7; }
  hid_t dataspace_mem = H5Screate_simple(2, heatdatas->local_dims, NULL); if(dataspace_mem < 0) { return -5; }
  if(H5Dread(dset, H5T_NATIVE_DOUBLE, dataspace_mem, dspace, H5P_DEFAULT, heatdatas -> buf) < 0) {return -8; }
  if(H5Dclose(dset) < 0) { return -9; }
  if(H5Fclose(file_id) < 0) { return -10; }

  return 0;
}

int sum_heatdatas(heatdatas_t* heatdatas, double* global_sum)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double (*buf) [heatdatas->local_dims[1]] = heatdatas->buf;
  double sum = 0;
  for(int row = 0; row < heatdatas->local_dims[0]; ++row)
  {
    for(int col = 0; col < heatdatas->local_dims[1]; ++col)
    {
      sum += buf[row][col];
    }
  }
  double gsum = 0;
  MPI_Reduce(&sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  *global_sum = *global_sum / (heatdatas->global_dims[0]*heatdatas->global_dims[1]);
  fprintf(stderr, "##rank %d, sum = %f, global_sum = %f\n", rank, sum, gsum);
  //for(int i = 0; i < heatdatas->local_dims[0]*heatdatas->local_dims[1]; ++i)
  //{
  //  fprintf(stderr, "%f ", ((double*)heatdatas->buf)[i]);
  //}
  return 0;
}

int sum_dim_heatdatas(heatdatas_t* heatdatas, double* global_sum, int dim)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int other_dim;
  if(dim == 0)
  {
    other_dim = 1;
  }
  else
  {
    other_dim = 0;
  }
  double (*buf) [heatdatas->local_dims[1]] = heatdatas->buf;
      double* local_sum = malloc(heatdatas->local_dims[other_dim] * sizeof(double));
      memset(local_sum, 0, heatdatas->local_dims[other_dim] * sizeof(double));
  for(int row = 0; row < heatdatas->local_dims[0]; ++row)
  {
    for(int col = 0; col < heatdatas->local_dims[1]; ++col)
    {
      if(dim == 0)
      {
        local_sum[col] += buf[row][col];
      }
      else
      {
        local_sum[row] += buf[row][col];
      }
    }
  }
    MPI_Reduce(local_sum, global_sum, heatdatas->local_dims[other_dim], MPI_DOUBLE, MPI_SUM, 0, heatdatas->comm_dims[dim]);
      //if(dim == 0 && heatdatas->ranks_dims[dim] == 0)
      if(heatdatas->ranks_dims[dim] == 0)
      {
        for(int i = 0; i < heatdatas->local_dims[other_dim]; ++i)
        {
          global_sum[i] /= heatdatas->global_dims[dim];
          //fprintf(stderr, "rank %d dim = %d divide => %f\n", rank, dim, global_sum[i]);
          //fprintf(stderr, "rank %d y_mean, %f\n", rank_comm_world, global_sum_rows[i]);
        }
      }

  return 0;
}

int read_dataset_dims_hdf5(heatdatas_t* heatdatas, char* filename, char* dataset_name)
{
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); if(file_id < 0) { return -1; }
  hid_t dset = H5Dopen (file_id, dataset_name, H5P_DEFAULT); if(dset < 0) { return -2; }
  hid_t dspace = H5Dget_space(dset); if(dspace < 0) { return -3; }
  heatdatas -> ndims = H5Sget_simple_extent_ndims(dspace); if(heatdatas -> ndims < 0) { return -4; }
  //heatdatas -> global_dims = malloc(heatdatas -> ndims * sizeof(hsize_t)); if(heatdatas -> local_dims == NULL) { return -5; }
  //heatdatas -> local_dims = malloc(heatdatas -> ndims * sizeof(hsize_t)); if(heatdatas -> local_dims == NULL) { return -5; }
  if(H5Sget_simple_extent_dims(dspace, heatdatas -> global_dims, NULL) < 0) { return -6; }
  //for(int i = 0; i < heatdatas -> ndims; ++i)
  for(int i = 0; i < 2; ++i)
  {
    heatdatas -> local_dims[i] = heatdatas -> global_dims[i];
  }
  if(H5Dclose(dset) < 0) { return -9; }
  if(H5Fclose(file_id) < 0) { return -10; }

  return 0;
}

int repart_datasets_hdf5(heatdatas_t* heatdatas) 
{

//  heatdatas -> ranks_dims = malloc(heatdatas -> ndims * sizeof(int));
  
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int comm_size; MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int nrows, ncols; 
	// number of processes in the x dimension
	nrows = sqrt(comm_size);
	if ( nrows<1 ) nrows=1; 
	// number of processes in the y dimension
	ncols = comm_size/nrows;
	// make sure the total number of processes is correct
	if (nrows*ncols != comm_size) {
        fprintf(stderr, "Error: invalid number of processes\n");
        abort();
   }

  //if(rank == 0) fprintf(stderr, "rank %d: nrows:ncols = %d:%d\n", rank, nrows, ncols);
  int row_rank = (rank) / (ncols);
  int col_rank = rank - row_rank * (ncols);
  int nrows_for_process = heatdatas->local_dims[0] / nrows;
  int first_row = nrows_for_process * row_rank;
  //fprintf(stderr, "1. rank %d, row_rank = %d, nrows_for_process = %d, nrows = %d, %lld %% %d = %lld\n", rank, row_rank, nrows_for_process, nrows, heatdatas->local_dims[0], nrows, heatdatas->local_dims[0] % nrows);
  if(row_rank < heatdatas->local_dims[0] % nrows)
  {
    nrows_for_process++;
  }
  //fprintf(stderr, "2. rank %d, nrows_for_process = %d, nrows = %d\n", rank, nrows_for_process, nrows);
  if(row_rank < heatdatas -> local_dims[0] % nrows)
  {
    first_row += row_rank;
  }
  else
  {
    first_row += heatdatas -> local_dims[0] % nrows;
  }
  int ncols_for_process = heatdatas->local_dims[1] / ncols;
  int first_col = ncols_for_process * col_rank;
  if(col_rank < heatdatas->local_dims[1] % ncols)
  {
    ncols_for_process++;
  }
  if(col_rank < heatdatas -> local_dims[1] % ncols)
  {
    first_col += col_rank;
  }
  else
  {
    first_col += heatdatas -> local_dims[1] % ncols;
  }

//  fprintf(stderr, "rank %d: row_rank = %d col_rank = %d\n\t rows = %d-%d(%d) first_col = %d-%d(%d)\n\t heatdatas = %p\n", rank, row_rank, col_rank, first_row, first_row + nrows_for_process - 1, nrows_for_process, first_col, first_col + ncols_for_process - 1, ncols_for_process, &(heatdatas->dims));
//  heatdatas->offsets = malloc(2 * sizeof(hsize_t));
 // heatdatas->comm_dims = malloc(2 * sizeof(MPI_Comm));
  heatdatas->offsets[0] = first_row;
  heatdatas->offsets[1] = first_col;
  //fprintf(stderr, "3. rank %d, nrows_for_process = %d, nrows = %d\n", rank, nrows_for_process, nrows);
  heatdatas->local_dims[0] = nrows_for_process;
  heatdatas->local_dims[1] = ncols_for_process;

  MPI_Comm_split(MPI_COMM_WORLD, col_rank, 0, &(heatdatas->comm_dims[0]));
  MPI_Comm_split(MPI_COMM_WORLD, row_rank, 0, &(heatdatas->comm_dims[1]));
  MPI_Comm_rank(heatdatas->comm_dims[0], &(heatdatas->ranks_dims[0]));
  MPI_Comm_rank(heatdatas->comm_dims[1], &(heatdatas->ranks_dims[1]));

  return 0; 
}

herr_t step_iteration(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)
{
  assert(op_data != NULL);
  steps_t* steps = op_data;
  if(steps -> capacity == 0)
  {
    steps -> steps = malloc(sizeof(int));
    steps -> size = 0;
    steps -> capacity = 1;
  }
  else if(steps -> size == steps -> capacity)
  {
    steps -> steps = realloc(steps -> steps, steps -> capacity * STEPS_REALLOCATIONS_FACTOR * sizeof(int));
  }
  sscanf(name, "%d/last", steps -> steps + steps -> size);
  steps -> size++;
  return 0;
}



int get_steps(char* filename, steps_t* steps)
{

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); if(file_id < 0) { return -1; }
  hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT); if(group_id < 0) { return -2; }
  //H5Lvisit(group_id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, print_groups, NULL);
  if(H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, step_iteration, steps) < 0) { return -3; }
  return 0;
}

int string_length_int(int n)
{
  if(n == 0)
  {
    return 0;
  }
  if(n < 0)
    n = -n;

  return log10(n) + 1;
}

int new_step(int step, char* filename, MPI_Comm comm)
{
  int str_length_step = string_length_int(step);
  int length_step_path = 1 + str_length_step + 1;
  char* step_path = malloc(length_step_path * sizeof(char));
  snprintf(step_path, length_step_path, "/%d", step);
  //fprintf(stderr, "STEP_PATH: %s\n", step_path);

  MPI_Info info;
  MPI_Info_create(&info);
  hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(plist_id, comm, info);
  hid_t file_id;
  file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id);
  hid_t group = H5Gcreate (file_id, step_path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Gclose(group);
  H5Fclose(file_id);
  return 0;

}

int new_file(char* filename)
{
          hid_t file_id;
          MPI_Info info;
          MPI_Info_create(&info);
          hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
          H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, info);
          file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
          if(file_id < 0)
          {
            fprintf(stderr, "Erreur creation fichier\n");
            return -1;
          }
          H5Fclose(file_id);
          return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  char input_filename[] = "heat.h5";
  char output_filename[] = "diags.h5";
  int rank_comm_world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_comm_world);
  {
    
    steps_t steps;
    init_steps(&steps);
    int ret_get_steps = get_steps(input_filename, &steps);
    if(ret_get_steps != 0)
    {
      fprintf(stderr, "Error in get_steps: return %d instead of 0\n", ret_get_steps);
      exit(EXIT_FAILURE);
    }
    
    new_file(output_filename);

    for(int i = 0; i < steps.size; ++i)
    {
    
      int str_length_step = string_length_int(steps.steps[i]);
      int length_step_path = 1 + str_length_step + 1;
      char* step_path = malloc(length_step_path * sizeof(char));
      snprintf(step_path, length_step_path, "/%d", steps.steps[i]);

      int length_last_path = 1 + str_length_step + 1 + strlen("last") + 1;
      char* last_path = malloc(length_last_path * sizeof(char));
      snprintf(last_path, length_last_path, "/%d/%s", steps.steps[i], "last");
      
      int length_x_mean_path = 1 + str_length_step + 1 + strlen("x_mean") + 1;
      char* x_mean_path = malloc(length_x_mean_path * sizeof(char));
      snprintf(x_mean_path, length_x_mean_path, "/%d/%s", steps.steps[i], "x_mean");
      
      int length_y_mean_path = 1 + str_length_step + 1 + strlen("y_mean") + 1;
      char* y_mean_path = malloc(length_y_mean_path * sizeof(char));
      snprintf(y_mean_path, length_y_mean_path, "/%d/%s", steps.steps[i], "y_mean");
      
      int length_mean_path = 1 + str_length_step + 1 + strlen("mean") + 1;
      char* mean_path = malloc(length_mean_path * sizeof(char));
      snprintf(mean_path, length_mean_path, "/%d/%s", steps.steps[i], "mean");

      heatdatas_t heatdatas;
      int err_read = read_dataset_dims_hdf5(&heatdatas, input_filename, last_path);
      if(err_read != 0)
      {
        fprintf(stderr, "Error in read_dataset_dims_hdf5: return %d instead of 0\n", err_read);
        exit(EXIT_FAILURE);
      }
      repart_datasets_hdf5(&heatdatas);
      err_read = read_dataset_known_dims_hdf5(&heatdatas, input_filename, last_path);
      if(err_read != 0)
      {
        fprintf(stderr, "Error in read_dataset_known_dims_hdf5: return %d instead of 0\n", err_read);
        exit(EXIT_FAILURE);
      }

      int rank_row;
      heatdatas_rank_row(&heatdatas, &rank_row);
      int rank_col;
      heatdatas_rank_col(&heatdatas, &rank_col);
      hdsize_t n_local_rows;
      heatdatas_n_local_rows(&heatdatas, &n_local_rows);
      hdsize_t n_local_cols;
      heatdatas_n_local_cols(&heatdatas, &n_local_cols);
      hdsize_t n_global_rows;
      heatdatas_n_global_rows(&heatdatas, &n_global_rows);
      hdsize_t n_global_cols;
      heatdatas_n_global_cols(&heatdatas, &n_global_cols);
      hdsize_t offset_rows;
      heatdatas_offset_rows(&heatdatas, &offset_rows);
      hdsize_t offset_cols;
      heatdatas_offset_cols(&heatdatas, &offset_cols);
      MPI_Comm comm_rows;
      heatdatas_comm_rows(&heatdatas, &comm_rows);
      MPI_Comm comm_cols;
      heatdatas_comm_cols(&heatdatas, &comm_cols);

      double global_sum = 0;
      double* global_sum_rows = malloc(n_local_cols * sizeof(double));
      double* global_sum_cols = malloc(n_local_rows * sizeof(double));

      sum_heatdatas(&heatdatas, &global_sum);
      sum_dim_heatdatas(&heatdatas, global_sum_rows, 0);
      sum_dim_heatdatas(&heatdatas, global_sum_cols, 1);

      int step;
      steps_get(&steps, i, &step);
      new_step(step, output_filename, MPI_COMM_WORLD);

      if(rank_row == 0)
      {
      int ret_save_1 = save_in_hdf5(output_filename, y_mean_path, n_global_cols, n_local_cols, offset_cols, global_sum_rows, comm_cols);
      if(ret_save_1 != 0)
      {
        fprintf(stderr, "Error in save_in_hdf5: return %d instead of 0\n", ret_save_1);
        exit(EXIT_FAILURE);
      }
      }

      if(rank_col == 0)
      {
      int ret_save_1 = save_in_hdf5(output_filename, x_mean_path, n_global_rows, n_local_rows, offset_rows, global_sum_cols, comm_rows);
      if(ret_save_1 != 0)
      {
        fprintf(stderr, "Error in save_in_hdf5: return %d instead of 0\n", ret_save_1);
        exit(EXIT_FAILURE);
      }
      }

      if(rank_comm_world == 0)
      {
      int ret_save_1 = save_in_hdf5(output_filename, mean_path, 1, 1, 0, &global_sum, MPI_COMM_SELF);
      if(ret_save_1 != 0)
      {
        fprintf(stderr, "Error in save_in_hdf5: return %d instead of 0\n", ret_save_1);
        exit(EXIT_FAILURE);
      }
      }

    }
  }
  MPI_Finalize();
  return 0;
}
