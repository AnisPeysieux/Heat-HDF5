#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <heatdatas.h>
#include <steps.h>
#include <hdf5_utils.h>
#include <utils.h>

int sum_heatdatas(heatdatas_t* heatdatas, double* global_sum)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  hdsize_t n_local_cols;
  hdsize_t n_local_rows;
  hdsize_t n_global_cols;
  hdsize_t n_global_rows;
  heatdatas_n_local_cols(heatdatas, &n_local_cols);
  heatdatas_n_local_rows(heatdatas, &n_local_rows);
  heatdatas_n_global_cols(heatdatas, &n_global_cols);
  heatdatas_n_global_rows(heatdatas, &n_global_rows);
  void* void_buf;
  heatdatas_buf(heatdatas, &void_buf);
  double (*buf) [n_local_cols] = void_buf;
  
  double sum = 0;
  for(int row = 0; row < n_local_rows; ++row)
  {
    for(int col = 0; col < n_local_cols; ++col)
    {
      sum += buf[row][col];
    }
  }
  
  //double gsum = 0;
  MPI_Reduce(&sum, global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  *global_sum = *global_sum / (n_global_rows*n_global_cols);
  //fprintf(stderr, "##rank %d, sum = %f, global_sum = %f\n", rank, sum, gsum);
  return 0;
}

int sum_dim_heatdatas(heatdatas_t* heatdatas, double* global_sum, int dim)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  hdsize_t n_local_cols;
  hdsize_t n_local_rows;
  hdsize_t n_global_cols;
  hdsize_t n_global_rows;
  MPI_Comm comm;
  int rank_dim;
  
  heatdatas_n_local_cols(heatdatas, &n_local_cols);
  heatdatas_n_local_rows(heatdatas, &n_local_rows);
  heatdatas_n_global_cols(heatdatas, &n_global_cols);
  heatdatas_n_global_rows(heatdatas, &n_global_rows);
  void* void_buf;
  heatdatas_buf(heatdatas, &void_buf);
  double (*buf) [n_local_cols] = void_buf;

  hdsize_t n_local_other_dim;
  hdsize_t n_global_dim;
  if(dim == 0)
  {
    heatdatas_comm_rows(heatdatas, &comm);
    heatdatas_rank_row(heatdatas, &rank_dim);
    n_global_dim = n_global_rows;
    n_local_other_dim = n_local_cols;
  }
  else
  {
    heatdatas_comm_cols(heatdatas, &comm);
    heatdatas_rank_col(heatdatas, &rank_dim);
    n_global_dim = n_global_cols;
    n_local_other_dim = n_local_rows;
  }

  double* local_sum = malloc(n_local_other_dim * sizeof(double));
  memset(local_sum, 0, n_local_other_dim * sizeof(double));

  for(int row = 0; row < n_local_rows; ++row)
  {
    for(int col = 0; col < n_local_cols; ++col)
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
    MPI_Reduce(local_sum, global_sum, n_local_other_dim, MPI_DOUBLE, MPI_SUM, 0, comm);
      if(rank_dim == 0)
      {
        for(int i = 0; i < n_local_other_dim; ++i)
        {
          global_sum[i] /= n_global_dim;
          //fprintf(stderr, "rank %d dim = %d divide => %f\n", rank, dim, global_sum[i]);
          //fprintf(stderr, "rank %d y_mean, %f\n", rank_comm_world, global_sum_rows[i]);
        }
      }

  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  char input_filename[] = "heat.h5";
  char output_filename[] = "diags.h5";
  int rank_comm_world;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_comm_world);
  steps_t steps;
  steps_init(&steps);
  int ret_get_steps = steps_load(input_filename, &steps);
  if(ret_get_steps != 0)
  {
    fprintf(stderr, "Error in get_steps: return %d instead of 0\n", ret_get_steps);
    exit(EXIT_FAILURE);
  }
  
  new_file(output_filename);

  int n_steps;
  steps_size(&steps, &n_steps);
  for(int i = 0; i < n_steps; ++i)
  {
    int step;
    steps_get(&steps, i, &step);
  
    int str_length_step = string_length_int(step);
    int length_step_path = 1 + str_length_step + 1;
    char* step_path = malloc(length_step_path * sizeof(char));
    snprintf(step_path, length_step_path, "/%d", step);

    int length_last_path = 1 + str_length_step + 1 + strlen("last") + 1;
    char* last_path = malloc(length_last_path * sizeof(char));
    snprintf(last_path, length_last_path, "/%d/%s", step, "last");
    
    int length_x_mean_path = 1 + str_length_step + 1 + strlen("x_mean") + 1;
    char* x_mean_path = malloc(length_x_mean_path * sizeof(char));
    snprintf(x_mean_path, length_x_mean_path, "/%d/%s", step, "x_mean");
    
    int length_y_mean_path = 1 + str_length_step + 1 + strlen("y_mean") + 1;
    char* y_mean_path = malloc(length_y_mean_path * sizeof(char));
    snprintf(y_mean_path, length_y_mean_path, "/%d/%s", step, "y_mean");
    
    int length_mean_path = 1 + str_length_step + 1 + strlen("mean") + 1;
    char* mean_path = malloc(length_mean_path * sizeof(char));
    snprintf(mean_path, length_mean_path, "/%d/%s", step, "mean");

    heatdatas_t heatdatas;
    int err_read = heatdatas_load_dims(&heatdatas, input_filename, last_path);
    if(err_read != 0)
    {
      fprintf(stderr, "Error in read_dataset_dims_hdf5: return %d instead of 0\n", err_read);
      exit(EXIT_FAILURE);
    }
    heatdatas_distribute(&heatdatas);
    hdsize_t margins[4] = {0, 0, 0, 0};
    err_read = heatdatas_load_dataset(&heatdatas, input_filename, last_path, margins);
    if(err_read != 0)
    {
      fprintf(stderr, "Error in read_dataset_known_dims_hdf5: return %d instead of 0\n", err_read);
      exit(EXIT_FAILURE);
    }

    if(rank_comm_world == 0)
      heatdatas_fprint(&heatdatas, stderr);

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
  MPI_Finalize();
  return 0;
}
