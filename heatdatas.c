#include <heatdatas.h>
#include <stdlib.h>
#include <hdf5.h>
#include <math.h>

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

int heatdatas_buf(heatdatas_t* heatdatas, void** buf)
{
  *buf = heatdatas->buf;
  return 0;
}

int heatdatas_load_dims(heatdatas_t* heatdatas, char* filename, char* dataset_name)
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

int heatdatas_distribute(heatdatas_t* heatdatas) 
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

int heatdatas_load_dataset(heatdatas_t* heatdatas, char* filename, char* dataset_name)
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


