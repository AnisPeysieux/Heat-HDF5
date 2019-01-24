#include <heatdatas.h>
#include <stdlib.h>
#include <hdf5.h>
#include <math.h>
#include <utils.h>

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

int heatdatas_set_n_local_rows(heatdatas_t* heatdatas, hdsize_t nrows)
{
  heatdatas -> local_dims[0] = nrows;
  return 0;
}

int heatdatas_set_n_local_cols(heatdatas_t* heatdatas, hdsize_t ncols)
{
  heatdatas -> local_dims[1] = ncols;
  return 0;
}

int heatdatas_set_n_global_rows(heatdatas_t* heatdatas, hdsize_t nrows)
{
  heatdatas -> global_dims[0] = nrows;
  return 0;
}

int heatdatas_set_n_global_cols(heatdatas_t* heatdatas, hdsize_t ncols)
{
  heatdatas -> global_dims[1] = ncols;
  return 0;
}

int heatdatas_set_offset_rows(heatdatas_t* heatdatas, hdsize_t offset_rows)
{
  heatdatas -> offsets[0] = offset_rows;
  return 0;
}

int heatdatas_set_offset_cols(heatdatas_t* heatdatas, hdsize_t offset_cols)
{
  heatdatas -> offsets[1] = offset_cols;
  return 0;
}

int heatdatas_set_margins(heatdatas_t* heatdatas, int margins[4])
{
  heatdatas -> margins[0] = margins[0];
  heatdatas -> margins[1] = margins[1];
  heatdatas -> margins[2] = margins[2];
  heatdatas -> margins[3] = margins[3];
  
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

int heatdatas_set_comm_from_MPI_Cart(heatdatas_t* heatdatas, MPI_Comm cart_comm) 
{
  int rank_comm_world; 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_comm_world);
	int pcoord[2]; MPI_Cart_coords(cart_comm, rank_comm_world, 2, pcoord);
  int dims[2];
  int coords[2];
  int periods[2];
  MPI_Cart_get(cart_comm, 2, dims, periods, coords);
  heatdatas->ranks_dims[0] = pcoord[0];
  heatdatas->ranks_dims[1] = pcoord[1];
  MPI_Comm_split(MPI_COMM_WORLD, heatdatas->ranks_dims[1], heatdatas->ranks_dims[0], &(heatdatas->comm_dims[0]));
  MPI_Comm_split(MPI_COMM_WORLD, heatdatas->ranks_dims[0], heatdatas->ranks_dims[1], &(heatdatas->comm_dims[1]));
  MPI_Comm_rank(heatdatas->comm_dims[0], &(heatdatas->ranks_dims[0]));
  MPI_Comm_rank(heatdatas->comm_dims[1], &(heatdatas->ranks_dims[1]));


  return 0; 
}

//margin: {up, left, down, right}
int heatdatas_load_dataset(heatdatas_t* heatdatas, char* filename, char* dataset_name, hdsize_t margins[4])
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); if(file_id < 0) { return -1; }
  hid_t dset = H5Dopen (file_id, dataset_name, H5P_DEFAULT); if(dset < 0) { return -2; }
  hid_t dspace = H5Dget_space(dset); if(dspace < 0) { return -3; }
  if(H5Sselect_hyperslab(dspace, H5S_SELECT_SET, heatdatas->offsets, NULL, heatdatas->local_dims, NULL) < 0) { return -4; }
  //if(H5Sget_simple_extent_dims(dspace, heatdatas -> dims, NULL) < 0) { return -6; }
	heatdatas -> buf  = malloc(sizeof(double)*(heatdatas->local_dims[1] + margins[1] + margins[3])*(heatdatas->local_dims[0] + margins[0] + margins[2])); if(heatdatas -> buf == NULL) { return -7; }
  hsize_t mem_size[2];
  mem_size[0] = heatdatas->local_dims[0] + margins[0] + margins[2];
  mem_size[1] = heatdatas->local_dims[1] + margins[1] + margins[3];
  hid_t dataspace_mem = H5Screate_simple(2, mem_size, NULL); if(dataspace_mem < 0) { return -5; }
  //hid_t dataspace_mem = H5Screate_simple(2, heatdatas->local_dims, NULL); if(dataspace_mem < 0) { return -5; }
  H5Sselect_hyperslab(dataspace_mem, H5S_SELECT_SET, margins, NULL, heatdatas->local_dims, NULL);
  heatdatas->margins[0] = margins[0];
  heatdatas->margins[1] = margins[1];
  heatdatas->margins[2] = margins[2];
  heatdatas->margins[3] = margins[3];
  if(H5Dread(dset, H5T_NATIVE_DOUBLE, dataspace_mem, dspace, H5P_DEFAULT, heatdatas -> buf) < 0) {return -8; }
  if(H5Dclose(dset) < 0) { return -9; }
  if(H5Fclose(file_id) < 0) { return -10; }

  return 0;
}

int heatdatas_fprint(heatdatas_t* heatdatas, FILE* out)
{
  fprintf(out, "(%lld/%lld) * (%lld/%lld)\n", heatdatas->local_dims[0], heatdatas->global_dims[0], heatdatas->local_dims[1], heatdatas->global_dims[1]);
  fprintf(out, "First row: %lld, first column: %lld\n", heatdatas->offsets[0], heatdatas->offsets[1]);
  fprintf(out, "Ranks: row %d, column %d\n", heatdatas->ranks_dims[0], heatdatas->ranks_dims[1]);
  fprintf(out, "Margin up: %lld down: %lld left: %lld, right: %lld\n", heatdatas->margins[0], heatdatas->margins[2], heatdatas->margins[1], heatdatas->margins[3]);

  double (*buf) [heatdatas->local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3]] = heatdatas->buf;
  for(int row = 0; row < heatdatas->local_dims[0] + heatdatas->margins[0] + heatdatas->margins[2]; ++row)
  {
//    if(row == heatdatas->margins[0])
//    {
//      int width = string_length_int(buf[row][heatdatas->margins[0]]) + (heatdatas->local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3]) * 12 + 10;
//      //for(int col = 0; col < heatdatas -> local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3]; ++col)
//      for(int col = 0; col < width; ++col)
//      {
//        fprintf(out, "-");
//      }
//      fprintf(out, "\n");
//    }
    for(int col = 0; col < heatdatas -> local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3]; ++col)
    {
//      if(col == heatdatas->margins[1])
//      {
//        fprintf(out, "|\t");
//      }
      fprintf(out, "%.5f\t\t", buf[row][col]);
//      if(col == heatdatas -> local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3] - 1)
//      {
//        fprintf(out, "|\t");
//      }
    }
    fprintf(out, "\n");

//    if(row == heatdatas -> local_dims[0] + heatdatas->margins[0] + heatdatas->margins[2] - 1)
//    {
//      int width = string_length_int(buf[row][heatdatas->margins[0]]) + (heatdatas->local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3]) * 12 + 10;
//      //for(int col = 0; col < heatdatas -> local_dims[1] + heatdatas->margins[1] + heatdatas->margins[3]; ++col)
//      for(int col = 0; col < width; ++col)
//      {
//        fprintf(out, "-");
//      }
//      fprintf(out, "\n");
//    }


  }
  return 0;
}

