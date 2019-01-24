#ifndef HEATDATAS_HEADER
#define HEATDATAS_HEADER

#include <hdf5.h>

struct heatdatas_s
{
  int ndims;
  hsize_t local_dims[2];
  hsize_t global_dims[2];
  hsize_t offsets[2];
  int ranks_dims[2];
  MPI_Comm comm_dims[2];
  hsize_t margins[4];
  void* buf;
};
typedef struct heatdatas_s heatdatas_t;
typedef hsize_t hdsize_t;

int heatdatas_n_local_rows(heatdatas_t* heatdatas, hdsize_t* nrows);
int heatdatas_n_local_cols(heatdatas_t* heatdatas, hdsize_t* ncols);
int heatdatas_n_global_rows(heatdatas_t* heatdatas, hdsize_t* nrows);
int heatdatas_n_global_cols(heatdatas_t* heatdatas, hdsize_t* ncols);
int heatdatas_offset_rows(heatdatas_t* heatdatas, hdsize_t* offset_rows);
int heatdatas_offset_cols(heatdatas_t* heatdatas, hdsize_t* offset_cols);
int heatdatas_comm_rows(heatdatas_t* heatdatas, MPI_Comm* comm_rows);
int heatdatas_comm_cols(heatdatas_t* heatdatas, MPI_Comm* comm_cols);
int heatdatas_rank_row(heatdatas_t* heatdatas, int* rank_row);
int heatdatas_rank_col(heatdatas_t* heatdatas, int* rank_col);
int heatdatas_buf(heatdatas_t* heatdatas, void** buf);
int heatdatas_set_n_local_rows(heatdatas_t* heatdatas, hdsize_t nrows);
int heatdatas_set_n_local_cols(heatdatas_t* heatdatas, hdsize_t ncols);
int heatdatas_set_n_global_rows(heatdatas_t* heatdatas, hdsize_t nrows);
int heatdatas_set_n_global_cols(heatdatas_t* heatdatas, hdsize_t ncols);
int heatdatas_set_offset_rows(heatdatas_t* heatdatas, hdsize_t offset_rows);
int heatdatas_set_offset_cols(heatdatas_t* heatdatas, hdsize_t offset_cols);
int heatdatas_set_margins(heatdatas_t* heatdatas, int margins[4]);
int heatdatas_load_dims(heatdatas_t* heatdatas, char* filename, char* dataset_name);
int heatdatas_distribute(heatdatas_t* heatdatas);
int heatdatas_set_comm_from_MPI_Cart(heatdatas_t* heatdatas, MPI_Comm cart_comm);
int heatdatas_load_dataset(heatdatas_t* heatdatas, char* filename, char* dataset_name, hdsize_t margins[4]);
int heatdatas_fprint(heatdatas_t* heatdatas, FILE* out);

#endif
