#include <mpi.h>
#include <hdf5.h>
#include <hdf5_hl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <heatdatas.h>

#define NO_POSTTREATMENT 0
#define IN_SITU_POSTREATMENT 1
#define IN_TRANSIT_POSTREATMENT 2

/** A function to initialize the temperature at t=0
 * @param      dsize  size of the local data block (including ghost zones)
 * @param      pcoord position of the local data block in the array of data blocks
 * @param[out] dat    the local data block to initialize
 */
void init(int dsize[2], int pcoord[2], double dat[dsize[0]][dsize[1]])
{
	// initialize everything to 0
	for (int yy=0; yy<dsize[0]; ++yy) {
		for (int xx=0; xx<dsize[1]; ++xx) {
			dat[yy][xx] = 0;
		}
	}
	// except the boundary condition at x=0 if our block is at the boundary itself
	if ( pcoord[1] == 0 ) {
		for (int yy=0; yy<dsize[0]; ++yy) {
			dat[yy][0] = 1000000;
		}
	}
}

/** A function to compute the temperature at t+delta_t based on the temperature at t
 * @param      dsize  size of the local data block (including ghost zones)
 * @param      pcoord position of the local data block in the array of data blocks
 * @param[in]  cur    the current value (t) of the local data block
 * @param[out] next   the next value (t+delta_t) of the local data block
 */
void iter(int dsize[2], double cur[dsize[0]][dsize[1]], double next[dsize[0]][dsize[1]])
{
	int xx, yy;
	// copy the boundary values at x=0 (Dirichlet boundary condition)
	for (xx=0; xx<dsize[1]; ++xx) {
		next[0][xx] = cur[0][xx];
	}
	for (yy=1; yy<dsize[0]-1; ++yy) {
		// copy the boundary values at y=0 (Dirichlet boundary condition)
		next[yy][0] = cur[yy][0];
		for (xx=1; xx<dsize[1]-1; ++xx) {
			next[yy][xx] =
			    (cur[yy][xx]   *.5)
			    + (cur[yy][xx-1] *.125)
			    + (cur[yy][xx+1] *.125)
			    + (cur[yy-1][xx] *.125)
			    + (cur[yy+1][xx] *.125);
		}
		// copy the boundary values at y=YMAX (Dirichlet boundary condition)
		next[yy][dsize[1]-1] = cur[yy][dsize[1]-1];
	}
	// copy the boundary values at x=XMAX (Dirichlet boundary condition)
	for (xx=0; xx<dsize[1]; ++xx) {
		next[dsize[0]-1][xx] = cur[dsize[0]-1][xx];
	}
}

/** A function to update the values of the ghost zones
 * @param      cart_comm a MPI Cartesian communicator including all processes arranged in grid
 * @param      dsize     size of the local data block (including ghost zones)
 * @param[out] next      the next value (t+delta_t) of the local data block
 */
void exchange(MPI_Comm cart_comm, int dsize[2], double cur[dsize[0]][dsize[1]])
{
	MPI_Status status;
	int rank_source, rank_dest;
	static MPI_Datatype column, row;
	static int initialized = 0;
	
    // Build the MPI datatypes if this is the first time this function is called
	if ( !initialized ) {
        // A vector column when exchanging width neighbours on left/right
		MPI_Type_vector(dsize[0]-2, 1, dsize[1], MPI_DOUBLE, &column);
		MPI_Type_commit(&column);
        // A row column when exchanging width neighbours on top/down
		MPI_Type_contiguous(dsize[1]-2, MPI_DOUBLE, &row);
		MPI_Type_commit(&row);
		initialized = 1;
	}
	
	// send to the bottom, receive from the top
	MPI_Cart_shift(cart_comm, 0, 1, &rank_source, &rank_dest);
	MPI_Sendrecv(&cur[dsize[0]-2][1], 1, row, rank_dest,   100, /* send row before ghost */
	    &cur[0][1], 1, row, rank_source, 100, /* receive 1st row (ghost) */
	    cart_comm, &status);
	    
	// send to the top, receive from the bottom
	MPI_Cart_shift(cart_comm, 0, -1, &rank_source, &rank_dest);
	MPI_Sendrecv(&cur[1][1], 1, row, rank_dest,   100, /* send column after ghost */
	    &cur[dsize[0]-1][1], 1, row, rank_source, 100, /* receive last column (ghost) */
	    cart_comm, &status);
	    
	// send to the right, receive from the left
	MPI_Cart_shift(cart_comm, 1, 1, &rank_source, &rank_dest);
	MPI_Sendrecv(&cur[1][dsize[1]-2], 1, column, rank_dest,   100, /* send column before ghost */
	    &cur[1][0], 1, column, rank_source, 100, /* receive 1st column (ghost) */
	    cart_comm, &status);
    
	// send to the left, receive from the right
	MPI_Cart_shift(cart_comm, 1, -1, &rank_source, &rank_dest);
	MPI_Sendrecv(&cur[1][1], 1, column, rank_dest,   100, /* send column after ghost */
	    &cur[1][dsize[1]-1], 1, column, rank_source, 100, /* receive last column (ghost) */
	    cart_comm, &status);
}

int comp(const void *elem1, const void *elem2)
{
  int e1 = *((int*)elem1);
  int e2 = *((int*)elem2);
  if(e1 > e2)
  {
    return 1;
  }
  else if(e2 > e1)
  {
    return -1;
  }
  else
  {
    return 0;
  }
}

/** A function to parse command line arguments
 * @param      argc      number of arguments received on the command line
 * @param[in]  argv      values of arguments received on the command line
 * @param[out] nb_iter   number of iterations to execute
 * @param[out] dsize     size of the local data block (including ghost zones)
 * @param[out] cart_comm a MPI Cartesian communicator including all processes arranged in grid
 */
void parse_args( int argc, char *argv[], int *nb_iter, int **posttreatment_steps, int *posttreatment_steps_size, int *posttreatment_method, int dsize[2], MPI_Comm *cart_comm )
{
  if(argc < 6)
  {
		printf("Usage: %s <Nb_iter> <height> <width> -output [<n> ...]\n", argv[0]);
		exit(1);
	}
	
	// total number of processes
	int comm_size; MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	int psize[2];
    
	// number of processes in the x dimension
	psize[0] = sqrt(comm_size);
	if ( psize[0]<1 ) psize[0]=1; 
	// number of processes in the y dimension
	psize[1] = comm_size/psize[0];
	// make sure the total number of processes is correct
	if (psize[0]*psize[1] != comm_size) {
        fprintf(stderr, "Error: invalid number of processes\n");
        abort();
    }
	
	// number of iterations
	*nb_iter = atoi(argv[1]);
    
	// global width of the problem
	dsize[1] = atoi(argv[3]);
	if ( dsize[1]%psize[1] != 0) {
        fprintf(stderr, "Error: invalid problem width\n");
        abort();
    }
	// width of the local data block (add boundary or ghost zone: 1 point on each side)
	dsize[1]  = dsize[1]/psize[1]  + 2;
	
	// global height of the problem
	dsize[0] = atoi(argv[2]);
	if ( dsize[0]%psize[0] != 0) {
        fprintf(stderr, "Error: invalid problem height\n");
        abort();
    }
	// height of the local data block (add boundary or ghost zone: 1 point on each side)
	dsize[0] = dsize[0]/psize[0] + 2;
	
    // creation of the communicator
	int cart_period[2] = { 0, 0 };
	MPI_Cart_create(MPI_COMM_WORLD, 2, psize, cart_period, 1, cart_comm);
  
  *posttreatment_steps = malloc((argc - 5) * sizeof(int));
  *posttreatment_steps_size = argc - 5;
  for(int i = 0; i < argc - 5; ++i)
  {
    printf("(*posttreatment_steps)[%d] <- argv[%d] = %s\n", i, i + 5, argv[i + 5]);
    (*posttreatment_steps)[i] = atoi(argv[i + 5]);
  }
  *posttreatment_method = NO_POSTTREATMENT;
  qsort(*posttreatment_steps, argc - 5, sizeof(int), comp);
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

int save_in_hdf5 (MPI_Comm cart_comm, char* filename, int step, char* previous_name, char* last_name, int dsize[2], int pcoord[2], double(*previous)[], double(*last)[])
{
  assert(filename != NULL);
  assert(previous_name != NULL);
  assert(last_name != NULL);
  assert(previous != NULL);
  assert(last != NULL);
  assert(strcmp(previous_name, last_name) != 0);
  //printf("save %d\n", step);
  static int first_access = 1;  
  //Get dimensions of the communicator
  int dims[2];
  int periods[2];
  int coords[2];
  MPI_Cart_get(cart_comm, 2, dims, periods, coords);
  
  //Create access properties
  hid_t plist_id_1 = H5Pcreate(H5P_FILE_ACCESS);
  if(plist_id_1 == -1) { return -1; }
  if(H5Pset_fapl_mpio(plist_id_1, MPI_COMM_WORLD, MPI_INFO_NULL) < 0) { return -2; }

  //Create write properties
  hid_t plist_id_2 = H5Pcreate(H5P_DATASET_XFER);
  if(plist_id_2 == -1) { return -3; }
  if(H5Pset_dxpl_mpio(plist_id_2, H5FD_MPIO_COLLECTIVE) < 0) { return -4; }
  
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  //Create file
  hid_t snapshot_file_id;
  if(first_access)
  {
    snapshot_file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_1);
    first_access = 0;
  }
  else
  {
    snapshot_file_id = H5Fopen(filename, H5F_ACC_RDWR, plist_id_1);
    if(snapshot_file_id < 0) { return -5; }
  }
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  //Computing dataspace sizes
  hsize_t dims_file[2] = {(dsize[0] - 2) * dims[0], (dsize[1] - 2) * dims[1]};
  hsize_t dims_mem[2] = {dsize[0], dsize[1]};
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  //Create dataspaces
  hid_t dataspace_file = H5Screate_simple(2, dims_file, NULL); if(dataspace_file < 0) { return -6; }
  hid_t dataspace_mem = H5Screate_simple(2, dims_mem, NULL); if(dataspace_mem < 0) { return -7; }
  //fprintf(stderr,"LINE: %d\n", __LINE__);

  //Computing slabs area
  hsize_t start_mem[2] = {1, 1};
  hsize_t count_mem[2] = {dsize[0] - 2, dsize[1] - 2};
  hsize_t start_file[2] = {pcoord[0] * (dsize[0] - 2), pcoord[1] * (dsize[1] - 2)};
  hsize_t count_file[2] = {dsize[0] - 2, dsize[1] - 2};
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  //Create slabs
  if(H5Sselect_hyperslab(dataspace_mem, H5S_SELECT_SET, start_mem, NULL, count_mem, NULL) < 0) { return -8; }  
  if(H5Sselect_hyperslab(dataspace_file, H5S_SELECT_SET, start_file, NULL, count_file, NULL) < 0) { return -9; }
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  //fprintf(stderr, "dsize[0] = %d, dsize[1] = %d, pcoord[0] = %d, pcoord[1] = %d, start_mem[0] = %lld, start_mem[1] = %lld, count_mem[0] = %lld, count_mem[1] = %lld, start_file[0] = %lld, start_file[1] = %lld, count_file[0] = %lld, count_file[1] = %lld",
  //        dsize[0], dsize[1], pcoord[0], pcoord[1], start_mem[0], start_mem[1], count_mem[0], count_mem[1], start_file[0], start_file[1], count_file[0], count_file[1]);

  //Compute size of /<iter>/<name>
  
  //Creating datasets
  
  int str_length_step = string_length_int(step);
  int length_previous_path = 1 + str_length_step + 1 + strlen(previous_name) + 1;
  int length_last_path = 1 + str_length_step + strlen(previous_name) + 1;
  int length_group_path = 1 + str_length_step + 1;
  char* previous_path = malloc (length_previous_path * sizeof(char));
  char* last_path = malloc(length_last_path * sizeof(char));
  char* group_path = malloc(length_group_path * sizeof(char));
  snprintf(previous_path, length_previous_path, "/%d/%s", step, previous_name);
  snprintf(last_path, length_last_path, "/%d/%s", step, last_name);
  snprintf(group_path, length_last_path, "/%d", step);

  hid_t gcpl = H5Pcreate (H5P_LINK_CREATE);
  hid_t group = H5Gcreate (snapshot_file_id, group_path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  //hid_t dataset_previous = H5Dcreate(snapshot_file_id, previous_path, H5T_NATIVE_DOUBLE, dataspace_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  hid_t dataset_previous = H5Dcreate(group, previous_name, H5T_NATIVE_DOUBLE, dataspace_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  //fprintf(stderr, "previous_name:%s\n", previous_name);
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  if(dataset_previous < 0) { return -10; }
  //hid_t dataset_last = H5Dcreate(snapshot_file_id, last_path, H5T_NATIVE_DOUBLE, dataspace_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dataset_last = H5Dcreate(group, last_name, H5T_NATIVE_DOUBLE, dataspace_file, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  if(dataset_last < 0) { return -11; }
  //fprintf(stderr,"LINE: %d\n", __LINE__);

  //Writing in datasets
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  if(H5Dwrite(dataset_previous, H5T_NATIVE_DOUBLE, dataspace_mem, dataspace_file, plist_id_2, previous) < 0) { return -12; }
  //fprintf(stderr,"LINE: %d\n", __LINE__);
  if(H5Dwrite(dataset_last, H5T_NATIVE_DOUBLE, dataspace_mem, dataspace_file, plist_id_2, last) < 0) { return -13; }
  //fprintf(stderr,"LINE: %d\n", __LINE__);

  //Closing ressources
  if(H5Pclose (gcpl) < 0) { return -14; }
  if(H5Gclose (group) < 0) { return -15; }
  if(H5Sclose(dataspace_file) < 0) { return -16; }
  if(H5Sclose(dataspace_mem) < 0) { return -17; }
  if(H5Dclose(dataset_previous) < 0) { return -18; }
  if(H5Dclose(dataset_last) < 0) { return -19; }
  if(H5Fclose(snapshot_file_id) < 0) { return -20; }

  return 0;
}

int next_step = 0;
int main( int argc, char* argv[] )
{
	// initialize the MPI library
	MPI_Init(&argc, &argv);
	
	// parse the command line arguments
  int nb_iter;
	int dsize[2];
  int *posttreatment_steps;
  int posttreatment_steps_size;
  int posttreatment_method;
    MPI_Comm cart_comm;
	parse_args(argc, argv, &nb_iter, &posttreatment_steps, &posttreatment_steps_size, &posttreatment_method, dsize, &cart_comm);
//	printf("posttreatment_steps_size = %d\n", posttreatment_steps_size);
//  for(int i = 0; i < posttreatment_steps_size; ++i)
//  {
//    printf("step %d = %d\n",i,  posttreatment_steps[i]);
//  }
//  printf("posttreatment_method = %d\n", posttreatment_method);
//  printf("nb_iter = %d\n", nb_iter);
    // find the coordinate of the 
	int pcoord_1d; MPI_Comm_rank(MPI_COMM_WORLD, &pcoord_1d);
	int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int pcoord[2]; MPI_Cart_coords(cart_comm, pcoord_1d, 2, pcoord);
  
  // allocate data for the current iteration
	double(*cur)[dsize[1]]  = malloc(sizeof(double)*dsize[1]*dsize[0]);
	// initialize data at t=0
	init(dsize, pcoord, cur);
    
	// allocate data for the next iteration
	double(*next)[dsize[1]] = malloc(sizeof(double)*dsize[1]*dsize[0]);
	
  fprintf(stderr, "rank %d, pcoord={%d, %d} dsize={%d, %d}\n", rank, pcoord[0], pcoord[1], dsize[0], dsize[1]);
	// the main (time) iteration
  heatdatas_t heatdatas;
  int dsize2[2];
  dsize2[0] = atoi(argv[2]);
  dsize2[1] = atoi(argv[3]);
  //heatdatas_distribute_from_MPI_Cart(&heatdatas, cart_comm, dsize2);
  heatdatas_set_comm_from_MPI_Cart(&heatdatas, cart_comm);
  heatdatas_set_n_local_rows(&heatdatas, dsize[0]-2);
  heatdatas_set_n_local_cols(& heatdatas, dsize[1]-2);
  heatdatas_set_n_global_rows(&heatdatas, dsize2[0]);
  heatdatas_set_n_global_cols(&heatdatas, dsize2[1]);
  heatdatas_set_offset_rows(&heatdatas, pcoord[0] * (dsize[0] - 2));
  heatdatas_set_offset_cols(&heatdatas, pcoord[1] * (dsize[1] - 2));
  int margins[4] = {1, 1, 1, 1};
  heatdatas_set_margins(&heatdatas, margins);
  
  fprintf(stderr, "\x1B[31m  rank %d, pcoord = {%d, %d} dsize = {%d, %d} \x1B[0m  \nlocal_dims = {%lld, %lld}, global_dims = {%lld, %lld}, offsets = {%lld, %lld}, rank_dims = {%d, %d}, margins = {%lld, %lld, %lld, %lld}\n",
          rank, pcoord[0], pcoord[1], dsize[0], dsize[1],
          heatdatas.local_dims[0], heatdatas.local_dims[1], heatdatas.global_dims[0], heatdatas.global_dims[1], heatdatas.offsets[0], heatdatas.offsets[1], heatdatas.ranks_dims[0], heatdatas.ranks_dims[1], heatdatas.margins[0], heatdatas.margins[1], heatdatas.margins[2], heatdatas.margins[3]);
	for (int ii=0; ii<nb_iter; ++ii) {
		// compute the temperature at the next iteration
		iter(dsize, cur, next);
        
        // update ghost zones
		exchange(cart_comm, dsize, next);
        // switch the current and next buffers
		double (*tmp)[dsize[1]] = cur; cur = next; next = tmp;
    if(next_step < posttreatment_steps_size && ii == (posttreatment_steps[next_step]))
    {
      int errcode = save_in_hdf5 (cart_comm, "heat.h5", posttreatment_steps[next_step], "previous", "last", dsize, pcoord, next, cur);
      if(errcode < 0)
      {
        fprintf(stderr, "Error during save: error code %d\n", errcode);
        exit(EXIT_FAILURE);
      }
      next_step++;
    }
	}


	// free memory
	free(cur);
	free(next);

 	// finalize MPI
	MPI_Finalize();
    
	return 0;
}
