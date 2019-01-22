#include <stdlib.h>
#include <hdf5.h>
#include <mpi.h>
#include <utils.h>
#include <assert.h>

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


