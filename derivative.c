#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <assert.h>

void write_data(hsize_t dims[2], const char* file_name, double derivative_buf[dims[0]][dims[1]])
{
    // Creating the file that will contains the derivative
    hid_t file_id = H5Fcreate (file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Creating the dataspace for derivative
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

    // Creating the dataset for derivative
    hid_t dataset_id = H5Dcreate(file_id, "/derivative", H5T_IEEE_F64LE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write data to the dataset
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, derivative_buf);

    // Closing dataset
    H5Dclose(dataset_id);

    // Closing dataspace
    H5Sclose(dataspace_id);

    //Closing the file
    H5Fclose( file_id );
}

int main()
{
    // The file name
    char filename[] = "heat.h5";  

    // Opening file
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    // Opening datasets /last and /previous
    hid_t last_dset = H5Dopen (file_id, "/last", H5P_DEFAULT);
    hid_t previous_dset = H5Dopen (file_id, "/previous", H5P_DEFAULT);

    // Getting the two dataspaces
    hid_t last_dspace = H5Dget_space(last_dset);
    hid_t previous_dspace = H5Dget_space(previous_dset);

    // Getting dimensions number for each dataspace
    const int last_ndims = H5Sget_simple_extent_ndims(last_dspace);
    const int previous_ndims = H5Sget_simple_extent_ndims(previous_dspace);

    // Checking that dimensions numbers of the two dataspaces are equal
    assert(previous_ndims == last_ndims);
    
    // Allocating dimensions size arrays
    hsize_t* last_dims = malloc(last_ndims * sizeof(hsize_t));
    hsize_t* previous_dims = malloc(previous_ndims * sizeof(hsize_t));

    // Getting dimensions size for each dataspace
    H5Sget_simple_extent_dims(last_dspace, last_dims, NULL);
    H5Sget_simple_extent_dims(previous_dspace, previous_dims, NULL);

    // Checking that dimensions size of the two dataspaces are equal (We suppose there is 2 dimensions)
    assert(last_dims[0] == previous_dims[0] && last_dims[1] == previous_dims[1]);

    // Once we are sure that dimensions sizes are the same, we free the buffer previous_dims
    free(previous_dims);

    // We allocate buffers that will contain the two datasets data
    double(*last_buf)[last_dims[1]]  = malloc(sizeof(double)*last_dims[1]*last_dims[0]);
    double(*previous_buf)[last_dims[1]]  = malloc(sizeof(double)*last_dims[1]*last_dims[0]);
    
    // Reading the two datasets and storing them in the two allocated buffers
    H5Dread(last_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  last_buf);
    H5Dread(previous_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  previous_buf);

    // Computing the derivative and storing the result in a buffer allocated previously
    for (size_t row = 0; row < last_dims[0]; ++row) {
        for (size_t col = 0; col < last_dims[1]; ++col) {
            last_buf[row][col] = last_buf[row][col] - previous_buf[row][col];
        }
    }

    // Writing the result in an hdf5 file called "diag.h5", the dataset name is derivative.
    write_data(last_dims, "diags.h5", last_buf);

    // We free all the remaining allocated buffers.
    free(last_dims);
    free(last_buf);
    free(previous_buf);

    // Closing dataspaces
    H5Sclose(last_dspace);
    H5Sclose(previous_dspace);
    
    // Closing datasets
    H5Dclose(last_dset);
    H5Dclose(previous_dset);
    
    // // Closing file
    H5Fclose(file_id);
    
    return 0;
}
