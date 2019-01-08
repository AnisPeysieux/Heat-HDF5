#include <stdio.h>
#include <stdlib.h>
//#include <hdf5.h>
#include <hdf5_hl.h>


// h5pcc -g -Wall -o laplacian laplacian.c -lm
// ./laplacian

void iter(int dsize[2], double cur[dsize[0]][dsize[1]], double next[dsize[0]-2][dsize[1]-2])
{
    int xx, yy;
    for (yy=1; yy<dsize[0]-1; ++yy) {
        for (xx=1; xx<dsize[1]-1; ++xx) {
            next[yy - 1][xx - 1] =
            (cur[yy][xx]   *.5)
            + (cur[yy][xx-1] *.125)
            + (cur[yy][xx+1] *.125)
            + (cur[yy-1][xx] *.125)
            + (cur[yy+1][xx] *.125); 
        }
        
    }
}


int main()
{
    // NOM DU FICHIER DUQUEL ON RECUPERE LES DONNEES
    char filename[] = "heat.h5";
    
    
    // OUVERTURE DU FICHIER
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    
    // OUVERTURE DATASET ET DATASPACE
    hid_t last_dset = H5Dopen (file_id, "/last", H5P_DEFAULT);
    hid_t last_dspace = H5Dget_space(last_dset);
    
    // ON RECUPERE LES DIMENSIONS
    const int last_ndims = H5Sget_simple_extent_ndims(last_dspace);
    
    // ON ALLOUE UN TABLEAU DE DIMENSIONS AVEC CELLES QU'ON A RECUPERE
    hsize_t* last_dims;
    last_dims = malloc(last_ndims * sizeof(hsize_t));
    H5Sget_simple_extent_dims(last_dspace, last_dims, NULL);
    
    // ON ALLOUE UN TABLEAU last_buf DANS LEQUEL ON MET LES DONNEES DU FICHIER
    double(*last_buf)[last_dims[1]]  = malloc(sizeof(double)*last_dims[1]*last_dims[0]) ;
    H5Dread(last_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,  last_buf); 
    
    // ON CREE UN TABLEAU DE DIMENSION AVEC DES "ZONES FANTOMES", POUR METTRE LES 1000000 SUR LE BORD GAUCHE ET LES 0 SUR LES AUTRES BORDS
    int laplac_dims[2] = { last_dims[0] + 2, last_dims[1] + 2 } ;
    
    // ON ALLOUE UN TABLEAU 2D AVEC CES DIMENSIONS
    double(*last_buf_million)[laplac_dims[1]]  = malloc(sizeof(double)*(laplac_dims[1])*laplac_dims[0]) ;
    
    
    // ON REMPLIE LE TABLEAU (HORS ZONES FANTOMES) AVEC LES DONNEES DE LAST_BUF (récupérées dans le fichier)
    for ( int i = 1 ; i < laplac_dims[0] - 1 ; i++ ) {
        for (int j = 1 ; j < laplac_dims[1] - 1 ; j++) {             
            last_buf_million[i][j] = last_buf[i-1][j-1];
            
        }
    }
    
    // ON REMPLIE LES ZONES FANTOMES, 1000000 SUR LE BORD GAUCHE, ZERO SUR LES AUTRES
    for ( int i = 0; i < laplac_dims[0] ; i++ ) {                
        for (int j = 0 ; j < laplac_dims[1] ; j++) {
            last_buf_million[i][0] = 1000000 ;
            last_buf_million[i][laplac_dims[1] - 1] = 0 ;
            last_buf_million[0][j] = 0;
            last_buf_million[laplac_dims[0] - 1][j] = 0;
        }
    }
    
    
    /* 
     * for ( int i = 0; i < laplac_dims[0] ; i++) {
     *    for ( int j = 0 ; j < laplac_dims[1] ; j++) {
     *        printf("%f *", last_buf_million[i][j] );
     *        
}
if (i % (laplac_dims[0] - 1) == 0) { printf("\n") ; }
} */
    
    
    // ON ALLOUE UN TABLEAU DANS LEQUEL ON METTRA LE RESULTAT DU LAPLACIEN
    double(*laplac_res)[last_dims[1]]  = malloc(sizeof(double)*last_dims[1]*last_dims[0]) ;
    
    // ON APPELLE ITER
    iter(laplac_dims , last_buf_million, laplac_res);               
    
    /*
     *  for ( int i = 0; i < last_dims[0] ; i++) {
     *    for ( int j = 0 ; j < last_dims[1] ; j++) {
     *        printf("%f *", last_buf[i][j] );                                    // affichage du tableau avec les mailles fantomes et le million
     *        if ((j != 0) && (j % (last_dims[0] - 1) == 0)) { printf("\n") ; }
}

}

printf("\n \n \n");

for ( int i = 0; i < last_dims[0] ; i++) {
    for ( int j = 0 ; j < last_dims[1] ; j++) {                    // affichage du resultat du laplacien
        printf("%f *", laplac_res[i][j] );
        if ((j != 0) && (j % (last_dims[0] - 1) == 0)) { printf("\n") ; }
}

}
*/
    
    
    
    // CREATION DU FICHIER
    hid_t file_create = H5Fcreate("diags.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT) ; 
    
    // CREATION DATASPACE DATASET
    hid_t dataspace = H5Screate_simple(2, last_dims, NULL);
    hid_t dataset = H5Dcreate(file_create, "/laplacian", H5T_IEEE_F64LE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // ECRITURE
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, (double*)laplac_res);
    
    // LIBERATIONS DE L'ESPACE MEMOIRE ET FERMETURES
    free(laplac_res);
    free(last_buf);
    free(last_buf_million);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    
    H5Fclose(file_create);
    H5Fclose(file_id);
    H5Dclose(last_dset);
    
    return 0;
    
}
