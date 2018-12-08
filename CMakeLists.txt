cmake_minimum_required(VERSION 3.10)

project(HEAT)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(
       heat
       heat.c
)

add_executable(
       mean
       mean.c
)

add_executable(
       laplacian
       laplacian.c
)

add_executable(
       derivative
       derivative.c
)

target_include_directories(heat PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})
target_include_directories(mean PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})
target_include_directories(laplacian PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})
target_include_directories(derivative PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})

target_link_libraries(heat ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
target_link_libraries(mean ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
target_link_libraries(laplacian ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
target_link_libraries(derivative ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
