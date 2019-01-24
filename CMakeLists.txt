#test
cmake_minimum_required(VERSION 3.10)

project(HEAT)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_compile_options(-Wall)

add_library(
       heatdatas SHARED
       heatdatas.c
       steps.c
)

target_include_directories(heatdatas PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIRS} .)
target_link_libraries(heatdatas ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)


add_executable(
       heat
       heat.c
)

add_executable(
       mean
       mean.c
       utils.c
       hdf5_utils.c
)

add_executable(
       laplacian
       laplacian.c
)

add_executable(
       derivative
       derivative.c
)
message("HDF5_INCLUDE_DIRS = ${HDF5_INCLUDE_DIRS}")
message("HDF5_LIBRARIES = ${HDF5_LIBRARIES}")
message("HDF5_HL_LIBRARIES = ${HDF5_HL_LIBRARIES}")
message("HDF5_IS_PARALLEL = ${HDF5_IS_PARALLEL}")
message("HDF5_C_COMPILER_EXECUTABLE = ${HDF5_C_COMPILER_EXECUTABLE}")
message("HDF5_CXX_COMPILER_EXECUTABLE = ${HDF5_CXX_COMPILER_EXECUTABLE}")
message("HDF5_FOUND = ${HDF5_FOUND}")
target_include_directories(heat PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIRS} .)
target_include_directories(mean PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS} .)
target_include_directories(laplacian PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS} .)
target_include_directories(derivative PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS} .)

target_link_libraries(heat heatdatas ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
target_link_libraries(mean heatdatas ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
target_link_libraries(laplacian heatdatas ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
target_link_libraries(derivative heatdatas ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
