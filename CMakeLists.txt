cmake_minimum_required(VERSION 3.10)

project(HEAT)
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

add_executable(
       heat
       heat.c

)
target_include_directories(heat PUBLIC ${MPI_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})

target_link_libraries(heat ${MPI_LIBRARIES} ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} m)
