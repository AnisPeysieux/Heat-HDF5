project(HEAT)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(
       heat
       heat.c

)

target_link_libraries(heat ${MPI_LIBRARIES})
target_link_libraries(heat m)
