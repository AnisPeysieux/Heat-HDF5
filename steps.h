#ifndef STEPS_HEADER
#define STEPS_HEADER

#include <hdf5.h>

#define STEPS_REALLOCATIONS_FACTOR 2

struct steps_s
{
  int capacity;
  int size;
  int* steps;
};
typedef struct steps_s steps_t;

void steps_init(steps_t* steps);
int steps_get(steps_t* steps, int n, int* out);
herr_t step_iteration(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data);
int steps_load(char* filename, steps_t* steps);

#endif
