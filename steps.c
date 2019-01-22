#include <steps.h>
#include <assert.h>
#include <stdlib.h>

void steps_init(steps_t* steps)
{
  steps -> capacity = 0;
  steps -> size = 0;
  steps -> steps = NULL;
}

int steps_get(steps_t* steps, int n, int* out)
{
  *out = steps->steps[n];
  return 0;
}

herr_t step_iteration(hid_t g_id, const char *name, const H5L_info_t *info, void *op_data)
{
  assert(op_data != NULL);
  steps_t* steps = op_data;
  if(steps -> capacity == 0)
  {
    steps -> steps = malloc(sizeof(int));
    steps -> size = 0;
    steps -> capacity = 1;
  }
  else if(steps -> size == steps -> capacity)
  {
    steps -> steps = realloc(steps -> steps, steps -> capacity * STEPS_REALLOCATIONS_FACTOR * sizeof(int));
  }
  sscanf(name, "%d/last", steps -> steps + steps -> size);
  steps -> size++;
  return 0;
}

int steps_load(char* filename, steps_t* steps)
{

  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT); if(file_id < 0) { return -1; }
  hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT); if(group_id < 0) { return -2; }
  //H5Lvisit(group_id, H5_INDEX_CRT_ORDER, H5_ITER_NATIVE, print_groups, NULL);
  if(H5Literate(group_id, H5_INDEX_NAME, H5_ITER_NATIVE, NULL, step_iteration, steps) < 0) { return -3; }
  return 0;
}


