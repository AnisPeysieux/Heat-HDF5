#include <math.h>

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


