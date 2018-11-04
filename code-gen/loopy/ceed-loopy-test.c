#include "ceed-loopy.h"

int test00(void) {
  PyObject *t;

  t = PyTuple_New(3);
  PyTuple_SetItem(t,0,PyLong_FromLong(1L));

  return 0;
}
