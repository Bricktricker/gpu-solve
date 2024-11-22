#pragma once

#ifndef SYCL_IF
#define SYCL_IF if
#endif
#ifndef  SYCL_ELSE_IF
#define SYCL_ELSE_IF else if
#endif
#ifndef  SYCL_ELSE
#define SYCL_ELSE else
#endif
#ifndef SYCL_END
#define SYCL_END
#endif
#ifndef SYCL_FOR
#define SYCL_FOR(init, condition, increment) \
  for (init; condition; increment)
#endif

#ifdef SYCL_GTX_TARGET
using cl::sycl::int1;
using cl::sycl::double1;
#else
#define int1 int
#define double1 double
#endif
