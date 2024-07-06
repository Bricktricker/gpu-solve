#include "SYCL/handler.h"

#include "SYCL/context.h"
#include "SYCL/queue.h"

using namespace cl::sycl;
using namespace detail;

std::unordered_map<cl::sycl::string_class, cl_kernel> handler::kernel_cache;

context handler::get_context(queue* q) {
  return q->get_context();
}
