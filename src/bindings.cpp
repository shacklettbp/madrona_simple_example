#include "mgr.hpp"

#include <madrona/macros.hpp>

#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weverything"
#endif
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>
#if defined(MADRONA_CLANG) || defined(MADRONA_GCC)
#pragma GCC diagnostic pop
#endif

namespace nb = nanobind;

namespace SimpleExample {

NB_MODULE(madrona_simple_example_python, m) {
    nb::class_<Manager> (m, "SimpleSimulator")
        .def("__init__", [](Manager *self,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t num_rectangles_per_world,
                            double bounds_min_x,
                            double bounds_max_x,
                            double bounds_min_y,
                            double bounds_max_y,
                            int64_t min_width,
                            int64_t max_width,
                            int64_t min_height,
                            int64_t max_height,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .numRectanglesPerWorld = (uint32_t)num_rectangles_per_world,
                .boundsMinX = float(bounds_min_x),
                .boundsMaxX = float(bounds_max_x),
                .boundsMinY = float(bounds_min_y),
                .boundsMaxY = float(bounds_max_y),
                .minWidth = int32_t(min_width),
                .maxWidth = int32_t(max_width),
                .minHeight = int32_t(min_height),
                .maxHeight = int32_t(max_height),
                .debugCompile = debug_compile,
            });
        }, nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("num_rectangles_per_world"),
           nb::arg("bounds_min_x"),
           nb::arg("bounds_max_x"),
           nb::arg("bounds_min_y"),
           nb::arg("bounds_max_y"),
           nb::arg("min_width"),
           nb::arg("max_width"),
           nb::arg("min_height"),
           nb::arg("max_height"),
           nb::arg("debug_compile") = true)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("position_tensor", &Manager::positionTensor)
        .def("num_overlaps_tensor", &Manager::numOverlapsTensor)
    ;
}

}
