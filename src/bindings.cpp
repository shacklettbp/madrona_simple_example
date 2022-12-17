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
                            int64_t num_agents_per_world,
                            bool debug_compile) {
            new (self) Manager(Manager::Config {
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .numAgentsPerWorld = (uint32_t)num_agents_per_world,
                .debugCompile = debug_compile,
            });
        }, nb::arg("gpu_id"), nb::arg("num_worlds"),
           nb::arg("num_agents_per_world"), nb::arg("debug_compile") = true)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("position_tensor", &Manager::positionTensor)
    ;
}

}
