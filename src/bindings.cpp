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
    nb::module_::import_("madrona_python");

    nb::enum_<Manager::ExecMode>(m, "ExecMode")
        .value("CPU", Manager::ExecMode::CPU)
        .value("CUDA", Manager::ExecMode::CUDA)
        .export_values();

    nb::class_<Manager> (m, "SimpleSimulator")
        .def("__init__", [](Manager *self,
                            Manager::ExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t num_obstacles,
                            int64_t render_width,
                            int64_t render_height,
                            bool enable_render) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .numObstacles = (uint32_t)num_obstacles,
                .renderWidth = (uint32_t)render_width,
                .renderHeight = (uint32_t)render_height,
                .enableRender = enable_render,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("num_obstacles"),
           nb::arg("render_width") = 0,
           nb::arg("render_height") = 0,
           nb::arg("enable_render") = false)
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("position_tensor", &Manager::positionTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
    ;
}

}
