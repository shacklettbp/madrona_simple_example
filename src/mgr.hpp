#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace SimpleExample {

class Manager {
public:
    enum class ExecMode {
        CPU,
        CUDA,
    };

    struct Config {
        ExecMode execMode;
        int gpuID;
        uint32_t numWorlds;
        uint32_t numAgentsPerWorld;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor resetTensor() const;
    MADRONA_IMPORT madrona::py::Tensor actionTensor() const;
    MADRONA_IMPORT madrona::py::Tensor positionTensor() const;

private:
    struct Impl;
    struct CPUImpl;
    struct GPUImpl;

    std::unique_ptr<Impl> impl_;
};

}
