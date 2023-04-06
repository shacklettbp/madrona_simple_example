#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace SimpleExample {

class Manager {
public:
    struct Config {
        int gpuID;
        uint32_t numWorlds;
        uint32_t numAgentsPerWorld;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::Tensor resetTensor() const;
    MADRONA_IMPORT madrona::py::Tensor actionTensor() const;
    MADRONA_IMPORT madrona::py::Tensor positionTensor() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
