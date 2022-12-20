#pragma once

#include <memory>

#include <madrona/python.hpp>

namespace SimpleExample {

class Manager {
public:
    struct Config {
        int gpuID;
        uint32_t numWorlds;
        uint32_t numRectanglesPerWorld;
        float boundsMinX;
        float boundsMaxX;
        float boundsMinY;
        float boundsMaxY;
        int32_t minWidth;
        int32_t maxWidth;
        int32_t minHeight;
        int32_t maxHeight;
        bool debugCompile;
    };

    MADRONA_IMPORT Manager(const Config &cfg);
    MADRONA_IMPORT ~Manager();

    MADRONA_IMPORT void step();

    MADRONA_IMPORT madrona::py::GPUTensor resetTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor actionTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor positionTensor() const;
    MADRONA_IMPORT madrona::py::GPUTensor numOverlapsTensor() const;

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
