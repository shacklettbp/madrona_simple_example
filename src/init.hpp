#pragma once

#include <madrona/physics.hpp>

namespace SimpleExample {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    madrona::phys::ObjectManager *rigidBodyObjMgr;
    int32_t numObstacles;
};

}
