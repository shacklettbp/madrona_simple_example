#pragma once

namespace SimpleExample {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    int32_t numAgents;
};

}
