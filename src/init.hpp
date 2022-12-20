#pragma once

namespace SimpleExample {

struct EpisodeManager {
    std::atomic_uint32_t curEpisode;
};

struct WorldInit {
    EpisodeManager *episodeMgr;
    int32_t numRectangles;
    float boundsMinX;
    float boundsMaxX;
    float boundsMinY;
    float boundsMaxY;
    int32_t minWidth;
    int32_t maxWidth;
    int32_t minHeight;
    int32_t maxHeight;
};

}
