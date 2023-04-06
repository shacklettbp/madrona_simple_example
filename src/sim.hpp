#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace SimpleExample {

// 3D Position & Quaternion Rotation
// These classes are defined in madrona/components.hpp
using madrona::base::Position;
using madrona::base::Rotation;

class Engine;

struct WorldReset {
    int32_t resetNow;
};

struct Action {
    madrona::math::Vector3 positionDelta; // Continuous action
};

struct Agent : public madrona::Archetype<
    Position,
    Action
> {};

struct Sim : public madrona::WorldBase {
    struct Config {};

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    madrona::Entity *agents;
    int32_t numAgents;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
