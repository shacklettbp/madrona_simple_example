#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace SimpleExample {

// 3D Position & Scale
// These classes are defined in madrona/components.hpp
// The majority of this code assumes everything is in the z = 0
// plane, so only 2D is really supported by this example
using madrona::base::Position;
using madrona::base::Scale;

class Engine;

struct WorldReset {
    int32_t resetNow;
};

struct Action {
    madrona::math::Vector3 positionDelta; // Continuous action
};

struct OverlapInfo {
    int32_t numOverlaps;
};

struct Rectangle : public madrona::Archetype<
    Position,
    Scale,
    Action,
    OverlapInfo
> {};

struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry);

    static void setupTasks(madrona::TaskGraph::Builder &builder);

    Sim(Engine &ctx, const WorldInit &init);

    WorldInit initCfg;
    madrona::Entity *rectangles;
    RNG rng;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
