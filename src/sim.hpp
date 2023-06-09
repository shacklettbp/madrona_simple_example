#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/mw.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace SimpleExample {

// 3D Position & Quaternion Rotation
// These classes are defined in include/madrona/components.hpp
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::phys::Velocity;
using madrona::phys::ResponseType;
using madrona::phys::ExternalForce;
using madrona::phys::ExternalTorque;

class Engine;

struct WorldReset {
    int32_t resetNow;
};

enum class MoveAction : int32_t {
    Wait = 0,
    Forward = 1,
    Left = 2,
    Right = 3,
    TurnLeft = 4,
    TurnRight = 5,
};

// A pure physics obstacle with no other components
struct Obstacle : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID
> {};

// Same as Obstacle but with an additional action and camera components
struct Agent : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    Velocity,
    ObjectID,
    ResponseType,
    madrona::phys::solver::SubstepPrevState,
    madrona::phys::solver::PreSolvePositional,
    madrona::phys::solver::PreSolveVelocity,
    ExternalForce,
    ExternalTorque,
    madrona::phys::broadphase::LeafID,
    MoveAction,
    madrona::render::ViewSettings
> {};

struct Sim : public madrona::WorldBase {
    struct Config {
        bool enableRender;
    };

    static void registerTypes(madrona::ECSRegistry &registry,
                              const Config &cfg);

    static void setupTasks(madrona::TaskGraph::Builder &builder,
                           const Config &cfg);

    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    madrona::Entity agent;
    madrona::Entity plane;

    madrona::Entity *obstacles;
    int32_t numObstacles;

    bool enableRender;
};

class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
};

}
