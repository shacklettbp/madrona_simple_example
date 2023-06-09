#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace SimpleExample {

constexpr inline float deltaT = 1.f / 30.f;
constexpr inline CountT numPhysicsSubsteps = 1;

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<MoveAction>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();
    registry.registerArchetype<Obstacle>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, MoveAction>(1);
    registry.exportColumn<Agent, Position>(2);
}

static void generateWorld(Engine &ctx, CountT num_obstacles)
{
    RigidBodyPhysicsSystem::reset(ctx);

    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    // Randomly place obstacles
    const math::Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (CountT i = 0; i < num_obstacles; i++) {
        Entity e = ctx.data().obstacles[i] = ctx.makeEntity<Obstacle>();

        math::Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        ctx.get<Position>(e) = pos;
        ctx.get<Rotation>(e) = Quat { 1, 0, 0, 0 };
        ctx.get<Scale>(e) = Diag3x3 { 1, 1, 1 };
        ctx.get<Velocity>(e) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ObjectID>(e) = ObjectID { 1 };
        ctx.get<ResponseType>(e) = ResponseType::Dynamic;
        ctx.get<ExternalForce>(e) = Vector3::zero();
        ctx.get<ExternalTorque>(e) = Vector3::zero();
        ctx.get<broadphase::LeafID>(e) =
            RigidBodyPhysicsSystem::registerEntity(ctx, e, ObjectID { 1 });
    }

    ctx.get<broadphase::LeafID>(ctx.data().plane) =
        RigidBodyPhysicsSystem::registerEntity(ctx, ctx.data().plane,
                                               ObjectID { 0 });

    // Reset the position of the agent
    Entity agent = ctx.data().agent;
    ctx.get<Position>(agent) = Vector3 { 0, 0, 1 };
    ctx.get<Rotation>(agent) = Quat { 1, 0, 0, 0 };
    ctx.get<Velocity>(agent) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ExternalForce>(agent) = Vector3::zero();
    ctx.get<ExternalTorque>(agent) = Vector3::zero();
    ctx.get<broadphase::LeafID>(agent) =
        RigidBodyPhysicsSystem::registerEntity(ctx, agent, ObjectID { 1 });
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (!reset.resetNow) {
        return;
    }
    reset.resetNow = false;

    const CountT num_obstacles = ctx.data().numObstacles;

    // Delete old obstacles
    for (CountT i = 0; i < num_obstacles; i++) {
        ctx.destroyEntity(ctx.data().obstacles[i]);
    }

    generateWorld(ctx, num_obstacles);
}

inline void actionSystem(Engine &,
                         MoveAction action,
                         const Rotation &agent_rot,
                         ExternalForce &agent_force,
                         ExternalTorque &agent_torque)
{
    // Translate from discrete actions to forces
    switch (action) {
    case MoveAction::Wait: {
        // Do nothing
    } break;
    case MoveAction::Forward: {
        agent_force += agent_rot.rotateVec(math::fwd);
    } break;
    case MoveAction::Left: {
        agent_force -= agent_rot.rotateVec(math::right);
    } break;
    case MoveAction::Right: {
        agent_force += agent_rot.rotateVec(math::right);
    } break;
    case MoveAction::TurnLeft: {
        agent_torque -= agent_rot.rotateVec(math::up);
    } break;
    case MoveAction::TurnRight: {
        agent_torque += agent_rot.rotateVec(math::up);
    } break;
    default: __builtin_unreachable();
    }
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
                                    deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &cfg)
{
    auto reset_sys =
        builder.addToGraph<ParallelForNode<Engine, resetSystem, WorldReset>>({});


#ifndef MADRONA_GPU_MODE
    auto post_reset = reset_sys;
#else
    // in GPU mode, need to queue up systems that sort and compact the ECS
    // tables in order to reclaim memory
    auto sort_obstacles = queueSortByWorld<Obstacle>(builder, {reset_sys});
    auto sort_agent = queueSortByWorld<Obstacle>(builder, {sort_obstacles});

    auto post_reset = sort_agent;
#endif

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        MoveAction, Rotation, ExternalForce, ExternalTorque>>({post_reset});

    auto bvh_sys = RigidBodyPhysicsSystem::setupBroadphaseTasks(
        builder, {action_sys});

    auto substep_sys = RigidBodyPhysicsSystem::setupSubstepTasks(
        builder, {bvh_sys}, numPhysicsSubsteps);

    auto phys_cleanup_sys = RigidBodyPhysicsSystem::setupCleanupTasks(
        builder, {substep_sys});

    auto sim_done = phys_cleanup_sys;

    if (cfg.enableRender) {
        sim_done = render::RenderingSystem::setupTasks(builder, {sim_done});
    }

#ifdef MADRONA_GPU_MODE
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({sim_done});
    (void)recycle_sys;
#endif
}


Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    if (ctx.data().enableRender) {
        render::RenderingSystem::reset(ctx);
    }

    RigidBodyPhysicsSystem::init(ctx, init.rigidBodyObjMgr, deltaT,
                                 numPhysicsSubsteps, -9.8 * math::up,
                                 init.numObstacles + 2,
                                 init.numObstacles * init.numObstacles / 2,
                                 10);

    // Make a buffer that will last the duration of simulation for storing
    // obstacle entity IDs
    obstacles = (Entity *)rawAlloc(sizeof(Entity) * init.numObstacles);
    numObstacles = init.numObstacles;
    enableRender = cfg.enableRender;

    agent = ctx.makeEntity<Agent>();
    ctx.get<MoveAction>(agent) = MoveAction::Wait;
    ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(agent) = ObjectID { 1 };
    ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
    ctx.get<render::ViewSettings>(agent) =
        render::RenderingSystem::setupView(ctx, 90.f, 0.001f,
                                           math::up * 0.5f, { 0 });

    // Create ground plane during initialization
    plane = ctx.makeEntity<Obstacle>();
    ctx.get<Position>(plane) = Vector3::zero();
    ctx.get<Rotation>(plane) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(plane) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(plane) = ObjectID { 0 };
    ctx.get<ResponseType>(plane) = ResponseType::Static;

    generateWorld(ctx, numObstacles);
    ctx.singleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
