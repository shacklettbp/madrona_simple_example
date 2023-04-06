#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace SimpleExample {

constexpr inline float deltaT = 1.f / 30.f;

constexpr inline CountT numPhysicsSubsteps = 4;

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);
    RigidBodyPhysicsSystem::registerTypes(registry);
    render::RenderingSystem::registerTypes(registry);

    registry.registerComponent<Action>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, Action>(1);
    registry.exportColumn<Agent, Position>(2);
}

static void resetWorld(Engine &ctx)
{
    if (ctx.data().enableRender) {
        render::RenderingSystem::reset(ctx);
    }

    RigidBodyPhysicsSystem::reset(ctx);

    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    const math::Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (int32_t i = 0; i < ctx.data().numAgents; i++) {
        Entity agent = ctx.data().agents[i];

        math::Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        ctx.getUnsafe<Position>(agent) = pos;
        ctx.getUnsafe<Rotation>(agent) = Quat { 1, 0, 0, 0 };
        ctx.getUnsafe<Velocity>(agent) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.getUnsafe<ExternalForce>(agent) = Vector3::zero();
        ctx.getUnsafe<ExternalTorque>(agent) = Vector3::zero();
        ctx.getUnsafe<broadphase::LeafID>(agent) =
            RigidBodyPhysicsSystem::registerEntity(ctx, agent, ObjectID { 1 });
    }
}

inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    if (!reset.resetNow) {
        return;
    }
    reset.resetNow = false;

    resetWorld(ctx);
}

inline void actionSystem(Engine &, Action &action, Position &pos)
{
    // Update agent's position
    pos += action.positionDelta;

    // Clear action for next step
    action.positionDelta = Vector3 {0, 0, 0};
}

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &cfg)
{
    auto reset_sys =
        builder.addToGraph<ParallelForNode<Engine, resetSystem, WorldReset>>({});

    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
        Action, Position>>({reset_sys});

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
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    agents = (Entity *)rawAlloc(sizeof(Entity) * init.numAgents);
    numAgents = init.numAgents;
    enableRender = cfg.enableRender;

    for (int32_t i = 0; i < init.numAgents; i++) {
        Entity agent = agents[i] = ctx.makeEntityNow<Agent>();

        ctx.getUnsafe<Action>(agent).positionDelta = Vector3 {0, 0, 0};
        ctx.getUnsafe<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.getUnsafe<ObjectID>(agent) = ObjectID { 1 };
        ctx.getUnsafe<ResponseType>(agent) = ResponseType::Dynamic;
    }

    // Initial reset
    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
