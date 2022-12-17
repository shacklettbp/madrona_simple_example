#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::phys;
using namespace madrona::math;

namespace SimpleExample {

constexpr inline float deltaT = 1.f / 30.f;

void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);

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
    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    ctx.data().rng = RNG::make(episode_idx);

    const math::Vector2 bounds { -10.f, 10.f };
    float bounds_diff = bounds.y - bounds.x;

    for (CountT i = 0; i < max_instances; i++) {
        Entity agent = ctx.data().agents[i];

        math::Vector3 pos {
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            bounds.x + ctx.data().rng.rand() * bounds_diff,
            1.f,
        };

        ctx.getUnsafe<Position>(agent) = pos;
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

inline void actionSystem(Engine &, const Action &action, Position &pos)
{
    // Update agent's position
    pos += action.positionDelta;
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys =
        builder.parallelForNode<Engine, resetSystem, WorldReset>({});

    auto action_sys = builder.parallelForNode<Engine, actionSystem,
        Action, Position>({reset_sys});

    (void)action_sys;

    printf("Setup done\n");
}


Sim::Sim(Engine &ctx, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    agents = (Entity *)rawAlloc(sizeof(Entity) * init.numAgents);

    for (CountT i = 0; i < init.numAgents; i++) {
        agents[i] = ctx.makeEntityNow<DynamicObject>();
    }

    // Initial reset
    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
