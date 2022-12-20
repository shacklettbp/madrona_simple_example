#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace SimpleExample {

constexpr inline float deltaT = 1.f / 30.f;

void Sim::registerTypes(ECSRegistry &registry)
{
    base::registerTypes(registry);

    registry.registerComponent<Action>();
    registry.registerComponent<OverlapInfo>();

    registry.registerSingleton<WorldReset>();

    registry.registerArchetype<Rectangle>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Rectangle, Action>(1);
    registry.exportColumn<Rectangle, Position>(2);
    registry.exportColumn<Rectangle, OverlapInfo>(3);
}

static void resetWorld(Engine &ctx)
{
    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().initCfg.episodeMgr;
    uint32_t episode_idx =
        episode_mgr.curEpisode.fetch_add(1, std::memory_order_relaxed);
    RNG &rng = ctx.data().rng = RNG::make(episode_idx);
    WorldInit &init_cfg = ctx.data().initCfg;

    int32_t width_diff =
        init_cfg.maxWidth - init_cfg.minWidth;
    int32_t height_diff =
        init_cfg.maxHeight - init_cfg.minHeight;

    float bounds_x_diff =
        init_cfg.boundsMaxX - init_cfg.boundsMinX;

    float bounds_y_diff =
        init_cfg.boundsMaxY - init_cfg.boundsMinY;

    for (int32_t i = 0; i < init_cfg.numRectangles; i++) {
        Entity rect = ctx.data().rectangles[i];

        math::Vector3 pos {
            init_cfg.boundsMinX + rng.rand() * bounds_x_diff,
            init_cfg.boundsMinY + rng.rand() * bounds_y_diff,
            1.f,
        };

        math::Vector3 scale {
            float(init_cfg.minWidth + int32_t(rng.rand() * width_diff)),
            float(init_cfg.minHeight + int32_t(rng.rand() * height_diff)),
            1.f,
        };

        ctx.getUnsafe<Position>(rect) = pos;
        ctx.getUnsafe<Scale>(rect) = scale;
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

inline void findOverlapsSystem(Engine &ctx,
                               const Entity &e,
                               const Position &pos,
                               const Scale &scale,
                               OverlapInfo &overlap_info)
{
    overlap_info.numOverlaps = 0;

    float width = scale.x;
    float height = scale.y;

    float min_x = pos.x - width / 2.f;
    float max_x = pos.x + width / 2.f;
    float min_y = pos.y - height / 2.f;
    float max_y = pos.y + height / 2.f;

    Entity *rectangles = ctx.data().rectangles;
    int32_t num_rectangles = ctx.data().initCfg.numRectangles;

    for (int32_t i = 0; i < num_rectangles; i++) {
        const Entity &o = rectangles[i];
        if (e == o) continue;

        const Vector3 &o_pos = ctx.getUnsafe<Position>(o);
        const Vector3 &o_dims = ctx.getUnsafe<Scale>(o);

        float o_width = o_dims.x;
        float o_height = o_dims.y;

        float o_min_x = o_pos.x - o_width / 2.f;
        float o_max_x = o_pos.x + o_width / 2.f;

        float o_min_y = o_pos.y - o_height / 2.f;
        float o_max_y = o_pos.y + o_height / 2.f;
        
        bool overlaps = min_x < o_max_x && o_min_x < max_x &&
               min_y < o_max_y && o_min_y < max_y;

        if (overlaps) {
            overlap_info.numOverlaps += 1;
        }
    }
}

void Sim::setupTasks(TaskGraph::Builder &builder)
{
    auto reset_sys =
        builder.parallelForNode<Engine, resetSystem, WorldReset>({});

    auto action_sys = builder.parallelForNode<Engine, actionSystem,
        Action, Position>({reset_sys});

    auto output_sys = builder.parallelForNode<Engine, findOverlapsSystem,
        Entity, Position, Scale, OverlapInfo>({action_sys});

    // Sort the rectangles by WorldID
    auto sort_sys =
        builder.sortArchetypeNode<Rectangle, WorldID>({output_sys});

    (void)sort_sys;

    printf("Setup done\n");
}


Sim::Sim(Engine &ctx, const WorldInit &init)
    : WorldBase(ctx),
      initCfg(init)
{
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    rectangles = (Entity *)rawAlloc(sizeof(Entity) * initCfg.numRectangles);

    for (int32_t i = 0; i < initCfg.numRectangles; i++) {
        rectangles[i] = ctx.makeEntityNow<Rectangle>();

        ctx.getUnsafe<Action>(rectangles[i]).positionDelta = Vector3 {0, 0, 0};
    }

    // Initial reset
    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, WorldInit);

}
