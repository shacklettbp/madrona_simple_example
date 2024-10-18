#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

using namespace madrona;
using namespace madrona::math;

namespace madsimple {

void Sim::registerTypes(ECSRegistry &registry, const Config &)
{
    base::registerTypes(registry);

    registry.registerComponent<Reset>();
    registry.registerComponent<Action>();
    registry.registerComponent<GridPos>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<CurStep>();

    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportColumn<Agent, Reset>((uint32_t)ExportID::Reset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, GridPos>((uint32_t)ExportID::GridPos);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>((uint32_t)ExportID::Done);
}

inline void tick(Engine &ctx,
                 Action &action,
                 Reset &reset,
                 GridPos &grid_pos,
                 Reward &reward,
                 Done &done,
                 CurStep &episode_step)
{
    const GridState *grid = ctx.data().grid;

    GridPos new_pos = grid_pos;

    switch (action) {
        case Action::Up: {
            new_pos.y += 1;
        } break;
        case Action::Down: {
            new_pos.y -= 1;
        } break;
        case Action::Left: {
            new_pos.x -= 1;
        } break;
        case Action::Right: {
            new_pos.x += 1;
        } break;
        default: break;
    }

    action = Action::None;

    if (new_pos.x < 0) {
        new_pos.x = 0;
    }

    if (new_pos.x >= grid->width) {
        new_pos.x = grid->width - 1;
    }

    if (new_pos.y < 0) {
        new_pos.y = 0;
    }

    if (new_pos.y >= grid->height) {
        new_pos.y = grid->height -1;
    }


    {
        const Cell &new_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

        if ((new_cell.flags & CellFlag::Wall)) {
            new_pos = grid_pos;
        }
    }

    const Cell &cur_cell = grid->cells[new_pos.y * grid->width + new_pos.x];

    bool episode_done = false;
    if (reset.resetNow != 0) {
        reset.resetNow = 0;
        episode_done = true;
    }

    if ((cur_cell.flags & CellFlag::End)) {
        episode_done = true;
    }

    uint32_t cur_step = episode_step.step;

    if (cur_step == ctx.data().maxEpisodeLength - 1) {
        episode_done = true;
    }

    if (episode_done) {
        done.episodeDone = 1.f;

        new_pos = GridPos {
            grid->startY,
            grid->startX,
        };

        episode_step.step = 0;
    } else {
        done.episodeDone = 0.f;
        episode_step.step = cur_step + 1;
    }

    // Commit new position
    grid_pos = new_pos;
    reward.r = cur_cell.reward;
}

void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(0);
    builder.addToGraph<ParallelForNode<Engine, tick,
        Action, Reset, GridPos, Reward, Done, CurStep>>({});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr),
      grid(init.grid),
      maxEpisodeLength(cfg.maxEpisodeLength)
{
    Entity agent = ctx.makeEntity<Agent>();
    ctx.get<Action>(agent) = Action::None;
    ctx.get<GridPos>(agent) = GridPos {
        grid->startY,
        grid->startX,
    };
    ctx.get<Reward>(agent).r = 0.f;
    ctx.get<Done>(agent).episodeDone = 0.f;
    ctx.get<CurStep>(agent).step = 0;
}

MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, WorldInit);

}
