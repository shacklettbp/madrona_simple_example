#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

using namespace madrona;
using namespace madrona::py;

namespace SimpleExample {

struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;

    inline Impl(const Config &c, EpisodeManager *ep_mgr)
        : cfg(c),
          episodeMgr(ep_mgr)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;
    virtual Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                Span<const int64_t> dims) = 0;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
    ExecT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(mgr_cfg, episode_mgr),
          cpuExec({
                  .numWorlds = mgr_cfg.numWorlds,
                  .renderWidth = 0,
                  .renderHeight = 0,
                  .numExportedBuffers = num_exported_buffers,
                  .cameraMode = render::CameraMode::None,
              }, sim_cfg, world_inits)
    {}

    inline virtual ~CPUImpl() final {
        delete episodeMgr;
    }

    inline virtual void run() final { cpuExec.run(); }
    
    inline virtual Tensor exportTensor(CountT slot,
                                       Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = cpuExec.getExported(slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::GPUImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;

    inline GPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(mgr_cfg, episode_mgr),
          gpuExec({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&sim_cfg,
                  .numUserConfigBytes = sizeof(Sim::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = mgr_cfg.numWorlds,
                  .numExportedBuffers = num_exported_buffers, 
                  .gpuID = (uint32_t)mgr_cfg.gpuID,
                  .cameraMode = render::CameraMode::None,
                  .renderWidth = 0,
                  .renderHeight = 0,
              }, {
                  "",
                  { SIMPLE_SRC_LIST },
                  { SIMPLE_COMPILE_FLAGS },
                  CompileConfig::OptMode::LTO,
                  CompileConfig::Executor::TaskGraph,
              })
    {}

    inline virtual ~GPUImpl() final {
        REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void run() final { gpuExec.run(); }
    
    virtual inline Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
        void *dev_ptr = gpuExec.getExported(slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
                                               int64_t num_agents_per_world,
                                               EpisodeManager *episode_mgr)
{
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            int32_t(num_agents_per_world),
        };
    }

    return world_inits;
}

Manager::Impl * Manager::Impl::init(const Config &cfg)
{
    const int64_t num_exported_buffers = 3;

    switch (cfg.execMode) {
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            cfg.numAgentsPerWorld, episode_mgr);

        return new CPUImpl(cfg, {}, episode_mgr, world_inits.data(),
                       num_exported_buffers);
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        FATAL("CUDA support not compiled in!");
#else
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        // Set the current episode count to 0
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds,
            cfg.numAgentsPerWorld, episode_mgr);

        return new GPUImpl(cfg, {}, episode_mgr, world_inits.data(),
                           num_exported_buffers);
#endif
    } break;
    default: return nullptr;
    }
}

MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

MADRONA_EXPORT Manager::~Manager() {}

MADRONA_EXPORT void Manager::step()
{
    impl_->run();
}

MADRONA_EXPORT Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(0, Tensor::ElementType::Int32,
                               {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(1, Tensor::ElementType::Float32,
        {impl_->cfg.numWorlds, impl_->cfg.numAgentsPerWorld, 3});
}

MADRONA_EXPORT Tensor Manager::positionTensor() const
{
    return impl_->exportTensor(2, Tensor::ElementType::Float32,
        {impl_->cfg.numWorlds, impl_->cfg.numAgentsPerWorld, 3});
}

}
