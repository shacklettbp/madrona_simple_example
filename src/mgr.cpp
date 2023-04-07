#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/physics_assets.hpp>

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
using namespace madrona::math;
using namespace madrona::phys;

namespace SimpleExample {

struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    EpisodeManager *episodeMgr;

    inline Impl(const Config &c, PhysicsLoader &&physics_loader,
                EpisodeManager *ep_mgr)
        : cfg(c),
          physicsLoader(std::move(physics_loader)),
          episodeMgr(ep_mgr)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;
    virtual Tensor exportTensor(CountT slot, Tensor::ElementType type,
                                Span<const int64_t> dims) = 0;

    virtual Tensor depthTensor() = 0;
    virtual Tensor rgbTensor() = 0;

    static inline Impl * init(const Config &cfg);
};

struct Manager::CPUImpl final : Manager::Impl {
    using ExecT = TaskGraphExecutor<Engine, Sim, Sim::Config, WorldInit>;
    ExecT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   PhysicsLoader &&physics_loader,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(mgr_cfg, std::move(physics_loader), episode_mgr),
          cpuExec({
                  .numWorlds = mgr_cfg.numWorlds,
                  .maxViewsPerWorld = 1,
                  .maxInstancesPerWorld = 1000,
                  .renderWidth = mgr_cfg.renderWidth,
                  .renderHeight = mgr_cfg.renderHeight,
                  .maxObjects = 2,
                  .numExportedBuffers = num_exported_buffers,
                  .cameraMode = mgr_cfg.enableRender ?
                      render::CameraMode::Perspective :
                      render::CameraMode::None,
                  .renderGPUID = mgr_cfg.gpuID,
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

    virtual Tensor depthTensor() final
    {
        return Tensor(
            cpuExec.depthObservations(), Tensor::ElementType::Float32,
            { cfg.numWorlds, cfg.renderHeight, cfg.renderWidth, 1 },
#ifdef MADRONA_LINUX
            cfg.gpuID
#else
            Optional<int>::none()
#endif
        );
    }

    virtual Tensor rgbTensor() final
    {
        return Tensor(
            cpuExec.rgbObservations(), Tensor::ElementType::UInt8,
            { cfg.numWorlds, cfg.renderHeight, cfg.renderWidth, 4 },
#ifdef MADRONA_LINUX
            cfg.gpuID
#else
            Optional<int>::none()
#endif
        );
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::GPUImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;

    inline GPUImpl(const Manager::Config &mgr_cfg,
                   const Sim::Config &sim_cfg,
                   PhysicsLoader &&physics_loader,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits,
                   uint32_t num_exported_buffers)
        : Impl(mgr_cfg, std::move(physics_loader), episode_mgr),
          gpuExec({
                  .worldInitPtr = world_inits,
                  .numWorldInitBytes = sizeof(WorldInit),
                  .userConfigPtr = (void *)&sim_cfg,
                  .numUserConfigBytes = sizeof(Sim::Config),
                  .numWorldDataBytes = sizeof(Sim),
                  .worldDataAlignment = alignof(Sim),
                  .numWorlds = mgr_cfg.numWorlds,
                  .maxViewsPerWorld = 1,
                  .numExportedBuffers = num_exported_buffers, 
                  .gpuID = (uint32_t)mgr_cfg.gpuID,
                  .cameraMode = mgr_cfg.enableRender ?
                      render::CameraMode::Perspective :
                      render::CameraMode::None,
                  .renderWidth = mgr_cfg.renderWidth,
                  .renderHeight = mgr_cfg.renderHeight,
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

    virtual Tensor depthTensor() final
    {
        return Tensor(
            gpuExec.depthObservations(), Tensor::ElementType::Float32,
            { cfg.numWorlds, cfg.renderHeight, cfg.renderWidth, 1 },
            cfg.gpuID);
    }

    virtual Tensor rgbTensor() final
    {
        return Tensor(
            gpuExec.rgbObservations(), Tensor::ElementType::UInt8,
            { cfg.numWorlds, cfg.renderHeight, cfg.renderWidth, 4 },
            cfg.gpuID);
    }
};
#endif

static void loadPhysicsObjects(PhysicsLoader &loader)
{
    DynArray<RigidBodyMetadata> metadatas(0);
    DynArray<AABB> aabbs(0);
    DynArray<CollisionPrimitive> prims(0);
    
    { // Plane: (0)
        metadatas.push_back({
            .invInertiaTensor = { 0.f, 0.f, 0.f },
            .invMass = 0.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        aabbs.push_back({
            .pMin = { -FLT_MAX, -FLT_MAX, -FLT_MAX },
            .pMax = { FLT_MAX, FLT_MAX, 0 },
        });

        prims.push_back({
            .type = CollisionPrimitive::Type::Plane,
            .plane = {},
        });
    }

    { // Cube: (1)
        metadatas.push_back({
            .invInertiaTensor = { 1.5f, 1.5f, 1.5f },
            .invMass = 1.f,
            .muS = 0.5f,
            .muD = 0.5f,
        });

        PhysicsLoader::LoadedHull cube_hull = loader.loadHullFromDisk(
            (std::filesystem::path(DATA_DIR) / "cube_collision.obj").c_str());

        aabbs.push_back(cube_hull.aabb);

        prims.push_back({
            .type = CollisionPrimitive::Type::Hull,
            .hull = {
                .halfEdgeMesh = cube_hull.collisionMesh,
            },
        });
    } 

    loader.loadObjects(metadatas.data(), aabbs.data(),
                       prims.data(), metadatas.size());
}

static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
                                               int64_t num_obstacles,
                                               ObjectManager *phys_obj_mgr,
                                               EpisodeManager *episode_mgr)
{
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
        world_inits[i] = WorldInit {
            episode_mgr,
            phys_obj_mgr,
            int32_t(num_obstacles),
        };
    }

    return world_inits;
}

Manager::Impl * Manager::Impl::init(const Config &mgr_cfg)
{
    // NEED to increase this if exporting more tensors from the simulator
    const int64_t num_exported_buffers = 3;

    // Load renderer meshes
    DynArray<imp::ImportedObject> imported_renderer_objs(0);

    {
        auto plane_obj = imp::ImportedObject::importObject(
            (std::filesystem::path(DATA_DIR) / "plane.obj").c_str());

        if (!plane_obj.has_value()) {
            FATAL("Failed to load plane render mesh");
        }

        imported_renderer_objs.emplace_back(std::move(*plane_obj));
    }

    {
        auto cube_obj = imp::ImportedObject::importObject(
            (std::filesystem::path(DATA_DIR) / "cube_render.obj").c_str());

        if (!cube_obj.has_value()) {
            FATAL("Failed to load plane cube mesh");
        }

        imported_renderer_objs.emplace_back(std::move(*cube_obj));
    }
    
    HeapArray<imp::SourceObject> renderer_objs(
        imported_renderer_objs.size());

    for (CountT i = 0; i < imported_renderer_objs.size(); i++) {
        renderer_objs[i] = imp::SourceObject {
            imported_renderer_objs[i].meshes,
        };
    }

    Sim::Config sim_cfg {
        mgr_cfg.enableRender,
    };

    switch (mgr_cfg.execMode) {
    case ExecMode::CPU: {
        EpisodeManager *episode_mgr = new EpisodeManager { 0 };

        PhysicsLoader phys_loader(PhysicsLoader::StorageType::CPU, 2);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits = setupWorldInitData(
            mgr_cfg.numWorlds, mgr_cfg.numObstacles,
            phys_obj_mgr, episode_mgr);

        auto *impl = new CPUImpl(mgr_cfg, sim_cfg, std::move(phys_loader),
            episode_mgr, world_inits.data(), num_exported_buffers);

        if (mgr_cfg.enableRender) {
            impl->cpuExec.loadObjects(renderer_objs);
        }

        return impl;
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
        FATAL("CUDA support not compiled in!");
#else
        EpisodeManager *episode_mgr = 
            (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
        // Set the current episode count to 0
        REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));

        PhysicsLoader phys_loader(PhysicsLoader::StorageType::CUDA, 2);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();

        HeapArray<WorldInit> world_inits = setupWorldInitData(
            mgr_cfg.numWorlds, mgr_cfg.numObstacles,
            phys_obj_mgr, episode_mgr);
        
         auto *impl = new GPUImpl(mgr_cfg, sim_cfg, std::move(phys_loader),
            episode_mgr, world_inits.data(), num_exported_buffers);

        if (mgr_cfg.enableRender) {
            impl->gpuExec.loadObjects(renderer_objs);
        }

        return impl;
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
    return impl_->exportTensor(1, Tensor::ElementType::Int32,
        {impl_->cfg.numWorlds, 1});
}

MADRONA_EXPORT Tensor Manager::positionTensor() const
{
    return impl_->exportTensor(2, Tensor::ElementType::Float32,
        {impl_->cfg.numWorlds, 3});
}

MADRONA_EXPORT Tensor Manager::depthTensor() const
{
    return impl_->depthTensor();
}

MADRONA_EXPORT Tensor Manager::rgbTensor() const
{
    return impl_->rgbTensor();
}

}
