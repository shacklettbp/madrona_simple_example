set(SIMULATOR_SRCS
    sim.hpp sim.cpp
)

add_library(simple_example_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(simple_example_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
)

add_library(simple_example_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(simple_example_mgr PRIVATE
    madrona_python_utils
    madrona_common
    madrona_importer
    simple_example_cpu_impl
    madrona_mw_cpu
    madrona_physics_assets
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            simple_example_gpu_srcs
        SOURCES_DEFN
            SIMPLE_SRC_LIST
        FLAGS_DEFN
            SIMPLE_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(simple_example_mgr PRIVATE
        simple_example_gpu_srcs
        madrona_mw_gpu
    )
endif()

target_compile_definitions(simple_example_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(madrona_simple_example_python
    bindings.cpp
)

target_link_libraries(madrona_simple_example_python PRIVATE
    madrona_hdrs
    simple_example_mgr
)
