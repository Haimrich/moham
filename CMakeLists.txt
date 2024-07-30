cmake_minimum_required(VERSION 3.14)
set (CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Werror -Wall -Wextra -fmax-errors=1 -pthread -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -flto")
add_compile_definitions(USE_ACCELERGY BUILD_BASE_DIR="${CMAKE_SOURCE_DIR}/lib/timeloop")

project(moham)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

include_directories(
    ${PROJECT_SOURCE_DIR}/src 
    ${PROJECT_SOURCE_DIR}/lib/timeloop/include
    ${PROJECT_SOURCE_DIR}/lib/timeloop/src
    ${PROJECT_SOURCE_DIR}/lib/timeloop/pat-public/src
)

set(TIMELOOP_SRC_DIR ${PROJECT_SOURCE_DIR}/lib/timeloop/src)
set(TIMELOOP_SRCS
    "${TIMELOOP_SRC_DIR}/loop-analysis/coordinate-space-tile-info.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/loop-state.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/tiling.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/tiling-tile-info.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/point.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/point-set-aahr.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/point-set-multi-aahr.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/nest-analysis-tile-info.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/nest-analysis.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/aahr-carve.cpp"
    "${TIMELOOP_SRC_DIR}/sparse-analysis/state.cpp"
    "${TIMELOOP_SRC_DIR}/sparse-analysis/sparse-analysis.cpp"
    "${TIMELOOP_SRC_DIR}/sparse-analysis/compute-gs-analyzer.cpp"
    "${TIMELOOP_SRC_DIR}/sparse-analysis/storage-gs-analyzer.cpp"
    "${TIMELOOP_SRC_DIR}/sparse-analysis/representation-analyzer.cpp"
    "${TIMELOOP_SRC_DIR}/sparse-analysis/storage-optimization-combiner.cpp"
    "${TIMELOOP_SRC_DIR}/loop-analysis/operation-type.cpp"
    "${TIMELOOP_SRC_DIR}/../pat-public/src/pat/pat.cpp"
    "${TIMELOOP_SRC_DIR}/mapping/loop.cpp"
    "${TIMELOOP_SRC_DIR}/mapping/nest.cpp"
    "${TIMELOOP_SRC_DIR}/model/model-base.cpp"
    "${TIMELOOP_SRC_DIR}/model/engine.cpp"
    "${TIMELOOP_SRC_DIR}/model/level.cpp"
    "${TIMELOOP_SRC_DIR}/model/arithmetic.cpp"
    "${TIMELOOP_SRC_DIR}/model/buffer.cpp"
    "${TIMELOOP_SRC_DIR}/model/topology.cpp"
    "${TIMELOOP_SRC_DIR}/model/network.cpp"
    "${TIMELOOP_SRC_DIR}/model/network-factory.cpp"
    "${TIMELOOP_SRC_DIR}/model/network-legacy.cpp"
    "${TIMELOOP_SRC_DIR}/model/network-reduction-tree.cpp"
    "${TIMELOOP_SRC_DIR}/model/network-simple-multicast.cpp"
    "${TIMELOOP_SRC_DIR}/model/sparse-optimization-info.cpp"
    "${TIMELOOP_SRC_DIR}/model/sparse-optimization-parser.cpp"
    "${TIMELOOP_SRC_DIR}/util/banner.cpp"
    "${TIMELOOP_SRC_DIR}/util/args.cpp"
    "${TIMELOOP_SRC_DIR}/util/numeric.cpp"
    "${TIMELOOP_SRC_DIR}/util/map2d.cpp"
    "${TIMELOOP_SRC_DIR}/workload/shape-models/problem-shape.cpp"
  # "${TIMELOOP_SRC_DIR}/workload/workload.cpp" # see _tl_workload_mt_fix.cc
    "${TIMELOOP_SRC_DIR}/workload/shape-models/operation-space.cpp"
    "${TIMELOOP_SRC_DIR}/workload/density-models/density-distribution.cpp"
    "${TIMELOOP_SRC_DIR}/workload/density-models/density-distribution-factory.cpp"
    "${TIMELOOP_SRC_DIR}/workload/density-models/fixed-structured-distribution.cpp"
    "${TIMELOOP_SRC_DIR}/workload/density-models/hypergeometric-distribution.cpp"
    "${TIMELOOP_SRC_DIR}/workload/density-models/banded-distribution.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/metadata-format.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/metadata-format-factory.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/run-length-encoding.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/uncompressed-offset-pair.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/coordinate-payload.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/uncompressed-bitmask.cpp"
    "${TIMELOOP_SRC_DIR}/workload/format-models/bitmask.cpp"
    "${TIMELOOP_SRC_DIR}/compound-config/compound-config.cpp"
    "${TIMELOOP_SRC_DIR}/data/cnn/cnn-layers.cpp"
    "${TIMELOOP_SRC_DIR}/data/cnn/cnn-densities.cpp"
    "${TIMELOOP_SRC_DIR}/mapping/mapping.cpp"
    "${TIMELOOP_SRC_DIR}/mapping/parser.cpp"
    "${TIMELOOP_SRC_DIR}/mapping/arch-properties.cpp"
    "${TIMELOOP_SRC_DIR}/mapping/constraints.cpp"
    "${TIMELOOP_SRC_DIR}/mapspaces/mapspace-base.cpp"
    "${TIMELOOP_SRC_DIR}/mapspaces/mapspace-factory.cpp"
    "${TIMELOOP_SRC_DIR}/mapspaces/subspaces.cpp"
    "${TIMELOOP_SRC_DIR}/mapspaces/uber.cpp"
)

file(GLOB_RECURSE MOHAM_SRCS CONFIGURE_DEPENDS 
        "${PROJECT_SOURCE_DIR}/src/*.cc"
        "${PROJECT_SOURCE_DIR}/include/*.cc"
    )
set_source_files_properties("${PROJECT_SOURCE_DIR}/src/onnx.pb.cc" PROPERTIES COMPILE_FLAGS -Wno-array-bounds)

add_executable(moham ${MOHAM_SRCS} ${TIMELOOP_SRCS})

# LIBCONFIG++
find_library(CONFIG++_LIBRARY NAMES config++ PATH ${LIBDIR})
include_directories(${INCLUDEDIR})

# YAML
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# PYTHON
#find_package(PythonLibs 3 REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})

# BOOST
set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS iostreams serialization program_options filesystem) # python
message(STATUS "Boost version: ${Boost_VERSION}") 
include_directories(${Boost_INCLUDE_DIRS}) 

# PROTOBUF
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# OPENMP
find_package(OpenMP REQUIRED)

# Linking
#target_link_libraries(moham PRIVATE ${PYTHON_LIBRARIES})
target_link_libraries(moham PRIVATE ${CONFIG++_LIBRARY})
target_link_libraries(moham PRIVATE ${YAML_CPP_LIBRARIES})
target_link_libraries(moham PRIVATE ${Boost_LIBRARIES})
target_link_libraries(moham PRIVATE ${Protobuf_LIBRARIES})
target_link_libraries(moham PUBLIC OpenMP::OpenMP_CXX)
target_link_options(moham PRIVATE -static-libgcc -static-libstdc++)

install(TARGETS moham RUNTIME DESTINATION bin)