#ifndef MOHAM_TIMELOOP_H_
#define MOHAM_TIMELOOP_H_

#include "compound-config/compound-config.hpp"
#include "model/engine.hpp"
#include "model/sparse-optimization-parser.hpp"
#include "mapping/arch-properties.hpp"
#include "mapping/constraints.hpp"
#include "workload/workload.hpp"
#include "mapspaces/mapspace-factory.hpp"
#include "mapping/parser.hpp"

namespace problem {
    // Fixing multithreading problems for 'problem::shape_' being a global variable.
    extern Shape shape_;
    #pragma omp threadprivate(shape_)
}

namespace timeloop 
{
    using CompoundConfig = config::CompoundConfig; 
    using CompoundConfigNode = config::CompoundConfigNode;

    using ArchProperties = ::ArchProperties;
    using Engine = model::Engine;
    using EngineSpecs = model::Engine::Specs;
    using Topology = model::Topology;
    using TopologySpecs = model::Topology::Specs;

    using SparseOptInfo = sparse::SparseOptimizationInfo;
    namespace Sparse = ::sparse;

    using Workload = problem::Workload;
    using Shape = problem::Shape;
    
    using problem::ParseWorkload;
    constexpr auto GetWorkloadShape = problem::GetShape;

    using MapSpace = mapspace::MapSpace;
    constexpr auto ParseMapSpace = mapspace::ParseAndConstruct;
    using Constraints = mapping::Constraints;

    constexpr auto ParseMapping = mapping::ParseAndConstruct;
}


#endif // MOHAM_TIMELOOP_H_
