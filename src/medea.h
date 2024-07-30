#ifndef MOHAM_MEDEA_H_
#define MOHAM_MEDEA_H_

#include "nsga.h"

#include <vector>
#include <random>

#include "common.h"
#include "graph.h"
#include "config.h"
#include "timeloop.h"
#include "accelergy.h"
#include "archtemplate.h"
#include "mapping.h"
#include "minimalarchspecs.h"

namespace moham
{

  class Medea : public NSGA
  {
  public:
  
    struct Individual : public NSGA::Individual<3>
    {
      Mapping genome;
      timeloop::Engine engine;
      MinimalArchSpecs arch;
      ArchTemplate::ID arch_template_id;

      typedef std::size_t ID;
      friend std::ostream &operator<<(std::ostream &out, const Individual &ind);
    };

    typedef std::vector<Individual> Population;

  protected:
    Config config_;
    
    Workload workload_;
    ArchTemplate& arch_template_;

    timeloop::MapSpace *global_mapspace_, *mapspace_;
    timeloop::Constraints* constraints_; // TODO fix bug

    Population population_, parent_population_, immigrant_population_, merged_population_;

    RandomGenerator128 *if_rng_, *lp_rng_, *db_rng_, *sp_rng_;

    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uni_distribution_;
    std::exponential_distribution<double> exp_distribution_;
    std::uniform_int_distribution<uint64_t> tour_distribution_;

  public:
  
    Medea(Config config, Workload &workload, ArchTemplate &arch_template);

    ~Medea();

    Population Run(std::string medea_out_path);

    Population Parse(std::string pareto_path);

  protected:
    // void InjectUserDefinedMapping(Population &pop, uint32_t id);

    std::vector<loop::Descriptor> GetSubnestAtLevel(const Mapping &mapping, unsigned level);
    LoopRange GetSubnestRangeAtLevel(const Mapping &mapping, unsigned level);
    uint64_t GetParallelAtLevel(const Mapping &mapping, spacetime::Dimension dim, uint64_t level);
    uint64_t GetDimFactorInSubnest(timeloop::Shape::FlattenedDimensionID dimension, std::vector<loop::Descriptor> &subnest);
    uint64_t GetStrideInSubnest(timeloop::Shape::FlattenedDimensionID dimension, std::vector<loop::Descriptor> &subnest);

    bool Evaluate(Mapping mapping, Individual &individual);
    bool EngineSuccess(std::vector<model::EvalStatus> &status_per_level);
    void UpdateArchitecture(Mapping& mapping, model::Engine& engine, MinimalArchSpecs& arch);

    void Crossover(const Mapping &parent_a, const Mapping &parent_b, Mapping &offspring_a, Mapping &offspring_b);

    void Mutation(Individual &individual);
    void RandomMutation(Mapping &mapping);    
    void FillMutation(model::Engine &engine, Mapping &mapping);
    void FanoutMutation(Mapping &mapping);

    double RandomPopulation(Population &population);
    void RandomIndividual(uint32_t p, Population &population);
    bool RandomMapping(Mapping *mapping);

    void FactorCompensation(const timeloop::Shape::FlattenedDimensionID &dim, const uint64_t stride, const uint64_t old_factor, const uint64_t new_factor, const uint64_t level, loop::Nest &nest);

    uint64_t Tournament();

    void AppendGenerationInfoToFile(std::ofstream &out, Population &pop, uint64_t gen_id);

    void OutputParetoFrontFiles(std::string out_path);
  };

}

#endif // MOHAM_MEDEA_H_