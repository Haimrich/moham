#ifndef MOHAM_MOHAM_H_
#define MOHAM_MOHAM_H_

#include "nsga.h"

#include <iostream>
#include <vector>
#include <unordered_map>

#include "config.h"
#include "graph.h"
#include "archtemplate.h"
#include "medea.h"

namespace moham {

    struct MemoryInterface {
        typedef std::size_t ID;

        ID id;
        std::pair<int,int> mesh_coordinates;
    };


    struct ArchInstance 
    {
        typedef std::size_t ID;

        ID id;
        ArchTemplate::ID arch_template_id;
        MinimalArchSpecs arch;
        timeloop::EngineSpecs arch_specs;
        double area;
        bool need_evaluation;
        unsigned num_assigned_layers;

        std::pair<int,int> mesh_coordinates;
        MemoryInterface::ID memory_interface_id;
    };


    class Moham : public NSGA
    {
    public:
        struct LayerEval {
            double energy, cycles, area, required_bandwidth;
            double start_time, end_time;
            bool need_evaluation;
        };

        struct Gene {
            double priority;
            Medea::Individual::ID mapping_id;
            ArchInstance::ID acc_instance_id;
        };

        struct BottleneckSegment {
            double start_time, end_time;
            ArchInstance::ID acc;
        };

        

        struct Individual : NSGA::Individual<3> 
        {
            std::vector<Gene> genome;

            std::vector<LayerEval> layer_evals;

            std::vector<ArchInstance> accelerators;
            std::vector<MemoryInterface> memory_interfaces;
            std::pair<int,int> mesh_dimensions;

            std::vector<Layer::ID> toposort;
            std::vector<BottleneckSegment> nip_bottlenecks;
            std::vector<BottleneckSegment> memory_bottlenecks;
        };

        typedef std::vector<Individual> Population;

    private:

        Config config_;
        Graph* graph_;
        std::vector<ArchTemplate>* arch_templates_;
        std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings_;

        std::vector<std::unordered_map<ArchTemplate::ID,std::array<std::pair<double,double>,3>>> workload_mappings_objectives_bounds_;
        std::vector<std::unordered_map<ArchTemplate::ID,std::vector<std::array<double,3>>>> workload_mappings_normalized_objectives_;

        Population population_, parent_population_, immigrant_population_, merged_population_;

        std::vector<std::mt19937_64> rng_;
        std::uniform_real_distribution<double> uni_distribution_;

    public:

        Moham(
            Config config, 
            Graph* graph, 
            std::vector<ArchTemplate>* arch_templates, 
            std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings
        );

        void Run();

        void RunRandom();
        
    private:

        Individual RandomIndividual();

        double RandomPopulation(Population& population);

        void InjectHeuristicallyGoodIndividuals(Population& population);
        
        std::vector<Layer::ID> GenerateTopologicalSorting(const Individual& individual);

        void Evaluate(Individual& individual);

        bool CheckIndividualValidity(Individual& individual, bool do_assert = true) const;

        void CalculatePackageMesh(Individual& individual);
        double EvaluateMakespan(Individual& individual, const std::vector<Layer::ID>& topological_sorting);
        double EvaluateNipEnergy(Individual& individual);

        void NegotiateArchitectures(Individual &individual);

        void Crossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b);
        void PriorityCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng);
        void SubAcceleratorCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng);
        void MappingCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng);
        
        void Mutation(Individual& ind);
        void SplittingMutation(Individual& ind, std::mt19937_64& rng);
        void MergingMutation(Individual& ind, std::mt19937_64& rng);
        void TemplateMutation(Individual& ind, std::mt19937_64& rng);
        void PositionMutation(Individual& ind, std::mt19937_64& rng);
        void PriorityMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng);
        void MappingMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng);
        void AssignmentMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng);

        void RemoveEmptySubAccelerators(Individual& ind) const;

        Medea::Individual::ID ConvertMappingToTemplate(Workload::ID workload_id, Medea::Individual::ID mapping_id, ArchTemplate::ID start_template, ArchTemplate::ID destination_template) const;

        std::size_t Tournament(const Population& pop);
        
        void OutputParetoScheduling(std::ofstream &out, const Population& pop);
        void OutputParetoNipBottlenecks(std::ofstream &out, const Population& pop);
        
        void AppendGenerationInfoToFile(std::ofstream &out, Population &pop, unsigned gen_id);

        void AssignBreadthFirstPriorities(Individual& ind);

        void OutputParetoFrontFiles(std::string out_path);
    };
}


#endif // MOHAM_MOHAM_H_