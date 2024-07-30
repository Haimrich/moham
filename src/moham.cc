#include "moham.h"

#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <limits>

#include "boost/filesystem.hpp"

#include "common.h"
#include "medea.h"
#include "timeloop.h"
#include "progressbar.h"
#include "archtemplate.h"
#include "graph.h"
#include "config.h"

namespace fs = boost::filesystem;

namespace moham 
{

    Moham::Moham(Config config, Graph* graph, std::vector<ArchTemplate>* arch_templates, std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings) :
        config_(config),
        graph_(graph),
        arch_templates_(arch_templates),
        workload_mappings_(workload_mappings),
        uni_distribution_(0, 1)
    {
        // Inizialization

        parent_population_.resize(config_.moham.population_size);
        population_.resize(config_.moham.population_size);
        immigrant_population_.resize(config_.moham.immigrant_population_size);
        merged_population_.resize(2 * config_.moham.population_size + config_.moham.immigrant_population_size);

        std::random_device rand_dev;
        auto seed = config.seed ? config.seed : rand_dev();
        for (int t = 0; t < omp_get_max_threads(); t++)
            rng_.emplace_back(seed + t);

        if (config.moham.num_threads > 0) {
            omp_set_num_threads(config_.moham.num_threads);
        }

        gen_stability_measures_.resize(config_.moham.stability_window);
        config_.moham.max_subaccelerators = config_.moham.max_subaccelerators > 0 ? config_.moham.max_subaccelerators : graph_->NumLayers();
        
        // Calculating mappings pareto fronts objective bounds for normalization in mapping crossover situations
        workload_mappings_objectives_bounds_.resize(workload_mappings_.size());
        workload_mappings_normalized_objectives_.resize(workload_mappings_.size());

        for (Workload::ID w = 0; w < workload_mappings_.size(); w++)
            for (const auto& template_mappings : workload_mappings_[w])
            {
                for (unsigned o = 0; o < workload_mappings_objectives_bounds_[w][template_mappings.first].size(); o++)
                {
                    auto& bounds = workload_mappings_objectives_bounds_[w][template_mappings.first][o];
                    bounds = std::make_pair(std::numeric_limits<double>::max(), 0.0);
                    for (const auto& ind : template_mappings.second)
                    {
                        bounds.first = std::min(bounds.first, ind.objectives[o]);
                        bounds.second = std::max(bounds.first, ind.objectives[o]);
                    }
                }

                workload_mappings_normalized_objectives_[w].emplace(template_mappings.first, template_mappings.second.size());
                for (Medea::Individual::ID i = 0; i < template_mappings.second.size(); i++)
                    for (unsigned o = 0; o < workload_mappings_normalized_objectives_[w][template_mappings.first][i].size(); o++)
                    {
                        const auto& bounds = workload_mappings_objectives_bounds_[w][template_mappings.first][o];
                        workload_mappings_normalized_objectives_[w][template_mappings.first][i][o] = (template_mappings.second[i].objectives[o] - bounds.first) / (bounds.second - bounds.first);
                    }
            }
        
        // TODO Calculate scheduling space

        std::cout << "🦋 MOHAM Scheduling " << graph_->NumLayers() << " layers.";

        // TODO check that for each layer in the graph there is at least one template that is allowed to execute it.

        // Single Objective Stuff
        single_obj_ = config_.moham.single_obj;
        prod_obj_ = config_.moham.prod_obj;
        weights_obj_ = config_.moham.weights_obj;
    }


    void Moham::Run()
    {  
        Timer search_timer; 
        search_timer.start();

        std::cout << "   MT Test - These numbers should be different: ";
        #pragma omp parallel for
        for (int i = 0; i < 6; i++) {
            #pragma omp critical
            std::cout << (std::size_t)problem::GetShape() % sizeof(timeloop::Shape) << " ";
        }
        std::cout << std::endl;

        std::string moham_out_path = config_.out_dir + "/moham/";
        fs::create_directories(moham_out_path + "pareto");

        const std::string pop_filename = "moham.populations.csv";
        std::ofstream population_file(moham_out_path + pop_filename);

        const std::string eval_filename = "moham.evaluations.csv";
        std::ofstream evaluations_file(moham_out_path + eval_filename);

        // === INITIAL POPULATION ===

        double rpt = RandomPopulation(parent_population_);
        InjectHeuristicallyGoodIndividuals(parent_population_);
        AssignRankAndCrowdingDistance(parent_population_);
        std::cout << "   Initial | " << std::fixed << std::setprecision(2) << rpt << " s" << std::endl;

        // === MAIN LOOP ===

        bool convergence_flag = false;

        for (unsigned g = 0; !convergence_flag && (g < config_.moham.num_generations && config_.moham.num_generations); g++) 
        {
            AppendGenerationInfoToFile(evaluations_file, parent_population_, g);

            std::stringstream ss;
            ss << "   G " << std::right << std::setfill('0') << std::setw(3) << g << "   ";

            unsigned success_cross_count = 0;
            
            ProgressBar bar;
            bar.start(population_.size(), ss.str());

            #pragma omp parallel for schedule(guided)
            for (unsigned i = 0; i < population_.size(); i += 2)
            {
                std::size_t parent_a = config_.moham.use_tournament ? Tournament(parent_population_) : i;
                std::size_t parent_b = config_.moham.use_tournament ? Tournament(parent_population_) : i + 1;

                Crossover(parent_population_[parent_a], parent_population_[parent_b], population_[i], population_[i+1]);

                for (unsigned j = 0; j < 2; j++) 
                {
                    Mutation(population_[i+j]);

                    if ( CheckIndividualValidity(population_[i+j]) ) {
                        #pragma omp atomic update
                        success_cross_count++;
                        Evaluate(population_[i+j]);
                    } 
                    else if (config_.moham.random_when_illegal) {
                        do {
                            population_[i+j] = RandomIndividual();
                        } while (!CheckIndividualValidity(population_[i+j], false));
                        Evaluate(population_[i+j]);
                    } 

                    ++bar;
                }
            }
            bar.stop();

            RandomPopulation(immigrant_population_);
            AppendGenerationInfoToFile(evaluations_file, parent_population_, g);

            Merging(merged_population_, parent_population_, population_, immigrant_population_);
            AssignRankAndCrowdingDistance(merged_population_);
            Survival(parent_population_, merged_population_, !config_.moham.use_tournament, rng_[0]);

            // Print Gen Info   
            std::cout << ss.str();
            ss.str(""); ss << "| " << std::fixed << std::setprecision(2) << bar.time_it_took() << " s";
            std::cout << std::setw(13) << ss.str();
            ss.str(""); ss << "| " << success_cross_count << "/" << std::left << population_.size();
            std::cout << std::setw(13) << ss.str() << "| R  " << RankString(parent_population_);

            // Stopping Criterion Check - Implementation of https://hal.inria.fr/hal-01909120/document Eq. 2
            if (config_.moham.stop_on_convergence) 
            {
                UpdateStabilityMeasure(parent_population_, g, config_.medea.stability_window);
                if (g >= config_.moham.stability_window)
                { 
                    stability_ = CalculateGenerationStability(config_.moham.stability_window);
                    convergence_flag = (stability_ < config_.moham.stability_threshold);
                    std::cout << " | S " << stability_;
                } else {
                    std::cout << " | S n/a";
                }
            }

            std::cout << std::endl;
            AppendGenerationInfoToFile(population_file, parent_population_, g);
        }

        // === TERMINATION ===

        if (convergence_flag) 
            std::cout << "   Stability detected." << std::endl;

        population_file.close();
        evaluations_file.close();

        const std::string pareto_filename = "moham.pareto_schedulings.csv";
        std::ofstream pareto_file(moham_out_path + pareto_filename);
        OutputParetoScheduling(pareto_file, parent_population_);
        pareto_file.close();

        const std::string nip_bottle_filename = "moham.pareto_nip_bottlenecks.csv";
        std::ofstream pareto_nip_file(moham_out_path + nip_bottle_filename);
        OutputParetoNipBottlenecks(pareto_nip_file, parent_population_);
        pareto_nip_file.close();

        double search_time = search_timer.stop();

        const std::string stats_filename = "moham.stats.txt";
        std::ofstream stats_file(moham_out_path + stats_filename);
        stats_file << "Search time: " << search_time << " seconds" << std::endl;
        stats_file.close();

        OutputParetoFrontFiles(moham_out_path + "pareto");

        std::cout << "   Search time: " << search_time << " seconds" << std::endl;
    }


    void Moham::RunRandom() 
    {
        Timer search_timer; 
        search_timer.start();

        std::cout << "   RANDOM SEARCH MODE" << std::endl;
        std::cout << "   MT Test - These numbers should be different: ";
        #pragma omp parallel for
        for (int i = 0; i < 6; i++) {
            #pragma omp critical
            std::cout << (std::size_t)problem::GetShape() % sizeof(timeloop::Shape) << " ";
        }
        std::cout << std::endl;

        std::string moham_out_path = config_.out_dir + "/moth/";
        fs::create_directories(moham_out_path);

        const std::string eval_filename = "moham.evaluations.csv";
        std::ofstream evaluations_file(moham_out_path + eval_filename);

        for (unsigned g = 0; g < config_.moham.num_generations; g++) 
        { 
            double rpt = RandomPopulation(parent_population_);
            AppendGenerationInfoToFile(evaluations_file, parent_population_, g);
            rpt += RandomPopulation(immigrant_population_);
            AppendGenerationInfoToFile(evaluations_file, immigrant_population_, g);

            std::cout << "   Random " << std::right << std::setfill('0') << std::setw(3) << g << " | " << std::fixed << std::setprecision(2) << rpt << " s" << std::endl;
        }
        
        evaluations_file.close();

        double search_time = search_timer.stop();

        const std::string stats_filename = "moham.stats.txt";
        std::ofstream stats_file(moham_out_path + stats_filename);
        stats_file << "Search time: " << search_time << " seconds" << std::endl;
        stats_file.close();

        std::cout << "   Search time: " << search_time << " seconds" << std::endl;
    }


    double Moham::RandomPopulation(Population& pop) 
    {
        ProgressBar bar;
        bar.start(pop.size(), "   ");

        #pragma omp parallel for schedule(guided)
        for (std::size_t p = 0; p < pop.size(); p++) 
        {
            do {
                pop[p] = RandomIndividual();
            } while (!CheckIndividualValidity(pop[p], false));

            Evaluate(pop[p]);
            ++bar;
        }

        bar.stop();
        return bar.time_it_took();
    }


    Moham::Individual Moham::RandomIndividual() 
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t num_layers = graph_->NumLayers();

        Individual ind;
        ind.layer_evals.resize(num_layers, {0,0,0,0,0,0,true});

        // Choosing how many subaccs
        unsigned num_accs = rng() % config_.moham.max_subaccelerators;
        if (num_accs == 0) num_accs = 1;
        ind.accelerators.resize(num_accs);

        // Choosing subaccs templates
        ArchInstance::ID instance_id = 0;
        for (auto& acc : ind.accelerators) 
        {
            acc.id = instance_id++;
            acc.arch_template_id = rng() % arch_templates_->size();
            acc.num_assigned_layers = 0;
        }

        ind.genome.resize(num_layers);

        for (std::size_t l = 0; l < num_layers; l++) 
        {
            // Try 5 times to find acc instance that allows this type of layer
            // validity check will do the rest
            for (unsigned tries = 0; tries < 5; tries++) 
            {
                ArchInstance::ID acc_instance_id = rng() % num_accs;
                ArchTemplate::ID template_id = ind.accelerators[acc_instance_id].arch_template_id;

                std::size_t possible_mappings = workload_mappings_[(*graph_)[l].workload_id].at(template_id).size();
                Medea::Individual::ID mapping_id = possible_mappings > 0 ? rng() % possible_mappings : 0;
                
                ind.genome[l] = (Gene){
                    .priority =         uni_distribution_(rng),
                    .mapping_id =       mapping_id,
                    .acc_instance_id =  acc_instance_id
                };

                if (possible_mappings > 0) {
                    ind.accelerators[acc_instance_id].num_assigned_layers++;
                    break;
                }
            }
        }

        //AssignBreadthFirstPriorities(ind);

        return ind;   
    }


    void Moham::InjectHeuristicallyGoodIndividuals(Population& population) 
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t num_layers = graph_->NumLayers();

        // Lowest latency
        {
            std::size_t individual_id = rng() % population.size();
            Individual& chosen_individual = population[individual_id];
            chosen_individual.accelerators.clear();

            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchTemplate::ID min_template = 0;
                Medea::Individual::ID min_mapping = 0;
                double min_latency = std::numeric_limits<double>::max();

                for (auto& template_mappings : workload_mappings_[(*graph_)[l].workload_id])
                    for (Medea::Individual::ID m = 0; m < template_mappings.second.size(); m++)
                        if (template_mappings.second[m].objectives[1] < min_latency)
                        {
                            min_template = template_mappings.first;
                            min_mapping = m;
                            min_latency = template_mappings.second[m].objectives[1];
                        }

                chosen_individual.genome[l].mapping_id = min_mapping;
                auto accelerator = std::find_if(chosen_individual.accelerators.begin(), chosen_individual.accelerators.end(), 
                                                [&](const ArchInstance& acc){ return acc.arch_template_id == min_template; });
                if (accelerator == chosen_individual.accelerators.end())
                {
                    ArchInstance::ID aid = chosen_individual.accelerators.size();
                    chosen_individual.accelerators.emplace_back();
                    chosen_individual.accelerators.back().id = aid;
                    chosen_individual.accelerators.back().arch_template_id = min_template;  
                    chosen_individual.accelerators.back().num_assigned_layers = 1;     
                    chosen_individual.genome[l].acc_instance_id = aid;
                } else {
                    chosen_individual.genome[l].acc_instance_id = accelerator->id;
                    accelerator->num_assigned_layers++;
                }
            }

            // Breadth-First topological ordering
            AssignBreadthFirstPriorities(chosen_individual);

            assert(CheckIndividualValidity(chosen_individual));
            Evaluate(chosen_individual);
        }

        // Lowest energy
        {
            std::size_t individual_id = rng() % population.size();
            Individual& chosen_individual = population[individual_id];
            chosen_individual.accelerators.clear();

            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchTemplate::ID min_template = 0;
                Medea::Individual::ID min_mapping = 0;
                double min_energy = std::numeric_limits<double>::max();

                for (auto& template_mappings : workload_mappings_[(*graph_)[l].workload_id])
                    for (Medea::Individual::ID m = 0; m < template_mappings.second.size(); m++)
                        if (template_mappings.second[m].objectives[0] < min_energy)
                        {
                            min_template = template_mappings.first;
                            min_mapping = m;
                            min_energy = template_mappings.second[m].objectives[1];
                        }

                chosen_individual.genome[l].mapping_id = min_mapping;
                auto accelerator = std::find_if(chosen_individual.accelerators.begin(), chosen_individual.accelerators.end(), 
                                                [&](const ArchInstance& acc){ return acc.arch_template_id == min_template; });
                if (accelerator == chosen_individual.accelerators.end())
                {
                    ArchInstance::ID aid = chosen_individual.accelerators.size();
                    chosen_individual.accelerators.emplace_back();
                    chosen_individual.accelerators.back().id = aid;
                    chosen_individual.accelerators.back().arch_template_id = min_template; 
                    chosen_individual.accelerators.back().num_assigned_layers = 1;     
                    chosen_individual.genome[l].acc_instance_id = aid;
                } else {
                    chosen_individual.genome[l].acc_instance_id = accelerator->id;
                    accelerator->num_assigned_layers++;
                }
            }

            AssignBreadthFirstPriorities(chosen_individual);

            assert(CheckIndividualValidity(chosen_individual));
            Evaluate(chosen_individual);
        }

    }


    void Moham::Evaluate(Individual& ind) 
    {
        NegotiateArchitectures(ind);

        for (Layer::ID l = 0; l < graph_->NumLayers(); l++)
        {
            const Layer& layer = (*graph_)[l];
            const Gene& gene = ind.genome[l];
            ArchInstance& accelerator = ind.accelerators[gene.acc_instance_id];

            // Evaluate if new layer mapping or instance architecture updated
            //if (ind.layer_evals[l].need_evaluation || accelerator.need_evaluation) 
            { 
                auto engine = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].engine;
                auto& mapping = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].genome;
                auto& workload = graph_->GetWorkloads()[layer.workload_id]; 
                auto* sparse_opt = &(*arch_templates_)[accelerator.arch_template_id].sparse_optimizations;
                
                problem::shape_ = workload.shape; // Update thread private global variable before evaluating
                
                engine.Spec(accelerator.arch_specs);
                auto status = engine.Evaluate(mapping, workload.workload, sparse_opt);

                if (!engine.IsEvaluated())
                {
                    std::cout << "Error in workload " << (*graph_)[l].workload_id << " with mapping " << ind.genome[l].mapping_id << std::endl;

                    for (auto &s : status)
                        std::cout << s.success << " " << s.fail_reason << std::endl;
                    YAML::Emitter yout;
                    for (auto& a : ind.accelerators)
                        yout << a.arch;
                    std::cout << yout.c_str() << std::endl;
                    exit(1);
                } 
                
                ind.layer_evals[l].energy = engine.Energy();
                ind.layer_evals[l].cycles = engine.Cycles();

                // Update Bandwidth Requirement
                const timeloop::Topology& topology = engine.GetTopology();
                auto arithmetic_minarch = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].arch.GetLevel(0);
                ind.layer_evals[l].required_bandwidth = (double)topology.LastLevelAccesses() / ((double)topology.ActualComputes() / (arithmetic_minarch.mesh_x * arithmetic_minarch.mesh_y));
            
                // Accelerator area
                accelerator.area = engine.Area();
                ind.layer_evals[l].area = engine.Area();

                // Evaluated
                ind.layer_evals[l].need_evaluation = false;
            }
        }

        for (auto& acc : ind.accelerators)
            acc.need_evaluation = false;
        
        // Evaluating Objectives

        std::fill(ind.objectives.begin(), ind.objectives.end(), 0.0);

        // Calculate MAS Package Mesh Topology if needed for energy or latency calculation
        if (config_.moham.nip_hop_energy > 0.0 || config_.moham.nip_link_bandwidth > 0.0)
            CalculatePackageMesh(ind);

        // Energy
        for (const auto& eval : ind.layer_evals)
            ind.objectives[0] += eval.energy;

        // Total NiP Hop Energy
        ind.objectives[0] += EvaluateNipEnergy(ind);

        // Makespan
        if (!config_.moham.xu_priority || ind.toposort.empty())
            ind.toposort = GenerateTopologicalSorting(ind);

        ind.objectives[1] = EvaluateMakespan(ind, ind.toposort);

        // Area
        if (config_.moham.negotiate_arch) {
            for (const auto& acc : ind.accelerators) 
                ind.objectives[2] += acc.area;
        } else {
            ind.objectives[2] = config_.moham.max_subaccelerators * ind.accelerators[0].area;
        }
    }


    bool Moham::CheckIndividualValidity(Individual& ind, bool do_assert) const 
    {
        ind.valid = false;
        
        if (ind.accelerators.empty()) return false;

        std::unordered_set<ArchInstance::ID> arch_instance_ids_test_set;        

        for (Layer::ID l = 0; l < graph_->NumLayers(); l++) 
        {
            ArchInstance::ID instance_id = ind.genome[l].acc_instance_id;
            arch_instance_ids_test_set.insert(instance_id);

            Workload::ID workload_id = (*graph_)[l].workload_id;
            ArchTemplate::ID template_id = ind.accelerators[instance_id].arch_template_id;
            Medea::Individual::ID mapping_id = ind.genome[l].mapping_id;
            
            if (workload_mappings_.at(workload_id).find(template_id) == workload_mappings_.at(workload_id).end())
                return false;

            std::size_t possible_mappings = workload_mappings_.at(workload_id).at(template_id).size();

            if (mapping_id >= possible_mappings) 
                return false;

            // TMP for debug
            std::unordered_map<ArchInstance::ID, unsigned> num_assigned_layers;
            for (auto& gene : ind.genome)
                num_assigned_layers[gene.acc_instance_id] += 1;
            for (auto& acc : ind.accelerators)
                if (num_assigned_layers[acc.id] != acc.num_assigned_layers) {
                    if (do_assert) assert(num_assigned_layers[acc.id] == acc.num_assigned_layers); 
                    return false;
                }
        }

        if (arch_instance_ids_test_set.size() != ind.accelerators.size())
            return false;

        for (ArchInstance::ID a = 0; a < ind.accelerators.size(); a++)
            if (arch_instance_ids_test_set.find(a) == arch_instance_ids_test_set.end())
                return false;

        ind.valid = true;
        return true;
    }

    // TODO?? Layer* instead of Layer::ID for performance
    std::vector<Layer::ID> Moham::GenerateTopologicalSorting(const Individual& ind)
    {
        std::size_t num_layers = graph_->NumLayers();

        std::vector<Layer::ID> sorting;
        sorting.reserve(num_layers);

        std::unordered_set<Layer::ID> start_nodes;

        std::vector<int> in_degree(num_layers, 0);

        for (std::size_t l = 0; l < num_layers; l++) 
        {
            in_degree[l] = (*graph_)[l].incoming.size();
            if (in_degree[l] == 0) start_nodes.insert(l);
        }   
        
        while (!start_nodes.empty()) 
        {
            // Find start_node with highest priority
            auto mit = std::max_element(start_nodes.begin(), start_nodes.end(), [&ind](size_t a, size_t b){
                return ind.genome[a].priority > ind.genome[b].priority;
            });

            Layer::ID layer_id = *mit;
            sorting.push_back(layer_id);
            start_nodes.erase(mit);

            for (auto& e : (*graph_)[layer_id].outgoing)
                if (--in_degree[e.layer->id] == 0)
                    start_nodes.insert(e.layer->id);
        }

        assert(sorting.size() == num_layers);
        return sorting;
    }


    void Moham::CalculatePackageMesh(Individual& ind) 
    {
        std::size_t num_accelerators = ind.accelerators.size();

        unsigned mesh_x = std::ceil(std::sqrt(num_accelerators));
        unsigned mesh_y = std::ceil((double)num_accelerators / mesh_x);

        unsigned num_mis = config_.moham.max_memory_interfaces_amount;
        for (auto& am : config_.moham.memory_interfaces_amount)
            if (num_accelerators <= am.first)
                num_mis = am.second;

        if (num_mis > 2 * mesh_y) num_mis = 2 * mesh_y;

        for (ArchInstance::ID said = 0; said < ind.accelerators.size(); said++)
            ind.accelerators[said].mesh_coordinates = {said % mesh_x, said / mesh_x};

        for (MemoryInterface::ID mid = 0; mid < num_mis; mid++)
        {
            int x = (mid % 2) ? (mesh_x + 1) : (-1);
            int yt = mid / 2;
            int y = 0;
            if (config_.moham.memory_interfaces_position == "corner") {
                y = (yt % 2) ? (mesh_y - 1 - yt / 2) : (yt / 2);
            } else if (config_.moham.memory_interfaces_position == "middle") {
                int middle = (mesh_y - 1) / 2;
                y = (yt % 2) ? (middle + 1 + yt / 2) : (middle - yt / 2);
            } else {
                assert(false);
            }

            ind.memory_interfaces.push_back({mid, std::make_pair(x,y)});
        }

        for (ArchInstance::ID said = 0; said < ind.accelerators.size(); said++)
        {
            std::vector<MemoryInterface::ID> nearest_mis;
            int min_distance = std::numeric_limits<int>::max();
            for (MemoryInterface::ID mid = 0; mid < num_mis; mid++)
            {
                auto sac = ind.accelerators[said].mesh_coordinates;
                auto mic = ind.memory_interfaces[mid].mesh_coordinates;
                int distance = std::abs(sac.first - mic.first) + std::abs(sac.second - mic.second);
                if (distance == min_distance) {
                    nearest_mis.push_back(mid);
                } else if (distance < min_distance) {
                    nearest_mis = std::vector<MemoryInterface::ID>(1, mid);
                    min_distance = distance;
                }
            }
            ind.accelerators[said].memory_interface_id = *max_element(nearest_mis.begin(), nearest_mis.end());
        }

    }


    double Moham::EvaluateNipEnergy(Individual& ind)
    {
        double total_hop_energy = 0.0;

        // Network-in-Package Energy Simulation
        if (config_.moham.nip_hop_energy > 0) {
            std::size_t num_layers = graph_->NumLayers();
            std::size_t num_mis = ind.memory_interfaces.size();

            // Layers served by each MemoryInterface
            std::vector<std::vector<Layer::ID>> layer_in_mis(num_mis);
            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchInstance::ID said = ind.genome[l].acc_instance_id;
                MemoryInterface::ID mid = ind.accelerators[said].memory_interface_id;
                layer_in_mis[mid].push_back(l);
            }

            // Hop energy from memory interface for each layer
            
            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchInstance::ID said = ind.genome[l].acc_instance_id;
                MemoryInterface::ID mid = ind.accelerators[said].memory_interface_id;
                auto sa_coords = ind.accelerators[said].mesh_coordinates;
                auto mi_coords = ind.memory_interfaces[mid].mesh_coordinates;
                double layer_distance_from_mi = std::abs(sa_coords.first - mi_coords.first) + std::abs(sa_coords.second - mi_coords.second);
                double layer_data_transfered = ind.layer_evals[l].cycles * ind.layer_evals[l].required_bandwidth;
                double layer_hop_energy = layer_data_transfered * config_.moham.nip_hop_energy * layer_distance_from_mi;
                total_hop_energy += layer_hop_energy;
            }
        }

        return total_hop_energy;
    }


    double Moham::EvaluateMakespan(Individual& ind, const std::vector<Layer::ID>& topological_sorting)
    {
        std::size_t num_layers = graph_->NumLayers();

        std::vector<double> layer_finishing_times(num_layers, 0.0);
        std::unordered_map<ArchInstance::ID, double> instances_finishing_times;

        for (Layer::ID l : topological_sorting)
        {
            const Layer& layer = (*graph_)[l];
            const Gene& gene = ind.genome[l];
            
            double start_time = 0.0;
            for (auto& dep : layer.incoming)
                start_time = std::max(layer_finishing_times[dep.layer->id], start_time);
            start_time = std::max(instances_finishing_times[gene.acc_instance_id], start_time);

            double end_time = start_time + ind.layer_evals[l].cycles;
            
            layer_finishing_times[layer.id] = end_time;
            instances_finishing_times[gene.acc_instance_id] = end_time;

            ind.layer_evals[l].start_time = start_time;
            ind.layer_evals[l].end_time = end_time;
        }

        // Network-in-Package Simulation
        if (config_.moham.nip_link_bandwidth > 0) {
            ind.nip_bottlenecks.clear();

            std::size_t num_mis = ind.memory_interfaces.size();

            // Layers served by each MemoryInterface
            std::vector<std::vector<Layer::ID>> layer_in_mis(num_mis);
            for (Layer::ID l = 0; l < num_layers; l++)
            {
                ArchInstance::ID said = ind.genome[l].acc_instance_id;
                MemoryInterface::ID mid = ind.accelerators[said].memory_interface_id;
                layer_in_mis[mid].push_back(l);
            }

            // Scegliamo intervallo da considerare.
            std::vector<double> time_hooks(num_mis, 0.0);

            while(true) {
                double min_hyp_hook = std::numeric_limits<double>::max();
                MemoryInterface::ID min_hyp_mid = 0;

                // Find mid with nearest segment end
                for (Layer::ID l = 0; l < num_layers; l++) 
                {
                    ArchInstance::ID said = ind.genome[l].acc_instance_id;
                    MemoryInterface::ID mid = ind.accelerators[said].memory_interface_id;
                    
                    if (ind.layer_evals[l].start_time > time_hooks[mid] &&
                        ind.layer_evals[l].start_time < min_hyp_hook )
                    {
                        min_hyp_hook = ind.layer_evals[l].start_time;
                        min_hyp_mid = mid;
                    }

                    if (ind.layer_evals[l].end_time > time_hooks[mid] &&
                        ind.layer_evals[l].end_time < min_hyp_hook ) 
                    {
                        min_hyp_hook = ind.layer_evals[l].end_time;
                        min_hyp_mid = mid;
                    }
                }

                // No more segments -> break
                if (min_hyp_hook == std::numeric_limits<double>::max()) 
                    break;

                double start = time_hooks[min_hyp_mid];
                double end = min_hyp_hook;
                
                std::vector<Layer::ID> affected_layers;

                double required_bandwidth = 0.0;
                for (Layer::ID l : layer_in_mis[min_hyp_mid])
                    if (ind.layer_evals[l].start_time <= start && end <= ind.layer_evals[l].end_time) {
                        required_bandwidth += ind.layer_evals[l].required_bandwidth;
                        affected_layers.push_back(l);
                    }

                if (required_bandwidth > config_.moham.nip_link_bandwidth)
                {
                    // Extending segment
                    double slowdown = required_bandwidth / config_.moham.nip_link_bandwidth;
                    double interval_length = end - start;
                    double overhead = std::ceil(slowdown * interval_length) - interval_length;

                    for (Layer::ID l : affected_layers)
                        ind.layer_evals[l].end_time += overhead;

                    // Fixing scheduling after segment

                    std::vector<double> layer_finishing_times(num_layers, 0.0);
                    std::unordered_map<ArchInstance::ID, double> instances_finishing_times;

                    for (Layer::ID l = 0; l < num_layers; l++)
                    {
                        ArchInstance::ID said = ind.genome[l].acc_instance_id;
                        MemoryInterface::ID mid = ind.accelerators[said].memory_interface_id;

                        if (ind.layer_evals[l].start_time <= time_hooks[mid])
                        {
                            layer_finishing_times[l] = ind.layer_evals[l].end_time;
                            instances_finishing_times[said] = std::max(instances_finishing_times[said], ind.layer_evals[l].end_time);
                        }
                    }

                    for (Layer::ID l : topological_sorting)
                    {
                        ArchInstance::ID said = ind.genome[l].acc_instance_id;
                        MemoryInterface::ID mid = ind.accelerators[said].memory_interface_id;

                        if (ind.layer_evals[l].start_time > time_hooks[mid])
                        {
                            const Layer& layer = (*graph_)[l];
                            
                            double start_time = instances_finishing_times[said];
                            for (auto& dep : layer.incoming)
                                start_time = std::max(layer_finishing_times[dep.layer->id], start_time);

                            double end_time = start_time + ind.layer_evals[l].cycles;
                            
                            layer_finishing_times[l] = end_time;
                            instances_finishing_times[said] = end_time;

                            ind.layer_evals[l].start_time = start_time;
                            ind.layer_evals[l].end_time = end_time;
                        }
                    }

                    for (Layer::ID l : affected_layers)
                        ind.nip_bottlenecks.push_back({start, end + overhead, ind.genome[l].acc_instance_id});
                
                    time_hooks[min_hyp_mid] = end + overhead;
                }
                else 
                {
                    time_hooks[min_hyp_mid] = end;
                }
            }
        }
        
        // System Bandwidth Bottlenecks (DRAM)

        if (config_.moham.system_bandwidth > 0.0) 
        {
            ind.memory_bottlenecks.clear();

            std::vector<double> sep_times;
            for (Layer::ID l = 0; l < num_layers; l++) {
                sep_times.push_back(ind.layer_evals[l].start_time);
                sep_times.push_back(ind.layer_evals[l].end_time);
            }
            std::sort( sep_times.begin(), sep_times.end() );
            sep_times.erase( std::unique( sep_times.begin(), sep_times.end() ), sep_times.end() );

            for (std::size_t interval = 0; interval < sep_times.size() - 1; interval++) 
            {
                double start = sep_times[interval];
                double end = sep_times[interval+1];
                
                double required_bandwidth = 0.0;
                for (Layer::ID l : topological_sorting)
                    if (ind.layer_evals[l].start_time <= start && end <= ind.layer_evals[l].end_time)
                        required_bandwidth += ind.layer_evals[l].required_bandwidth;

                if (required_bandwidth > config_.moham.system_bandwidth)
                {
                    double slowdown = required_bandwidth / config_.moham.system_bandwidth;
                    double interval_length = end - start;
                    double overhead = std::ceil(slowdown * interval_length) - interval_length;

                    for (std::size_t suc_interval = interval + 1; suc_interval < sep_times.size(); suc_interval++)
                        sep_times[suc_interval] += overhead;

                    for (Layer::ID l = 0; l < num_layers; l++)
                    {
                        if (ind.layer_evals[l].end_time >= end) ind.layer_evals[l].end_time += overhead;
                        if (ind.layer_evals[l].start_time >= end) ind.layer_evals[l].start_time += overhead;
                    }

                    ind.memory_bottlenecks.push_back({ sep_times[interval], sep_times[interval+1], 0});
                } 
            }
        }

        double makespan = 0.0;
        for (Layer::ID l = 0; l < num_layers; l++)
            if (ind.layer_evals[l].end_time > makespan)
                makespan = ind.layer_evals[l].end_time;

        return makespan;
    }


    void Moham::NegotiateArchitectures(Individual& ind)
    {
        if (config_.moham.negotiate_arch) 
        {
            std::unordered_map<ArchInstance::ID, MinimalArchSpecs> new_negotiated_archs;

            for (size_t l = 0; l < graph_->NumLayers(); l++)
            {
                ArchInstance::ID acc_instance_id = ind.genome[l].acc_instance_id;
                ArchTemplate::ID template_id = ind.accelerators[acc_instance_id].arch_template_id;
                
                Workload::ID workload_id = (*graph_)[l].workload_id;
                Medea::Individual::ID mapping_id = ind.genome[l].mapping_id;

                auto ait = new_negotiated_archs.find(acc_instance_id);

                if (ait != new_negotiated_archs.end()) {
                    ait->second &= workload_mappings_[workload_id][template_id][mapping_id].arch;
                } else { 
                    new_negotiated_archs[acc_instance_id] = workload_mappings_[workload_id][template_id][mapping_id].arch;
                }
            }

            ind.accelerators.resize(new_negotiated_archs.size());
            for (ArchInstance::ID instance_id = 0; instance_id < new_negotiated_archs.size(); instance_id++)
            {
                if (ind.accelerators[instance_id].arch != new_negotiated_archs.at(instance_id))
                {
                    ind.accelerators[instance_id].arch = new_negotiated_archs.at(instance_id);
                    ind.accelerators[instance_id].need_evaluation = true;
                
                    ArchTemplate& arch_template = (*arch_templates_)[ind.accelerators[instance_id].arch_template_id];

                    auto new_specs = timeloop::TopologySpecs(arch_template.arch_specs.topology); 

                    auto minimal_arithmetic = ind.accelerators[instance_id].arch.GetLevel(0);
                    auto arithmetic = new_specs.GetArithmeticLevel();
                    arithmetic->meshX = minimal_arithmetic.mesh_x;
                    arithmetic->meshY = minimal_arithmetic.mesh_y;
                    arithmetic->instances = minimal_arithmetic.mesh_x * minimal_arithmetic.mesh_y;

                    std::unordered_map<std::string, uint64_t> updates;

                    for (unsigned i = 1; i < arch_template.arch_specs.topology.NumLevels(); i++)
                    {
                        auto buffer = new_specs.GetStorageLevel(i - 1);
                        if (!buffer->size.IsSpecified())
                            continue;

                        auto minimal_buffer = ind.accelerators[instance_id].arch.GetLevel(i);
                        buffer->meshX = minimal_buffer.mesh_x;
                        buffer->meshY = minimal_buffer.mesh_y;
                        buffer->instances = minimal_buffer.mesh_x * minimal_buffer.mesh_y;
                        buffer->size = minimal_buffer.size;
                        buffer->effective_size = static_cast<uint64_t>(std::floor(minimal_buffer.size / buffer->multiple_buffering.Get()));

                        updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
                    }

                    std::string out_prefix = "moham." + std::to_string(omp_get_thread_num()) + "_tmp";
                    Accelergy::RT rt = arch_template.accelergy.GetReferenceTables(updates, out_prefix);

                    ind.accelerators[instance_id].arch_specs = (timeloop::EngineSpecs){.topology = new_specs};
                    ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyART(rt.area);
                    ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyERT(rt.energy);
                }
            }
        } 
        else 
        { 
            for (ArchInstance::ID instance_id = 0; instance_id < ind.accelerators.size(); instance_id++)
            {
                ArchTemplate& arch_template = (*arch_templates_)[ind.accelerators[instance_id].arch_template_id];
                auto new_specs = timeloop::TopologySpecs(arch_template.arch_specs.topology); 

                std::unordered_map<std::string, uint64_t> updates;

                for (unsigned i = 1; i < arch_template.arch_specs.topology.NumLevels(); i++)
                {
                    auto buffer = new_specs.GetStorageLevel(i - 1);
                    if (!buffer->size.IsSpecified())
                        continue;

                    updates[buffer->name.Get()] = buffer->size.Get() / buffer->block_size.Get();
                }

                std::string out_prefix = "moham." + std::to_string(omp_get_thread_num()) + "_tmp";
                Accelergy::RT rt = arch_template.accelergy.GetReferenceTables(updates, out_prefix);

                ind.accelerators[instance_id].arch_specs = (timeloop::EngineSpecs){.topology = new_specs};
                ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyART(rt.area);
                ind.accelerators[instance_id].arch_specs.topology.ParseAccelergyERT(rt.energy);
            }
        }
    }


    std::size_t Moham::Tournament(const Population& pop)
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t b1 = rng() % pop.size();
        std::size_t b2 = rng() % pop.size();

        if (pop[b1].rank < pop[b2].rank)
        {
            return b1;
        }
        else if (pop[b1].rank == pop[b2].rank)
        {
            if (pop[b1].crowding_distance > pop[b2].crowding_distance)
                return b1;
            else
                return b2;
        }
        else
        {
            return b2;
        }
    }


    void Moham::Crossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b)
    {
        offspring_a = parent_a;
        offspring_b = parent_b;

        auto& rng =  rng_[omp_get_thread_num()];

        if (uni_distribution_(rng) < config_.moham.priority_crossover_prob) 
            PriorityCrossover(parent_a, parent_b, offspring_a, offspring_b, rng);

        if (uni_distribution_(rng) < config_.moham.subacc_crossover_prob) 
            SubAcceleratorCrossover(parent_a, parent_b, offspring_a, offspring_b, rng);

        if (uni_distribution_(rng) < config_.moham.mapping_crossover_prob) 
            MappingCrossover(parent_a, parent_b, offspring_a, offspring_b, rng);

        // General crossover
        // std::swap_ranges(offspring_a.genome.begin(), offspring_a.genome.begin() + crossover_point, offspring_b.genome.begin());
    }


    void Moham::PriorityCrossover(const Individual& pa, const Individual& pb, Individual& osa, Individual& osb, std::mt19937_64& rng)
    {
        std::size_t num_layers = graph_->NumLayers();
        std::size_t crossover_point = rng() % num_layers;

        if (!config_.moham.xu_priority) {
            for (Layer::ID l = 0; l < crossover_point; l++) 
                std::swap(osb.genome[l].priority, osa.genome[l].priority);
            return;
        }

        Layer::ID pta = crossover_point;
        Layer::ID ptb = crossover_point;

        for (Layer::ID l = 0; l < num_layers; l++)
        {
            if (std::find(pa.toposort.begin(), pa.toposort.begin() + crossover_point, pb.toposort[l]) == (pa.toposort.begin() + crossover_point))
                osa.toposort[pta++] = pb.toposort[l];
            
            if (std::find(pb.toposort.begin(), pb.toposort.begin() + crossover_point, pa.toposort[l]) == (pb.toposort.begin() + crossover_point))
                osb.toposort[ptb++] = pa.toposort[l];
        }

        assert(pta == num_layers && ptb == num_layers);
    }


    void Moham::SubAcceleratorCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng)
    {
        std::size_t num_layers = graph_->NumLayers();
        ArchInstance::ID chosen_acc_id = rng() % parent_a.accelerators.size();

        // Moving accelerator scheduling from A to B
        if (chosen_acc_id < parent_b.accelerators.size())
        {
            // Accelerator already exists in B
            std::swap(offspring_a.accelerators[chosen_acc_id], offspring_b.accelerators[chosen_acc_id]);
            
            // Moving accelerator scheduling from B to A
            for (Layer::ID l = 0; l < num_layers; l++) 
            {
                if (parent_b.genome[l].acc_instance_id == chosen_acc_id)
                {
                    offspring_a.accelerators[offspring_a.genome[l].acc_instance_id].need_evaluation = true;

                    offspring_a.genome[l].mapping_id = parent_b.genome[l].mapping_id;
                    offspring_a.genome[l].acc_instance_id = parent_b.genome[l].acc_instance_id;
                    offspring_a.layer_evals[l].need_evaluation = true;

                }
            }
            // Moving A to B
            for (Layer::ID l = 0; l < num_layers; l++) 
            {
                if (parent_a.genome[l].acc_instance_id == chosen_acc_id)
                {
                    offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].need_evaluation = true;

                    offspring_b.genome[l].mapping_id = parent_a.genome[l].mapping_id;
                    offspring_b.genome[l].acc_instance_id = parent_a.genome[l].acc_instance_id;
                    offspring_b.layer_evals[l].need_evaluation = true;
                }
            } 

            // Count assigned layer to each accelerator
            for (auto& acc : offspring_a.accelerators) acc.num_assigned_layers = 0;
            for (auto& acc : offspring_b.accelerators) acc.num_assigned_layers = 0;
            for (Layer::ID l = 0; l < num_layers; l++) {
                offspring_a.accelerators[offspring_a.genome[l].acc_instance_id].num_assigned_layers++;
                offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].num_assigned_layers++;
            }

            offspring_a.accelerators[chosen_acc_id].need_evaluation = true;
            offspring_b.accelerators[chosen_acc_id].need_evaluation = true;

            RemoveEmptySubAccelerators(offspring_a);
            RemoveEmptySubAccelerators(offspring_b);
        } 
        else 
        {
            // Accelerator doesnt exist in B, we add it
            ArchInstance::ID new_instance_id = offspring_b.accelerators.size();
            offspring_b.accelerators.push_back(parent_a.accelerators[chosen_acc_id]);
            offspring_b.accelerators.back().id = new_instance_id;

            for (Layer::ID l = 0; l < num_layers; l++) 
            {
                if (parent_a.genome[l].acc_instance_id == chosen_acc_id)
                {
                    offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].need_evaluation = true;

                    offspring_b.genome[l].mapping_id = parent_a.genome[l].mapping_id;
                    offspring_b.genome[l].acc_instance_id = new_instance_id;
                    offspring_b.layer_evals[l].need_evaluation = true;
                }
            }

            // Count assigned layer to each accelerator
            for (auto& acc : offspring_b.accelerators) acc.num_assigned_layers = 0;
            for (Layer::ID l = 0; l < num_layers; l++) {
                offspring_b.accelerators[offspring_b.genome[l].acc_instance_id].num_assigned_layers++;
            }

            offspring_b.accelerators[new_instance_id].need_evaluation = true;
            RemoveEmptySubAccelerators(offspring_b);
        }


        // TODO Repair when templates are different
    }


    void Moham::MappingCrossover(const Individual& parent_a, const Individual& parent_b, Individual& offspring_a, Individual& offspring_b, std::mt19937_64& rng)
    {
        std::size_t crossover_point = rng() % graph_->NumLayers();

        for (Layer::ID l = 0; l < crossover_point; l++) 
        {
            ArchTemplate::ID template_a = parent_a.accelerators[parent_a.genome[l].acc_instance_id].arch_template_id;
            ArchTemplate::ID template_b = parent_b.accelerators[parent_b.genome[l].acc_instance_id].arch_template_id;
            Workload::ID workload_id = (*graph_)[l].workload_id;

            if (template_a == template_b)
            {
                std::swap(offspring_a.genome[l].mapping_id, offspring_b.genome[l].mapping_id);
            } 
            else 
            {
                offspring_a.genome[l].mapping_id = ConvertMappingToTemplate(workload_id, parent_b.genome[l].mapping_id, template_b, template_a);
                offspring_b.genome[l].mapping_id = ConvertMappingToTemplate(workload_id, parent_a.genome[l].mapping_id, template_a, template_b);
            }

            offspring_a.layer_evals[l].need_evaluation = true;
            offspring_b.layer_evals[l].need_evaluation = true;        
        }
    }
    

    void Moham::Mutation(Individual& ind)
    {
        auto& rng =  rng_[omp_get_thread_num()];
        std::size_t num_layers = graph_->NumLayers();

        if (uni_distribution_(rng) < config_.moham.splitting_mutation_prob)
            SplittingMutation(ind, rng);

        if (uni_distribution_(rng) < config_.moham.merging_mutation_prob)
            MergingMutation(ind, rng);

        if (uni_distribution_(rng) < config_.moham.template_mutation_prob)
            TemplateMutation(ind, rng);

        if (uni_distribution_(rng) < config_.moham.position_mutation_prob)
            PositionMutation(ind, rng);

        for (Layer::ID l = 0; l < num_layers; l++)
        {
            if (uni_distribution_(rng) < config_.moham.priority_mutation_prob)
                PriorityMutation(ind, l, rng);

            if (uni_distribution_(rng) < config_.moham.mapping_mutation_prob)
                MappingMutation(ind, l, rng);

            if (uni_distribution_(rng) < config_.moham.assignment_mutation_prob)
                AssignmentMutation(ind, l, rng);
        }
    }
    

    void Moham::SplittingMutation(Individual& ind, std::mt19937_64& rng)
    {
        if (ind.accelerators.size() >= config_.moham.max_subaccelerators)
            return;
        
        ArchInstance::ID acc_to_split = rng() % ind.accelerators.size();
        ArchInstance::ID new_acc_id = ind.accelerators.size();

        ind.accelerators.push_back(ind.accelerators[acc_to_split]);
        ind.accelerators[new_acc_id].id = new_acc_id;
        
        for (Layer::ID l = 0; l < graph_->NumLayers(); l++)
            if (ind.genome[l].acc_instance_id == acc_to_split)
            {
                if (uni_distribution_(rng) < 0.5) {
                    ind.genome[l].acc_instance_id = new_acc_id;
                    ind.accelerators[acc_to_split].num_assigned_layers--;
                } else {
                    ind.accelerators[new_acc_id].num_assigned_layers--;
                }
            }
        
        RemoveEmptySubAccelerators(ind); 
    }


    void Moham::MergingMutation(Individual& ind, std::mt19937_64& rng)
    {
        if (ind.accelerators.size() <= 1) return;
        ArchInstance::ID id_a = rng() % ind.accelerators.size();
        ArchInstance::ID id_b = rng() % ind.accelerators.size();
        if (id_a == id_b) return;
        
        ArchInstance::ID destination_id = std::min(id_a, id_b);
        ArchInstance::ID source_id = std::max(id_a, id_b);

        ArchTemplate::ID destination_template = ind.accelerators[destination_id].arch_template_id;
        ArchTemplate::ID source_template = ind.accelerators[source_id].arch_template_id;

        ind.accelerators[destination_id].num_assigned_layers += ind.accelerators[source_id].num_assigned_layers;
        ind.accelerators[source_id].num_assigned_layers = 0;
        
        for (Layer::ID l = 0; l < ind.genome.size(); l++)
            if (ind.genome[l].acc_instance_id == source_id) 
            {
                ind.genome[l].acc_instance_id = destination_id;

                if (destination_template != source_template && 
                    workload_mappings_[(*graph_)[l].workload_id][destination_template].size() > 0 )
                    ind.genome[l].mapping_id = ConvertMappingToTemplate((*graph_)[l].workload_id, ind.genome[l].mapping_id, source_template, destination_template);
            } 

        RemoveEmptySubAccelerators(ind);           

    }


    void Moham::TemplateMutation(Individual& ind, std::mt19937_64& rng)
    {
        ArchInstance::ID acc_id = rng() % ind.accelerators.size();
        ArchTemplate::ID start_template = ind.accelerators[acc_id].arch_template_id;

        ArchTemplate::ID destination_template = rng() % arch_templates_->size();
        if (start_template == destination_template) return;

        ind.accelerators[acc_id].arch_template_id = destination_template;
        for (Layer::ID l = 0; l < ind.genome.size(); l++)
            if (ind.genome[l].acc_instance_id == acc_id) 
                if (destination_template != start_template && 
                    workload_mappings_[(*graph_)[l].workload_id][destination_template].size() > 0 ) 
                    ind.genome[l].mapping_id = ConvertMappingToTemplate((*graph_)[l].workload_id, ind.genome[l].mapping_id, start_template, destination_template);
        
        ind.accelerators[acc_id].need_evaluation = true;
    }


    void Moham::PositionMutation(Individual& ind, std::mt19937_64& rng)
    {
        if (ind.accelerators.size() <= 1) return;

        ArchInstance::ID acc_id_a = rng() % ind.accelerators.size();
        ArchInstance::ID acc_id_b = rng() % ind.accelerators.size();

        if (acc_id_a == acc_id_b) return;

        for (Layer::ID l = 0; l < graph_->NumLayers(); l++)
        {
            if (ind.genome[l].acc_instance_id == acc_id_a)
            {
                ind.genome[l].acc_instance_id = acc_id_b;
            }
            else if (ind.genome[l].acc_instance_id == acc_id_b)
            {
                ind.genome[l].acc_instance_id = acc_id_a;
            }
        }

        std::swap(ind.accelerators[acc_id_a], ind.accelerators[acc_id_b]);
        std::swap(ind.accelerators[acc_id_a].id, ind.accelerators[acc_id_b].id);
    }


    void Moham::PriorityMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng)
    {
        if (!config_.moham.xu_priority) {
            ind.genome[l].priority = uni_distribution_(rng);
            return;
        }

        Layer::ID succ = std::numeric_limits<std::size_t>::max();
        for (auto& out_edge : (*graph_)[ind.toposort[l]].outgoing)
        {
            auto itr = std::find(ind.toposort.begin(), ind.toposort.end(), out_edge.layer->id);
            Layer::ID new_succ = std::distance(ind.toposort.begin(), itr);
            succ = std::min(new_succ, succ);
        }
        
        int span = (int)succ - l - 1;
        if (succ >= ind.toposort.size() || span <= 0) return;

        Layer::ID k = l + 1 + (rng() % span);
        for (auto& in_edge : (*graph_)[ind.toposort[k]].incoming)
        {
            auto itr = std::find(ind.toposort.begin(), ind.toposort.end(), in_edge.layer->id);
            Layer::ID pred = std::distance(ind.toposort.begin(), itr);
            if (pred > l) return;
        }

        std::swap(ind.toposort[l], ind.toposort[k]);
    }


    void Moham::MappingMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng)
    {
        Workload::ID workload_id = (*graph_)[l].workload_id;
        ArchTemplate::ID template_id = ind.accelerators.at(ind.genome[l].acc_instance_id).arch_template_id;

        auto& possible_mapings = workload_mappings_[workload_id][template_id];
        std::size_t num_possible_mappings = possible_mapings.size();
        if (num_possible_mappings == 0) return; // invalid mapping
        
        ind.genome[l].mapping_id = rng() % num_possible_mappings;
    }


    void Moham::AssignmentMutation(Individual& ind, Layer::ID l, std::mt19937_64& rng)
    {
        Medea::Individual::ID mapping_id = ind.genome[l].mapping_id;
        Workload::ID workload_id = (*graph_)[l].workload_id;
        ArchInstance::ID starting_instance_id = ind.genome[l].acc_instance_id;
        ArchTemplate::ID starting_template = ind.accelerators[starting_instance_id].arch_template_id;

        if ( mapping_id >= workload_mappings_[workload_id][starting_template].size() ) return; // invalid mapping

        std::size_t max_instance_id = std::min(ind.accelerators.size()+1, (std::size_t)config_.moham.max_subaccelerators);
        ArchInstance::ID destination_instance_id = rng() % max_instance_id;

        if (destination_instance_id == ind.accelerators.size()) 
        {
            ind.accelerators.push_back(ind.accelerators[starting_instance_id]);
            ind.accelerators.back().id = destination_instance_id;
            ind.accelerators.back().num_assigned_layers = 1;

            ind.genome[l].acc_instance_id = destination_instance_id;

            ind.accelerators[starting_instance_id].num_assigned_layers--;
        } 
        else 
        { 
            ArchTemplate::ID destination_template = ind.accelerators[destination_instance_id].arch_template_id;
            if (workload_mappings_[workload_id][destination_template].size() > 0) 
            {
                ind.genome[l].acc_instance_id = destination_instance_id;

                if (starting_template != destination_template)
                    ind.genome[l].mapping_id = ConvertMappingToTemplate(workload_id, mapping_id, starting_template, destination_template);
            
                ind.accelerators[starting_instance_id].num_assigned_layers--;
                ind.accelerators[destination_instance_id].num_assigned_layers++;
            }
        }

        RemoveEmptySubAccelerators(ind);
    }


    void Moham::RemoveEmptySubAccelerators(Individual& ind) const 
    {
        for(auto ita = ind.accelerators.begin(); ita != ind.accelerators.end(); )
        {
            if (ita->num_assigned_layers == 0)
            {
                ita = ind.accelerators.erase(ita); 
                for (auto itar = ita; itar != ind.accelerators.end(); itar++)
                    itar->id--;
                for (Layer::ID l = 0; l < ind.genome.size(); l++)
                    if ( ind.genome[l].acc_instance_id > (ArchInstance::ID)(ita - ind.accelerators.begin()) )
                        ind.genome[l].acc_instance_id--;
            }
            else
            {
                ++ita;
            }
        }
    }


    Medea::Individual::ID Moham::ConvertMappingToTemplate(Workload::ID workload_id, Medea::Individual::ID mapping_id, ArchTemplate::ID start_template, ArchTemplate::ID destination_template) const 
    {   
        if (mapping_id >= workload_mappings_[workload_id].at(start_template).size() ) return 0;

        auto cur_normalized_objectives = workload_mappings_normalized_objectives_[workload_id].at(start_template).at(mapping_id);
        Medea::Individual::ID nearest_point_id = 0;
        double minimum_distance = std::numeric_limits<double>::max();

        for (Medea::Individual::ID i = 0; i < workload_mappings_normalized_objectives_[workload_id].at(destination_template).size(); i++)
        {
            double distance = 0.0;
            for (unsigned o = 0; o < workload_mappings_normalized_objectives_[workload_id].at(destination_template)[i].size(); o++) {
                double obj = workload_mappings_normalized_objectives_[workload_id].at(destination_template)[i][o];
                distance += (obj-cur_normalized_objectives[o]) * (obj-cur_normalized_objectives[o]);
            }
            if (distance < minimum_distance) nearest_point_id = i;
        }

        return nearest_point_id;
    }


    void Moham::OutputParetoScheduling(std::ofstream &out, const Population& pop) 
    {
        out << "individual,layer,network,accelerator,template,start_time,end_time,energy,cycles,area,toposort" << std::endl;
        for (std::size_t i = 0, j = 0; i < pop.size(); i++)
        {
            if (pop[i].rank == 0) 
            {
                std::string toposort = std::to_string(pop[i].toposort[0]);
                for (Layer::ID l = 1; l < pop[i].toposort.size(); l++)
                    toposort += "-" + std::to_string(pop[i].toposort[l]);

                for (Layer::ID l = 0; l < graph_->NumLayers(); l++) 
                {           
                    out << j;
                    out << "," << l;
                    out << "," << (*graph_)[l].network;
                    out << "," << pop[i].genome[l].acc_instance_id;
                    out << "," << (*arch_templates_)[pop[i].accelerators[pop[i].genome[l].acc_instance_id].arch_template_id].name;
                    out << "," << pop[i].layer_evals[l].start_time;
                    out << "," << pop[i].layer_evals[l].end_time;
                    out << "," << pop[i].layer_evals[l].energy;
                    out << "," << pop[i].layer_evals[l].cycles;
                    out << "," << pop[i].layer_evals[l].area;
                    out << "," << toposort;
                    out << std::endl;
                }
                j++;
            }
        }
    }


    void Moham::OutputParetoNipBottlenecks(std::ofstream &out, const Population& pop) {
        out << "individual,accelerator,start_time,end_time" << std::endl;
        for (std::size_t i = 0, j = 0; i < pop.size(); i++)
        {
            if (pop[i].rank == 0) 
            {
                for (auto& bottle : pop[i].nip_bottlenecks) 
                {           
                    out << j;
                    out << "," << bottle.acc;
                    out << "," << bottle.start_time;
                    out << "," << bottle.end_time;
                    out << std::endl;
                }
                j++;
            }
        }
    }


    void Moham::AppendGenerationInfoToFile(std::ofstream &out, Population &pop, unsigned gen_id)
    {
        for (Individual &ind : pop)
        {
            out << gen_id << "," << ind.rank << "," << ind.crowding_distance << "," << ind.objectives[0] << "," << ind.objectives[1] << "," << ind.objectives[2] << std::endl;
        }

        out.flush();
    }


    void Moham::AssignBreadthFirstPriorities(Individual& ind) 
    {
        std::size_t num_layers = graph_->NumLayers();
        
        double increase = 1.0 / num_layers;
        double priority = 0.0;

        std::queue<Layer::ID> start_nodes;
        std::vector<int> in_degree(num_layers, 0);

        for (std::size_t l = 0; l < num_layers; l++) 
        {
            in_degree[l] = (*graph_)[l].incoming.size();
            if (in_degree[l] == 0) start_nodes.push(l);
        }   
        
        /*
        while (!start_nodes.empty()) 
        {
            Layer::ID layer_id = start_nodes.front();
            start_nodes.pop();

            ind.genome[layer_id].priority = priority;
            priority += increase;

            for (auto& e : (*graph_)[layer_id].outgoing)
                if (--in_degree[e.layer->id] == 0)
                    start_nodes.push(e.layer->id);
        }
        */

        std::unordered_set<Layer::ID> explored_nodes;
        unsigned debug_num_nodes = 0;

        while (!start_nodes.empty()) 
        {
            Layer::ID layer_id = start_nodes.front();
            start_nodes.pop();

            ind.genome[layer_id].priority = priority;
            priority += increase;

            for (auto& e : (*graph_)[layer_id].outgoing)
            {
                if (explored_nodes.count(e.layer->id) == 0)
                {                  
                    explored_nodes.insert(e.layer->id);
                    start_nodes.push(e.layer->id);
                }
            }
            
            debug_num_nodes++;
        }

        if (debug_num_nodes != num_layers) 
        {
            std::cout << debug_num_nodes << " " << num_layers << std::endl;
            assert(debug_num_nodes == num_layers);
        }
    }


    void Moham::OutputParetoFrontFiles(std::string out_path)
    {
        int max_digits = std::to_string(config_.moham.population_size).length();
        int ind_max_digits = std::to_string(graph_->NumLayers()).length();

        #pragma omp parallel for schedule(guided)
        for (std::size_t i = 0; i < parent_population_.size(); i++) 
        {
            auto& ind = parent_population_[i];
            if (ind.rank) continue;

            std::string ind_id = std::to_string(i);
            std::string ind_path = out_path + "/" + std::string(max_digits - ind_id.length(), '0') + ind_id;
            fs::create_directories(ind_path);

            for (std::size_t l = 0; l < graph_->NumLayers(); l++) {
                const Layer& layer = (*graph_)[l];
                const Gene& gene = ind.genome[l];
                ArchInstance& accelerator = ind.accelerators[gene.acc_instance_id];

                auto engine = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].engine;
                auto& mapping = workload_mappings_[layer.workload_id][accelerator.arch_template_id][gene.mapping_id].genome;
                auto& workload = graph_->GetWorkloads()[layer.workload_id]; 
                auto* sparse_opt = &(*arch_templates_)[accelerator.arch_template_id].sparse_optimizations;
                problem::shape_ = workload.shape; 
                
                engine.Spec(accelerator.arch_specs);
                engine.Evaluate(mapping, workload.workload, sparse_opt);

                std::string l_id = std::to_string(l);
                std::string stats_filename = ind_path + "/moham.stats." + std::string(ind_max_digits - l_id.length(), '0') + l_id + ".txt";
                std::ofstream stats_file(stats_filename);
                stats_file << engine << std::endl;
                stats_file.close();            
            }
        }
    }
    

}