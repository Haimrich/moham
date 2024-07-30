#include "medea.h"

#include <vector>
#include <ostream>
#include <numeric>
#include <chrono>

#ifdef _OPENMP
  #include <omp.h>
#endif
#include "yaml-cpp/yaml.h"
#include "boost/filesystem.hpp"

#include "progressbar.h"
#include "config.h"
#include "graph.h"
#include "archtemplate.h"
#include "accelergy.h"
#include "timeloop.h"

namespace fs = boost::filesystem;

namespace moham 
{
    Medea::Medea(Config config, Workload &workload, ArchTemplate &arch_template) :
        config_(config),
        workload_(workload),
        arch_template_(arch_template),
        uni_distribution_(0, 1),
        exp_distribution_(3.5),
        tour_distribution_(0, config_.medea.population_size - 1)
    {
        workload_.workload.SetDefaultDenseTensorFlag(arch_template_.sparse_optimizations.compression_info.all_ranks_default_dense);
        bool filter_spatial_fanout = arch_template_.sparse_optimizations.action_spatial_skipping_info.size() == 0;
        
        std::cout.setstate(std::ios_base::failbit); std::cerr.setstate(std::ios_base::failbit);
        
        global_mapspace_ = timeloop::ParseMapSpace(
            timeloop::CompoundConfigNode(),
            arch_template_.mapping_constraints_config[workload_.type], 
            arch_template_.arch_specs,
            workload_.workload,
            filter_spatial_fanout
        );

        mapspace_ = global_mapspace_->Split(1)[0];
        
        constraints_ = new timeloop::Constraints(arch_template_.arch_props, workload_.workload);
        constraints_->Parse(arch_template_.mapping_constraints_config[workload_.type]);

        std::cout.clear(); std::cerr.clear(); 

        std::cout << "🔥 MEDEA Mapping Search" << std::endl;
        std::cout << "   Workload - Id: " << workload_.id << " | Type: " << (unsigned)workload_.type << " | Name: " << workload_.Name() << std::endl;
        std::cout << "   Template: " << arch_template.name << " - Mapspace Approx. Size: " << std::scientific << mapspace_->Size().convert_to<double>() << std::endl;
        
        population_.resize(config.medea.population_size);
        parent_population_.resize(config.medea.population_size);
        immigrant_population_.resize(config.medea.immigrant_population_size);
        merged_population_.resize(2 * config.medea.population_size + config.medea.immigrant_population_size);
        
        if_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::IndexFactorization));
        lp_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::LoopPermutation));
        db_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::DatatypeBypass));
        sp_rng_ = new RandomGenerator128(mapspace_->Size(mapspace::Dimension::Spatial));
        
        std::random_device rand_dev;
        rng_ = std::mt19937_64(config.seed ? config.seed : rand_dev());

        if (config.medea.num_threads > 0) {
            omp_set_num_threads(config_.medea.num_threads);
        }

        gen_stability_measures_.resize(config_.medea.stability_window);

        // Global shape inizialization
        problem::shape_ = workload_.shape;
    }

    Medea::Population Medea::Run(std::string medea_out_path) {
        // === SETUP ===
        #pragma omp parallel copyin(problem::shape_)
        { (void) problem::shape_; }

        auto chrono_start = std::chrono::high_resolution_clock::now();

        const std::string stats_filename = "medea.stats.txt";
        const std::string pop_filename = "medea.populations.csv";
        const std::string workload_filename = "medea.workload.yaml";

        std::ofstream population_file(medea_out_path + pop_filename);

        // === INITIAL POPULATION ===

        double rpt = RandomPopulation(parent_population_);
        if (rpt < 0)
        {
            std::cout << "   Failed to generate random population." << std::endl; 
            return Population();
        }
        AssignRankAndCrowdingDistance(parent_population_);
        std::cout << "   Initial | " << std::fixed << std::setprecision(2) << rpt << " s" << std::endl;

        // === MAIN LOOP ===

        bool convergence_flag = false;

        for (unsigned g = 0; !convergence_flag && (g < config_.medea.num_generations && config_.medea.num_generations); g++) 
        {
            std::stringstream ss;
            ss << "   G " << std::right << std::setfill('0') << std::setw(3) << g << "   ";

            unsigned success_cross_count = 0;
            
            ProgressBar bar;
            bar.start(population_.size(), ss.str());

            #pragma omp parallel for schedule(guided) copyin(problem::shape_)
            for (unsigned i = 0; i < population_.size(); i += 2)
            {
                std::size_t parent_a = config_.medea.use_tournament ? Tournament() : i;
                std::size_t parent_b = config_.medea.use_tournament ? Tournament() : i + 1;

                Crossover(parent_population_[parent_a].genome, parent_population_[parent_b].genome,
                        population_[i].genome, population_[i + 1].genome);

                /*
                std::cout << "RPA: " << parent_population_[parent_a].genome.loop_nest.PrintCompact(tiling::TransposeMasks(parent_population_[parent_a].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "CPA: " << parent_population_[parent_a].genome.complete_loop_nest.PrintCompact(tiling::TransposeMasks(parent_population_[parent_a].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "ROA: " << population_[i].genome.loop_nest.PrintCompact(tiling::TransposeMasks(population_[i].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "COA: " << population_[i].genome.complete_loop_nest.PrintCompact(tiling::TransposeMasks(population_[i].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "RPB: " << parent_population_[parent_b].genome.loop_nest.PrintCompact(tiling::TransposeMasks(parent_population_[parent_b].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "CPB: " << parent_population_[parent_b].genome.complete_loop_nest.PrintCompact(tiling::TransposeMasks(parent_population_[parent_b].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "ROB: " << population_[i+1].genome.loop_nest.PrintCompact(tiling::TransposeMasks(population_[i+1].genome.datatype_bypass_nest)) << std::endl;
                std::cout << "COB: " << population_[i+1].genome.complete_loop_nest.PrintCompact(tiling::TransposeMasks(population_[i+1].genome.datatype_bypass_nest)) << std::endl;
                */

                for (unsigned j = 0; j < 2; j++) 
                {
                    Mutation(population_[i+j]);

                    if (Evaluate(population_[i+j].genome, population_[i+j])) {
                        #pragma omp atomic update
                        success_cross_count++;
                    } 
                    else if (config_.medea.random_when_illegal) {
                        RandomIndividual(i+j, population_);
                    } 

                    ++bar;
                }
            }
            bar.stop();

            RandomPopulation(immigrant_population_);

            Merging(merged_population_, parent_population_, population_, immigrant_population_);
            AssignRankAndCrowdingDistance(merged_population_);
            Survival(parent_population_, merged_population_, !config_.medea.use_tournament, rng_);

            // Print Gen Info   
            std::cout << ss.str();
            ss.str(""); ss << "| " << std::fixed << std::setprecision(2) << bar.time_it_took() << " s";
            std::cout << std::setw(15) << ss.str();
            ss.str(""); ss << "| " << success_cross_count << "/" << std::left << population_.size();
            std::cout << std::setw(15) << ss.str() << "| R  " << RankString(parent_population_);

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
        population_file.close();

        std::ofstream workload_file(medea_out_path + workload_filename);
        workload_file << workload_.Yaml() << std::endl;
        workload_file.close();

        OutputParetoFrontFiles(medea_out_path + "pareto/");

        auto chrono_end = std::chrono::high_resolution_clock::now();
        auto chrono_duration = std::chrono::duration_cast<std::chrono::seconds>(chrono_end - chrono_start).count();

        std::ofstream stats_file(medea_out_path + stats_filename);
        stats_file << "Search time: " << chrono_duration << " seconds" << std::endl;
        stats_file.close();

        std::cout << "   Search time: " << chrono_duration << " seconds" << std::endl;
        
        Population pareto;
        std::copy_if(parent_population_.begin(), parent_population_.end(), std::back_inserter(pareto), [](auto& ind){ return ind.rank == 0; });
        return pareto;
    }

    Medea::Population Medea::Parse(std::string pareto_path) {
        #pragma omp parallel copyin(problem::shape_)
        { (void) problem::shape_; }
        
        Population pareto;
        
        std::vector<fs::path> individual_filepaths;
        for (const auto &p : fs::directory_iterator(pareto_path))
            if ( fs::is_regular_file(p) && p.path().extension() == ".yaml" )
                individual_filepaths.emplace_back(p);
        
        std::sort(individual_filepaths.begin(), individual_filepaths.end());
        pareto.reserve(individual_filepaths.size());
        
        #pragma omp parallel for copyin(problem::shape_)
        for (Individual::ID i = 0; i < individual_filepaths.size(); i++) 
        {
            timeloop::CompoundConfig ind_config(individual_filepaths[i].c_str());
            auto root = ind_config.getRoot();

            Individual ind;
            ind.arch_template_id = arch_template_.id;
            
            #pragma omp critical (parse_mapping)
            {
                std::cout.setstate(std::ios_base::failbit); std::cerr.setstate(std::ios_base::failbit);
                ind.genome = timeloop::ParseMapping(root.lookup("mapping"), arch_template_.arch_specs, workload_.workload);
                std::cout.clear(); std::cerr.clear();
            }

            
            /*
            double area;
            root.lookup("stats").lookupValue("area", area);
            
            auto arch = MinimalArchSpecs(root.lookup("arch").getYNode());
            
            static unsigned count = 0;
            if (arch != ind.arch) 
            {
                
                YAML::Emitter em;
                //em << arch << root.lookup("mapping").getYNode();
                //em << ind.arch;
                //ind.genome.FormatAsYaml(em, arch_template_.arch_specs.topology.StorageLevelNames());
                em << arch << ind.arch;
                std::cout << em.c_str() << std::endl;
                std::cout << area << " " << ind.objectives[2] << std::endl;
                
               std::cout << ++count << std::endl;
               if (arch_template_.name == "eyeriss") assert(arch == ind.arch); 
            }
            */
            
           // TODO fix bugs
            
            //assert(Evaluate(ind.genome, ind));
            if ( Evaluate(ind.genome, ind) ) {
                #pragma omp critical (parse_pareto)
                pareto.push_back(ind);
            }
        }

        std::cout << "   Found " << pareto.size() << "/" << individual_filepaths.size() << " individuals in output folder." << std::endl;
        return pareto;
    }

    Medea::~Medea() 
    {
        if (global_mapspace_) delete global_mapspace_;
        if (constraints_) delete constraints_;

        if (if_rng_) delete if_rng_;
        if (lp_rng_) delete lp_rng_;
        if (db_rng_) delete db_rng_;
        if (sp_rng_) delete sp_rng_;
    };

    double Medea::RandomPopulation(Population &pop)
    {
        ProgressBar bar;
        bar.start(pop.size(), "   ");

        bool success = true;
        #pragma omp parallel for copyin(problem::shape_) schedule(guided)
        for (std::size_t p = 0; p < pop.size(); p++) 
        {
            if (!success) continue;
            bool valid_mapping;
            unsigned tries;
            for (tries = 100000; tries != 0; tries--) 
            {
                Mapping mapping;
                
                #pragma omp critical (random_mapping)
                valid_mapping = RandomMapping(&mapping);        

                if (valid_mapping)
                    if ( Evaluate(mapping, pop[p]) ) 
                        break;
            } 
            if (tries == 0) {
                success = false;
            }
            ++bar;
        }

        bar.stop();
        return success ? bar.time_it_took() : -1.0;
    }


    void Medea::AppendGenerationInfoToFile(std::ofstream &out, Population &pop, uint64_t gen_id)
    {
        for (Individual &ind : pop)
        {
            out << gen_id << "," << ind.rank << "," << ind.crowding_distance << "," << ind.objectives[0] << "," << ind.objectives[1] << "," << ind.objectives[2] << std::endl;
        }

        out.flush();
    }


    void Medea::OutputParetoFrontFiles(std::string out_path)
    {
        int max_digits = std::to_string(config_.medea.population_size).length();

        unsigned count = 0;
        for (auto &ind : parent_population_)
        {
            if (ind.rank)
                continue;

            std::string ind_id = std::to_string(count);

            std::string stats_filename = out_path + "medea.stats." + std::string(max_digits - ind_id.length(), '0') + ind_id + ".txt";
            std::ofstream stats_file(stats_filename);
            stats_file << ind.engine << std::endl;
            stats_file.close();

            std::string stats_yaml_filename = out_path + "medea.stats." + std::string(max_digits - ind_id.length(), '0') + ind_id + ".yaml";
            std::ofstream stats_yaml_file(stats_yaml_filename);

            stats_yaml_file << ind << std::endl;

            YAML::Emitter mapping_yaml;
            mapping_yaml << YAML::BeginMap << YAML::Key << "mapping" << YAML::Value << YAML::BeginSeq;
            ind.genome.FormatAsYaml(mapping_yaml, arch_template_.arch_specs.topology.StorageLevelNames());
            mapping_yaml << YAML::EndSeq << YAML::EndMap;
            stats_yaml_file << mapping_yaml.c_str() << std::endl;

            stats_yaml_file.close();

            count++;
        }
    }


    // ================================================================================== //

    bool Medea::EngineSuccess(std::vector<model::EvalStatus> &status_per_level)
    {
        return std::accumulate(status_per_level.begin(), status_per_level.end(), true,
                               [](bool cur, const model::EvalStatus &status)
                               { return cur && status.success; });
    }

    bool Medea::RandomMapping(Mapping *mapping)
    {
        // TODO brutta cosa da sistemare

        uint128_t if_id = if_rng_->Next() % mapspace_->Size(mapspace::Dimension::IndexFactorization);
        mapspace_->InitPruned(if_id);
        
        mapspace::ID mapping_id = mapspace::ID(mapspace_->AllSizes());
        mapping_id.Set(unsigned(mapspace::Dimension::IndexFactorization), if_id);
        mapping_id.Set(unsigned(mapspace::Dimension::LoopPermutation), lp_rng_->Next() % mapspace_->Size(mapspace::Dimension::LoopPermutation));
        mapping_id.Set(unsigned(mapspace::Dimension::DatatypeBypass), db_rng_->Next() % mapspace_->Size(mapspace::Dimension::DatatypeBypass));
        mapping_id.Set(unsigned(mapspace::Dimension::Spatial), sp_rng_->Next() % mapspace_->Size(mapspace::Dimension::Spatial));
        
        auto construction_status = mapspace_->ConstructMapping(mapping_id, mapping);
        

        return std::accumulate(construction_status.begin(), construction_status.end(), true,
                               [](bool cur, const mapspace::Status &status)
                               { return cur && status.success; });
    }

    LoopRange Medea::GetSubnestRangeAtLevel(const Mapping &mapping, unsigned level)
    {
        size_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
        size_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;
        return LoopRange(mapping.loop_nest.loops.begin() + start, mapping.loop_nest.loops.begin() + end);
    }

    uint64_t Medea::GetParallelAtLevel(const Mapping &mapping, spacetime::Dimension dim, uint64_t level)
    {
        uint64_t result = 1;
        LoopRange subnest = GetSubnestRangeAtLevel(mapping, level);
        for (auto &l : subnest)
            if (l.spacetime_dimension == dim)
                result *= l.end;
        return result;
    }

    std::vector<loop::Descriptor> Medea::GetSubnestAtLevel(const Mapping &mapping, unsigned level)
    {
        LoopRange subnest_range = GetSubnestRangeAtLevel(mapping, level);
        return std::vector<loop::Descriptor>(subnest_range.begin(), subnest_range.end());
    }

    uint64_t Medea::GetDimFactorInSubnest(problem::Shape::FlattenedDimensionID dimension, std::vector<loop::Descriptor> &subnest)
    {
        uint64_t factor = 1;
        for (auto &l : subnest)
            if (l.dimension == dimension)
                factor *= l.end;
        return factor;
    }

    uint64_t Medea::GetStrideInSubnest(problem::Shape::FlattenedDimensionID dimension, std::vector<loop::Descriptor> &subnest)
    {
        for (auto &l : subnest)
            if (l.dimension == dimension)
                return l.stride;

        return 1;
    }

    void Medea::UpdateArchitecture(Mapping &mapping, model::Engine &engine, MinimalArchSpecs &arch)
    {

        std::unordered_map<std::string, uint64_t> updates;

        auto new_specs = model::Topology::Specs(arch_template_.arch_specs.topology);
        for (unsigned i = 0; i < arch_template_.arch_specs.topology.NumStorageLevels(); i++)
        {
            auto buffer = new_specs.GetStorageLevel(i);

            auto utilized_capacity_pds = engine.GetTopology().GetStats().utilized_capacities.at(i);
            auto utilized_capacity = std::accumulate(utilized_capacity_pds.begin(), utilized_capacity_pds.end(), 0);

            if (!buffer->block_size.IsSpecified())
                continue;
            auto block_size = buffer->block_size.Get();

            if (!buffer->size.IsSpecified())
                continue;

            unsigned needed_depth = (utilized_capacity / block_size) + 1;
            unsigned remainder = needed_depth % config_.medea.buffer_update_granularity;
            unsigned new_depth = remainder ? needed_depth + config_.medea.buffer_update_granularity - remainder : needed_depth;

            buffer->size = new_depth * block_size;
            buffer->effective_size = static_cast<uint64_t>(std::floor(buffer->size.Get() / buffer->multiple_buffering.Get()));
            updates[buffer->name.Get()] = new_depth;
        }

        int i;

        for (i = arch_template_.arch_specs.topology.NumLevels() - 2; i > 0; i--)
        {
            auto buffer = new_specs.GetStorageLevel(i - 1);

            buffer->meshX = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceX, i) * new_specs.GetStorageLevel(i)->meshX.Get();
            buffer->meshY = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceY, i) * new_specs.GetStorageLevel(i)->meshY.Get();
            buffer->instances = buffer->meshX.Get() * buffer->meshY.Get();
        }

        if (i == 0)
        {
            auto arithmetic = new_specs.GetArithmeticLevel();

            arithmetic->meshX = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceX, i) * new_specs.GetStorageLevel(i)->meshX.Get();
            arithmetic->meshY = GetParallelAtLevel(mapping, spacetime::Dimension::SpaceY, i) * new_specs.GetStorageLevel(i)->meshY.Get();
            arithmetic->instances = arithmetic->meshX.Get() * arithmetic->meshY.Get();
        }

        std::string out_prefix = "medea." + std::to_string(omp_get_thread_num()) + "_tmp";
        Accelergy::RT rt = arch_template_.accelergy.GetReferenceTables(updates, out_prefix);

        model::Engine::Specs new_engine_specs;
        new_engine_specs.topology = new_specs;
        new_engine_specs.topology.ParseAccelergyART(rt.area);
        if (config_.medea.update_ert)
            new_engine_specs.topology.ParseAccelergyERT(rt.energy);
        engine.Spec(new_engine_specs);

        // Architectural updates for negotiator
        arch.Reset(arch_template_.arch_specs.topology.NumLevels());
        auto arithmetic = new_specs.GetArithmeticLevel();
        arch.Add(arithmetic->name.Get(), arithmetic->meshX.Get(), arithmetic->meshY.Get());

        for (unsigned j = 0; j < arch_template_.arch_specs.topology.NumStorageLevels(); j++)
        {
            auto buffer = new_specs.GetStorageLevel(j);
            if (buffer->size.IsSpecified())
                arch.Add(buffer->name.Get(), buffer->meshX.Get(), buffer->meshY.Get(), buffer->size.Get());
        }
    }

    bool Medea::Evaluate(Mapping mapping, Individual &individual)
    {
        individual.valid = false;
        
        // Sanity Checks
        for (unsigned l = 0; l < arch_template_.arch_props.FanoutX().size(); l++)
            if (GetParallelAtLevel(mapping, spacetime::Dimension::SpaceX, l) > arch_template_.arch_props.FanoutX().at(l))
                return false;
        for (unsigned l = 0; l < arch_template_.arch_props.FanoutY().size(); l++)
            if (GetParallelAtLevel(mapping, spacetime::Dimension::SpaceY, l) > arch_template_.arch_props.FanoutY().at(l))
                return false; 

        /*
        #pragma omp critical 
        {
            timeloop::Constraints other(arch_template_.arch_props, workload_.workload);
            other.Generate(&mapping);
            for (auto p : constraints_->Permutations())
                std::cout << p.first << " ";
            std::cout << std::endl;
            for (auto p : other.Permutations())
                std::cout << p.first << " ";
            std::cout << std::endl;
            std::cout << mapping.PrintCompact() << std::endl;
            if ( !constraints_->SatisfiedBy(&mapping) ) {
                std::cout << "non funziona niente" << std::endl;
            }
        }
        */

        timeloop::Engine engine;
        engine.Spec(arch_template_.arch_specs);

        // Lightweight pre-eval
        auto status_per_level = engine.PreEvaluationCheck(mapping, workload_.workload, &arch_template_.sparse_optimizations);
        if (!EngineSuccess(status_per_level))
            return false;

        // Heavyweight evaluation
        status_per_level = engine.Evaluate(mapping, workload_.workload, &arch_template_.sparse_optimizations);
        if (!EngineSuccess(status_per_level))
            return false;

        // Update storage capacities based on mappping -> update area
        UpdateArchitecture(mapping, engine, individual.arch);        
        status_per_level = engine.Evaluate(mapping, workload_.workload, &arch_template_.sparse_optimizations);
        assert(EngineSuccess(status_per_level));        

        // Population update
        individual.genome = mapping;
        individual.objectives[0] = engine.Energy();
        individual.objectives[1] = (double)engine.Cycles();
        individual.objectives[2] = engine.Area();
        individual.engine = engine;
        individual.arch_template_id = arch_template_.id;
        individual.valid = true;

        return true;
    }

    void Medea::FactorCompensation(const problem::Shape::FlattenedDimensionID &dim, const uint64_t stride, const uint64_t old_factor, const uint64_t new_factor, const uint64_t level, loop::Nest &nest)
    {

        if (new_factor < old_factor)
        {
            // Prima passare da old_factor a 1 poi da 1 a new_factor -> ricorsivo
            if (old_factor % new_factor)
            {
                FactorCompensation(dim, stride, old_factor, 1, level, nest);
                FactorCompensation(dim, stride, 1, new_factor, level, nest);
                return;
            }

            // Fattore diminuito -> compensiamo aumentando in RAM.
            int64_t factor = old_factor / new_factor;

            int64_t ram_level = nest.storage_tiling_boundaries.size() - 2;
            uint64_t ram_start = ram_level > 0 ? nest.storage_tiling_boundaries.at(ram_level) + 1 : 0;

            auto ram_loop = std::find_if(nest.loops.begin() + ram_start, nest.loops.end(), [&](const loop::Descriptor &x)
                                         { return x.dimension == dim; });

            if (ram_loop != nest.loops.end())
            {
                ram_loop->end *= factor;
                ram_loop->residual_end = ram_loop->end;
            }
            else
            {
                loop::Descriptor new_loop(dim, 0, factor, stride, spacetime::Dimension::Time);
                nest.loops.push_back(new_loop);
                nest.storage_tiling_boundaries.back()++;
            }
        }
        else if (new_factor > old_factor)
        {
            // Fattore aumentato -> Compensiamo diminuendo in RAM o nel primo che troviamo a scendere
            if (new_factor % old_factor)
            {
                FactorCompensation(dim, stride, old_factor, 1, level, nest);
                FactorCompensation(dim, stride, 1, new_factor, level, nest);
                return;
            }

            int64_t factor = new_factor / old_factor;

            for (int64_t l = nest.storage_tiling_boundaries.size() - 1; l >= 0 && factor != 1; l--)
            {
                // Cerca fattore da ridurre (escluso livello in cui l'abbiamo incrementato)

                if (l != (int64_t)level)
                {
                    uint64_t l_start = l > 0 ? nest.storage_tiling_boundaries.at(l - 1) + 1 : 0;
                    uint64_t l_end = nest.storage_tiling_boundaries.at(l) + 1;

                    for (auto l_loop = nest.loops.begin() + l_start; l_loop != nest.loops.begin() + l_end && factor != 1; l_loop++)
                    {
                        if (l_loop->dimension == dim)
                        {
                            uint64_t common = GCD(factor, l_loop->end);

                            factor /= common;
                            l_loop->end /= common;
                            l_loop->residual_end = l_loop->end;
                        }
                    }
                }
            }
        }
    }

    void Medea::Crossover(const Mapping &parent_a, const Mapping &parent_b, Mapping &offspring_a, Mapping &offspring_b)
    {
        uint64_t level;
        #pragma omp critical 
        {
            offspring_a = parent_a;
            offspring_b = parent_b;

            level = rng_() % (parent_a.loop_nest.storage_tiling_boundaries.size() - 1);
        }

        loop::Nest nest_a = parent_a.loop_nest;
        uint64_t a_start = level > 0 ? nest_a.storage_tiling_boundaries.at(level - 1) + 1 : 0;
        uint64_t a_end = nest_a.storage_tiling_boundaries.at(level) + 1;
        std::vector<loop::Descriptor> a_level(nest_a.loops.begin() + a_start, nest_a.loops.begin() + a_end);

        loop::Nest nest_b = parent_b.loop_nest;
        uint64_t b_start = level > 0 ? nest_b.storage_tiling_boundaries.at(level - 1) + 1 : 0;
        uint64_t b_end = nest_b.storage_tiling_boundaries.at(level) + 1;
        std::vector<loop::Descriptor> b_level(nest_b.loops.begin() + b_start, nest_b.loops.begin() + b_end);

        // Factor compensation
        for (unsigned idim = 0; idim < timeloop::GetWorkloadShape()->NumFactorizedDimensions; idim++)
        {
            problem::Shape::FlattenedDimensionID dimension = problem::Shape::FlattenedDimensionID(idim);

            uint64_t factor_a = GetDimFactorInSubnest(dimension, a_level);
            uint64_t factor_b = GetDimFactorInSubnest(dimension, b_level);
            uint64_t stride_a = GetStrideInSubnest(dimension, a_level);
            uint64_t stride_b = GetStrideInSubnest(dimension, b_level);

            FactorCompensation(dimension, stride_a, factor_a, factor_b, level, offspring_a.loop_nest);
            FactorCompensation(dimension, stride_b, factor_b, factor_a, level, offspring_b.loop_nest);
        }

        LoopRange range_a = GetSubnestRangeAtLevel(offspring_a, level);
        LoopRange range_b = GetSubnestRangeAtLevel(offspring_b, level);

        offspring_a.loop_nest.loops.erase(range_a.begin(), range_a.end());
        offspring_a.loop_nest.loops.insert(range_a.begin(), b_level.begin(), b_level.end());

        offspring_b.loop_nest.loops.erase(range_b.begin(), range_b.end());
        offspring_b.loop_nest.loops.insert(range_b.begin(), a_level.begin(), a_level.end());

        int64_t diff = a_level.size() - b_level.size();
#ifdef DNABUG
        std::cout << "DIFF: " << diff << std::endl;
#endif
        for (unsigned i = level; i < offspring_a.loop_nest.storage_tiling_boundaries.size(); i++)
            offspring_a.loop_nest.storage_tiling_boundaries[i] -= diff;

        for (unsigned i = level; i < offspring_b.loop_nest.storage_tiling_boundaries.size(); i++)
            offspring_b.loop_nest.storage_tiling_boundaries[i] += diff;

        // Swap datatype bypass
        for (unsigned pvi = 0; pvi < unsigned(problem::GetShape()->NumDataSpaces); pvi++)
        {
            bool bit_a = offspring_a.datatype_bypass_nest.at(pvi).test(level);
            bool bit_b = offspring_b.datatype_bypass_nest.at(pvi).test(level);

            if (bit_a)
                offspring_b.datatype_bypass_nest.at(pvi).set(level);
            else
                offspring_b.datatype_bypass_nest.at(pvi).reset(level);

            if (bit_b)
                offspring_a.datatype_bypass_nest.at(pvi).set(level);
            else
                offspring_a.datatype_bypass_nest.at(pvi).reset(level);
        }
    }

    void Medea::FanoutMutation(Mapping &mapping)
    {
        // Set spatial loops bounds to maximum possible
        for (uint32_t level = 0; level < mapping.loop_nest.storage_tiling_boundaries.size(); level++)
        {
            if (arch_template_.arch_props.Fanout(level) <= 1)
                continue;

            bool is_constrained;
            std::map<problem::Shape::FlattenedDimensionID, int> factors;
            try
            {
                auto tiling_level_id = arch_template_.arch_props.SpatialToTiling(level);
                factors = constraints_->Factors().at(tiling_level_id);
                is_constrained = true;
            }
            catch (const std::out_of_range &oor)
            {
                is_constrained = false;
            }

            std::vector<loop::Descriptor> level_nest = GetSubnestAtLevel(mapping, level);

            bool x_loop_found = false;
            bool y_loop_found = false;
            uint32_t x_product = 1;
            uint32_t y_product = 1;
            for (auto &s : level_nest)
            {
                if (s.spacetime_dimension == spacetime::Dimension::SpaceX)
                {
                    x_product *= s.end;
                    x_loop_found = true;
                }
                else if (s.spacetime_dimension == spacetime::Dimension::SpaceY)
                {
                    y_product *= s.end;
                    y_loop_found = true;
                }
            }

            if (x_loop_found || y_loop_found)
            {
                for (auto &s : level_nest)
                {
                    if (s.spacetime_dimension == spacetime::Dimension::Time)
                        continue;

                    uint64_t new_factor = 0;
                    if (is_constrained)
                    {
                        auto factor = factors.find(s.dimension);
                        if (factor != factors.end())
                        {
                            new_factor = factor->second;

                            if (s.spacetime_dimension == spacetime::Dimension::SpaceX && x_product * new_factor / s.end > arch_template_.arch_props.FanoutX(level))
                                continue;
                            if (s.spacetime_dimension == spacetime::Dimension::SpaceY && y_product * new_factor / s.end > arch_template_.arch_props.FanoutY(level))
                                continue;
                        }
                    }

                    if (new_factor == 0)
                    {
                        // std::cout << "FM_P: " << mapping.PrintCompact() << std::endl;
                        new_factor = s.spacetime_dimension == spacetime::Dimension::SpaceX ? arch_template_.arch_props.FanoutX(level) / (x_product / s.end) : arch_template_.arch_props.FanoutY(level) / (y_product / s.end);

                        if (new_factor == 0)
                            return;

                        if ((uint64_t)workload_.workload.GetFactorizedBound(s.dimension) < new_factor)
                            new_factor = workload_.workload.GetFactorizedBound(s.dimension);
                        else if (workload_.workload.GetFactorizedBound(s.dimension) % new_factor)
                            continue;
                        // TODO - Find greatest divisor of workload_.GetBound(s->dimension) less than Fanout (new_factor)
                    }

                    uint64_t old_factor = s.end;
                    s.end = new_factor;
                    s.residual_end = s.end;

                    FactorCompensation(s.dimension, s.stride, old_factor, s.end, level, mapping.loop_nest);
                }

                LoopRange range = GetSubnestRangeAtLevel(mapping, level);
                mapping.loop_nest.loops.erase(range.begin(), range.end());
                mapping.loop_nest.loops.insert(range.begin(), level_nest.begin(), level_nest.end());
            }

            if (!y_loop_found && level_nest.size() > 1)
            {
                /*
            std::cout << "FANOUT LEVEL " << level << " Size: " << level_nest.size() << std::endl;
            for (auto l : level_nest)
                std::cout << l;
            for (uint32_t level = 0; level < mapping.loop_nest.storage_tiling_boundaries.size(); level++)
                std::cout << " " << arch_props_.Fanout(level);
            std::cout << std::endl << "FM_P: " << mapping.PrintCompact() << std::endl;
            */

                unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
                level_nest = GetSubnestAtLevel(mapping, level);

                int loop_tc = -1;
                for (unsigned j = 0; j < level_nest.size(); j++)
                {
                    if ((unsigned)level_nest[j].end <= arch_template_.arch_props.FanoutY(level) &&
                        level_nest[j].end > 1 &&
                        (loop_tc == -1 || level_nest[j].end > level_nest[loop_tc].end))
                        loop_tc = j;
                }

                if (loop_tc > -1)
                {
                    mapping.loop_nest.loops.at(start + loop_tc).spacetime_dimension = spacetime::Dimension::SpaceY;
                    if (loop_tc != 0)
                        std::swap(mapping.loop_nest.loops.at(start + loop_tc), mapping.loop_nest.loops.at(start));
                }
            }
            // std::cout << "FM_D: " << mapping.PrintCompact() << std::endl;
        }
    }

    // Fill buffer at lower levels - Funziona solo con quelli che contengono un solo datatype per ora forse
    void Medea::FillMutation(model::Engine &engine, Mapping &mapping)
    {
        unsigned level = unsigned((arch_template_.arch_props.StorageLevels() - 1) * exp_distribution_(rng_));
        level = (level == arch_template_.arch_props.StorageLevels() - 1) ? arch_template_.arch_props.StorageLevels() - 2 : level;

        // Vedere bypass
        unsigned n_datatypes_in_buffer = 0;
        unsigned datatype = 0;

        for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
            if (mapping.datatype_bypass_nest[pv][level])
            {
                n_datatypes_in_buffer++;
                datatype = pv;
            }

        if (n_datatypes_in_buffer != 1)
            return; // Non supportato

        auto utilized_capacity = engine.GetTopology().GetStats().utilized_capacities.at(level).at(datatype);
        auto buffer_capacity = arch_template_.arch_specs.topology.GetStorageLevel(level)->size.Get();

        // FIXME - questa cosa non ha senso con l'input dataspace o comunque quando gli indici sono composti
        // Va considerato lo stride la dilation e tante altre cose...
        uint64_t factor_needed = utilized_capacity > 0 ? buffer_capacity / utilized_capacity : 1;
        if (factor_needed == 1)
        
            return;

        uint64_t start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
        uint64_t end = mapping.loop_nest.storage_tiling_boundaries.at(level) + 1;

        for (auto l = mapping.loop_nest.loops.begin() + start; l != mapping.loop_nest.loops.begin() + end; l++)
        {
            if (l->spacetime_dimension != spacetime::Dimension::Time)
                continue;

            // Verificare se dimensione appartiene a daataspace
            bool is_proj = false;
            for (auto &proj : problem::GetShape()->Projections[datatype])
                for (auto &projex : proj)
                    if (projex.first == problem::GetShape()->NumCoefficients && projex.second == l->dimension)
                        is_proj = true;

            if (!is_proj)
                continue;

            // Trovo fattore divisore della dimensione del workload, in maniera un po' brutta
            uint64_t factor_div = factor_needed;

            uint64_t max_factor = workload_.workload.GetFactorizedBound(l->dimension) / l->end;
            if (max_factor < factor_needed)
                continue;

            while (max_factor % factor_div)
                factor_div--;

            if (factor_div == 1)
                continue;

            uint64_t factor_avaiable = 1;
            for (auto ln = l + 1; ln != mapping.loop_nest.loops.end(); ln++)
                if (l->dimension == ln->dimension)
                    factor_avaiable *= l->end;

            if (factor_div > factor_avaiable)
                continue;

            uint64_t old_factor = l->end;
            l->end *= factor_div;
            l->residual_end = l->end;

            FactorCompensation(l->dimension, l->stride, old_factor, l->end, level, mapping.loop_nest);
            //std::cout << "Fill Mutation" << std::endl;
        }

        // 3D conv only :(

        // TODO INPUT DATATYPE
        // W = Stride * (P - 1) + Dilation * (R - 1)
        // H = Stride * (Q - 1) + Dilation * (S - 1)
    }

    void Medea::RandomMutation(Mapping &mapping)
    {
        // Random loop factor swapping
        if (uni_distribution_(rng_) < 0.5)
        {

            unsigned num_levels = mapping.loop_nest.storage_tiling_boundaries.size() - 1;
            unsigned level_a = unsigned(num_levels * uni_distribution_(rng_));
            unsigned level_b = unsigned(num_levels * uni_distribution_(rng_));
            if (level_a == level_b)
                return;

            auto level_a_nest = GetSubnestAtLevel(mapping, level_a);
            unsigned loop_a = unsigned(level_a_nest.size() * uni_distribution_(rng_));

            auto level_b_nest = GetSubnestAtLevel(mapping, level_b);
            unsigned loop_b = unsigned(level_b_nest.size() * uni_distribution_(rng_));

            if (level_a_nest[loop_a].spacetime_dimension != spacetime::Dimension::Time ||
                level_b_nest[loop_b].spacetime_dimension != spacetime::Dimension::Time)
                return;

            auto dim_a = level_a_nest.at(loop_a).dimension;
            int id_same_dim_in_b = -1;
            for (unsigned l = 0; l < level_b_nest.size(); l++)
                if (level_b_nest[l].dimension == dim_a && level_b_nest[l].spacetime_dimension == spacetime::Dimension::Time)
                    id_same_dim_in_b = l;

            auto dim_b = level_b_nest.at(loop_b).dimension;
            int id_same_dim_in_a = -1;
            for (unsigned l = 0; l < level_a_nest.size(); l++)
                if (level_a_nest[l].dimension == dim_b && level_a_nest[l].spacetime_dimension == spacetime::Dimension::Time)
                    id_same_dim_in_a = l;

            unsigned start_a = level_a > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level_a - 1) + 1 : 0;
            mapping.loop_nest.loops.at(start_a + loop_a).end = 1;
            mapping.loop_nest.loops.at(start_a + loop_a).residual_end = 1;
            
            if (id_same_dim_in_a >= 0)
            {
                mapping.loop_nest.loops.at(start_a + id_same_dim_in_a).end *= level_b_nest[loop_b].end;
                mapping.loop_nest.loops.at(start_a + id_same_dim_in_a).residual_end = mapping.loop_nest.loops.at(start_a + id_same_dim_in_a).end;
            }
            else
            {
                mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start_a + level_a_nest.size() - 1, level_b_nest[loop_b]);

                for (unsigned i = level_a; i < mapping.loop_nest.storage_tiling_boundaries.size(); i++)
                    mapping.loop_nest.storage_tiling_boundaries[i] += 1;
            }


            unsigned start_b = level_b > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level_b - 1) + 1 : 0;
            mapping.loop_nest.loops.at(start_b + loop_b).end = 1;
            mapping.loop_nest.loops.at(start_b + loop_b).residual_end = 1;

            if (id_same_dim_in_b >= 0)
            {
                mapping.loop_nest.loops.at(start_b + id_same_dim_in_b).end *= level_a_nest[loop_a].end;
                mapping.loop_nest.loops.at(start_b + id_same_dim_in_b).residual_end = mapping.loop_nest.loops.at(start_b + id_same_dim_in_b).end;
            }
            else
            {
                mapping.loop_nest.loops.insert(mapping.loop_nest.loops.begin() + start_b + level_b_nest.size() - 1, level_a_nest[loop_a]);

                for (unsigned i = level_b; i < mapping.loop_nest.storage_tiling_boundaries.size(); i++)
                    mapping.loop_nest.storage_tiling_boundaries[i] += 1;
            }

            // Random loop permutation
        }
        else
        {

            unsigned num_levels = mapping.loop_nest.storage_tiling_boundaries.size();
            unsigned level = unsigned(num_levels * uni_distribution_(rng_));
            assert(level < num_levels);

            unsigned start = level > 0 ? mapping.loop_nest.storage_tiling_boundaries.at(level - 1) + 1 : 0;
            auto level_nest = GetSubnestAtLevel(mapping, level);
            unsigned loop_a = start + unsigned(level_nest.size() * uni_distribution_(rng_));
            unsigned loop_b = start + unsigned(level_nest.size() * uni_distribution_(rng_));

            if (loop_a != loop_b &&
                mapping.loop_nest.loops.at(loop_a).spacetime_dimension == spacetime::Dimension::Time &&
                mapping.loop_nest.loops.at(loop_b).spacetime_dimension == spacetime::Dimension::Time)
                std::swap(mapping.loop_nest.loops.at(loop_a), mapping.loop_nest.loops.at(loop_b));
        }
    }

    void Medea::Mutation(Individual &individual)
    {
        if (uni_distribution_(rng_) < config_.medea.fill_mutation_prob && individual.engine.IsEvaluated())
            FillMutation(individual.engine, individual.genome);

        if (uni_distribution_(rng_) < config_.medea.parallel_mutation_prob)
            FanoutMutation(individual.genome);

        if (uni_distribution_(rng_) < config_.medea.random_mutation_prob)
            RandomMutation(individual.genome);
    }

    void Medea::RandomIndividual(uint32_t p, Population &pop)
    {
        bool valid_mapping;

        while(true) 
        {
            Mapping mapping;
            
            #pragma omp critical (random_mapping)
            valid_mapping = RandomMapping(&mapping);        
            
            if (valid_mapping)
                if ( Evaluate(mapping, pop[p]) ) 
                    return;
        } 
    }

    uint64_t Medea::Tournament()
    {
        uint64_t b1 = tour_distribution_(rng_);
        uint64_t b2 = tour_distribution_(rng_);

        if (parent_population_[b1].rank < parent_population_[b2].rank)
        {
            return b1;
        }
        else if (parent_population_[b1].rank == parent_population_[b2].rank)
        {
            if (parent_population_[b1].crowding_distance > parent_population_[b2].crowding_distance)
                return b1;
            else
                return b2;
        }
        else
        {
            return b2;
        }
    }


    std::ostream &operator<<(std::ostream &out, const Medea::Individual &ind)
    {
      YAML::Emitter yout;

      yout << YAML::BeginMap;
      yout << YAML::Key << "stats" << YAML::Value << YAML::BeginMap;
      yout << YAML::Key << "energy" << YAML::Value << ind.engine.Energy();
      yout << YAML::Key << "cycles" << YAML::Value << ind.engine.Cycles();
      yout << YAML::Key << "area" << YAML::Value << ind.engine.Area();
      yout << YAML::EndMap;
      yout << YAML::Key << "arch" << YAML::Value << ind.arch;
      yout << YAML::EndMap;

      out << yout.c_str();
      return out;
    }
}