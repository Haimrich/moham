#ifndef MOHAM_PARSING_H_
#define MOHAM_PARSING_H_

#include <fstream>
#include <cassert>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

#include "onnx.pb.h"

#include "tensor.h"
#include "layer.h"
#include "graph.h"
#include "archtemplate.h"
#include "config.h"
#include "benchmarks.h"

#ifdef _OPENMP
  #include <omp.h>
#else
  #define omp_get_num_threads() 1
  #define omp_get_max_threads() 1
#endif

namespace fs = boost::filesystem;

namespace moham { namespace parsing {

    // === Workload Parsing === //

    inline onnx::GraphProto ParseOnnxGraphProto(std::string onnx_file) {
        std::ifstream input(onnx_file, std::ios::ate | std::ios::binary);
        std::streamsize size = input.tellg();
        input.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        input.read(buffer.data(), size);

        onnx::ModelProto model;
        model.ParseFromArray(buffer.data(), size);

        return model.graph();
    }

    inline LayerType OnnxLayerType(const std::string op_type) {
        if (op_type == "Conv") return LayerType::CONV;
        if (op_type == "Gemm") return LayerType::DENSE;
        if (op_type == "MatMul") return LayerType::MATMUL;
        if (op_type == "ConvTranspose") return LayerType::TRANSPOSED_CONV;
        
        return LayerType::OTHER;
    }


    unsigned LoadOnnxFiles(Graph& graph, std::vector<std::string> onnx_files, std::vector<std::vector<std::string>> skipped_layers) {
        unsigned node_count = 0;

        for (unsigned w = 0; w < onnx_files.size(); w++) 
        {
            auto& workload_onnx = onnx_files[w];
            std::unordered_map<std::string, Tensor> parameter_tensors;
            std::unordered_map<std::string, Tensor> activation_tensors;
            std::unordered_map<std::string, std::vector<Edge>> dependancies;

            onnx::GraphProto model_graph = ParseOnnxGraphProto(workload_onnx);
            std::string model_name = model_graph.name();

            for (auto& input : model_graph.input()) 
            {
                if (input.has_type() && input.type().has_tensor_type() && input.type().tensor_type().has_shape()) 
                {
                    std::vector<size_t> shape;
                    auto shape_proto = input.type().tensor_type().shape();
                    for (int i = 0; i < shape_proto.dim_size(); ++i)
                        shape.push_back(shape_proto.dim(i).dim_value());
                    parameter_tensors[input.name()] = Tensor(input.name(), shape);
                }
            }
                
            for (auto& vinfo : model_graph.value_info()) 
            {
                if (vinfo.has_type() && vinfo.type().has_tensor_type() && vinfo.type().tensor_type().has_shape()) 
                {
                    std::vector<size_t> shape;
                    auto shape_proto = vinfo.type().tensor_type().shape();
                    for (int i = 0; i < shape_proto.dim_size(); ++i)
                        shape.push_back(shape_proto.dim(i).dim_value());
                    activation_tensors[vinfo.name()] = Tensor(vinfo.name(), shape);
                }
            }
            
            // Layer Fusion happens
            for (auto& node : model_graph.node()) 
            {
                // If conv layer new actual node
                if ( (node.op_type() == "Conv" || node.op_type() == "Gemm" || node.op_type() == "MatMul" || node.op_type() == "ConvTranspose") &&
                      std::find(skipped_layers[w].begin(), skipped_layers[w].end(), node.name()) == skipped_layers[w].end() ) {
                    
                    std::vector<Edge> incoming;
                    std::vector<Tensor> parameters;
                    std::vector<Tensor> inputs;
                    std::unordered_map<std::string, Tensor> attributes;

                    for (auto& ins : node.input()) 
                    {
                        // Preceding layer
                        auto dependancy = dependancies.find(ins);
                        if (dependancy != dependancies.end()) {
                            incoming.insert(incoming.end(), dependancy->second.begin(), dependancy->second.end());
                            std::sort( incoming.begin(), incoming.end() );
                            incoming.erase( std::unique( incoming.begin(), incoming.end() ), incoming.end() );       
                            inputs.push_back(activation_tensors[ins]);
                        } else {
                            // Parameter tensors
                            auto param = parameter_tensors.find(ins);
                            if (param != parameter_tensors.end()) {
                                parameters.push_back(param->second);
                            }
                            inputs.push_back(parameter_tensors[ins]);
                        }

                    }

                    // Consider only incoming edge not reachable from other edges
                    std::vector<Edge> incoming_necessary;
                    for (std::size_t d = 0; d < incoming.size(); d++) {
                        Layer::ID targetid = incoming[d].layer->id;
                        bool necessary = true;

                        for (std::size_t dc = 0; dc < incoming.size() && necessary; dc++) {
                            if (d != dc) {
                                Layer::ID startid = incoming[dc].layer->id;
                                std::vector<bool> layer_visited(node_count, false);
                                std::queue<Layer::ID> tovisit;
                                tovisit.push(startid);
                                while (!tovisit.empty() && necessary)
                                {
                                    Layer::ID id = tovisit.front();
                                    layer_visited[id] = true;
                                    if (id == targetid) {
                                        necessary = false;
                                    } else {
                                        tovisit.pop();
                                        for (auto& e : graph[id].incoming)
                                            if (!layer_visited[e.layer->id])
                                                tovisit.push(e.layer->id);
                                    }
                                }  
                            }
                        }
                        if (necessary) incoming_necessary.push_back(incoming[d]);
                    }
                    incoming = incoming_necessary;

                    // Attributes
                    for (auto& att : node.attribute()) 
                    {
                        if (att.type() == att.INTS) 
                        {
                            std::vector<size_t> shape;
                            auto shape_proto = att.ints();
                            shape.insert(shape.end(), shape_proto.begin(), shape_proto.end());
                            attributes[att.name()] = Tensor(att.name(), shape);
                        } 
                        else if (att.type() == att.INT) 
                        {
                            attributes[att.name()] = Tensor(att.name(), {(std::size_t)att.i()});
                        }
                        
                    }

                    LayerType type = OnnxLayerType(node.op_type());
                    Layer * layer_ptr = graph.AddNode(node.name(), model_name, type, attributes, parameters, incoming, inputs, activation_tensors[node.output()[0]]);
                    
                    for (auto& ous : node.output())
                        dependancies[ous] = std::vector<Edge>(1, {layer_ptr, activation_tensors[ous]});

                    node_count++;

                // Else merge with the previous one
                } else {
                    std::vector<Edge> incoming;

                    for (auto& ins : node.input()) {
                        auto dependancy = dependancies.find(ins);
                        if (dependancy != dependancies.end()) {
                            incoming.insert(incoming.end(), dependancy->second.begin(), dependancy->second.end());
                            std::sort( incoming.begin(), incoming.end() );
                            incoming.erase( std::unique( incoming.begin(), incoming.end() ), incoming.end() ); 
                        }
                    }

                    for (auto& ous : node.output())
                        dependancies[ous] = incoming;
                }
            }
        }

        return node_count;
    }

    unsigned LoadBenchmarks(Graph& graph, std::vector<std::string> benchmark_names, std::vector<std::vector<std::string>> skipped_layers) {
        assert(benchmark_names.size() == skipped_layers.size());

        unsigned node_count = 0;
        
        for (unsigned i = 0; i < benchmark_names.size(); i++) {
            auto func = benchmarks::benchmarksFunc.find(benchmark_names[i]);
            if (func == benchmarks::benchmarksFunc.end()) {
                std::cout << "'" << benchmark_names[i] << "' benchmark network model not supported." << std::endl;
                exit(1);
            }

            unsigned res = func->second(graph, skipped_layers[i]);
            node_count += res;
        }

        return node_count;
    }

    Graph ParseWorkloads(std::string config_file) 
    {
        std::cout << "⚙️  Parsing workload graph ..." << std::endl;

        // Reading ONNX filepaths from general yaml config file.
        YAML::Node config = YAML::LoadFile(config_file);
        auto workloads_node = config["workloads"];

        std::vector<std::string> onnx_files;
        std::vector<std::vector<std::string>> onnx_skipped_layers;
        std::vector<std::string> benchmarks;
        std::vector<std::vector<std::string>> benchmarks_skipped_layers;


        if ( workloads_node.IsSequence() ) {
            
            for (size_t i = 0; i < workloads_node.size(); i++) {
                if (workloads_node[i]["onnx"].IsDefined()) {
                    onnx_files.push_back(workloads_node[i]["onnx"].as<std::string>());
                    onnx_skipped_layers.push_back(workloads_node[i]["exclude"].as<std::vector<std::string>>(std::vector<std::string>()));
                } else if (workloads_node[i]["benchmark"].IsDefined()) {
                    benchmarks.push_back(workloads_node[i]["benchmark"].as<std::string>());
                    benchmarks_skipped_layers.push_back(workloads_node[i]["exclude"].as<std::vector<std::string>>(std::vector<std::string>()));
                } else {
                    std::cout << "Error while parsing workloads, no onnx or benchmark specified." << std::endl;
                    exit(1);
                }
            }
        } else {
            std::string workloads_folder = workloads_node.as<std::string>();
            if ( fs::is_directory(workloads_folder) )
                for (const auto & p : fs::directory_iterator(workloads_folder))
                    if (fs::is_regular_file(p) && p.path().extension() == ".onnx")
                        onnx_files.push_back(p.path().string());
        }

        Graph graph;

        // Parsing onnx files.
        LoadOnnxFiles(graph, onnx_files, onnx_skipped_layers);

        // Append Benchmark Networks to graph.
        LoadBenchmarks(graph, benchmarks, benchmarks_skipped_layers);

        std::cout << "   " << graph.NumLayers() << " layers and " << graph.NumWorkloads() << " unique workloads found." << std::endl << std::endl;

        return graph;
    }
    
    // === Architecture Parsing === //

    std::vector<ArchTemplate>* ParseArchitectureTemplates(std::string config_file) {
        std::cout << "🏛️  Parsing architectural templates ..." << std::endl;

        auto templates = new std::vector<ArchTemplate>();

        YAML::Node config = YAML::LoadFile(config_file);
        auto architecture_node = config["architecture"];
        std::string accelergy_path = config["accelergy-script-path"].as<std::string>("");
        std::string accelergy_components_path = architecture_node["accelergy-components"].as<std::string>("");

        if (architecture_node["templates"].IsMap()) 
        {
            for(auto it = architecture_node["templates"].begin(); it != architecture_node["templates"].end(); ++it) 
            {
                std::string name = it->first.as<std::string>();
                YAML::Node node = it->second;
                templates->emplace_back(templates->size(), name, accelergy_components_path, node, accelergy_path);
            }
        }
        
        if (templates->empty()) {
            std::cout << "   " << "Error in configuration file." << std::endl;
            exit(1);
        }        

        std::cout << "   " << templates->size() << " templates found." << std::endl << std::endl;

        return templates;
    }

    // === Search Configuration Parsing === //

    Config ParseSearch(std::string config_file, std::string output_dir) {
        std::cout << "🧬 Parsing search configuration ..." << std::endl;

        Config config;
        YAML::Node config_yaml = YAML::LoadFile(config_file);
        auto config_node = config_yaml["search"];

        auto bts = [](bool b) { return b ? "Yes" : "No"; };

        // GLOBAL
        config.out_dir                          = output_dir;

        // MEDEA
        auto medea_node = config_node["layer-mapper"];

        config.medea.num_threads                = medea_node["num-threads"].as<unsigned>(config.medea.num_threads);
        config.medea.num_generations            = medea_node["num-generations"].as<unsigned>(config.medea.num_generations);
        config.medea.population_size            = medea_node["population-size"].as<unsigned>(config.medea.population_size);
        config.medea.immigrant_population_size  = medea_node["immigrant-population-size"].as<unsigned>(config.medea.immigrant_population_size);
        config.medea.use_tournament             = medea_node["use-tournament"].as<bool>(config.medea.use_tournament);
        config.medea.random_mutation_prob       = medea_node["random-mutation-prob"].as<double>(config.medea.random_mutation_prob);
        config.medea.parallel_mutation_prob     = medea_node["parallel-mutation-prob"].as<double>(config.medea.parallel_mutation_prob);
        config.medea.fill_mutation_prob         = medea_node["fill-mutation-prob"].as<double>(config.medea.fill_mutation_prob);
        config.medea.crossover_prob             = medea_node["crossover-prob"].as<double>(config.medea.crossover_prob);
        config.medea.update_ert                 = medea_node["update-ert"].as<bool>(config.medea.update_ert);
        config.medea.buffer_update_granularity  = medea_node["buffer-update-granularity"].as<unsigned>(config.medea.buffer_update_granularity);
        config.medea.random_when_illegal        = medea_node["random-when-illegal"].as<bool>(config.medea.random_when_illegal);

        std::cout << "   Medea" << std::endl; 
        std::cout << std::left << std::fixed << std::setprecision(2) << "   " 
                  << std::setw(20) << "Num. Threads:" << std::setw(10) << (config.medea.num_threads ? config.medea.num_threads : omp_get_max_threads())
                  << std::setw(20) << "Num. Generations:" << std::setw(10) << config.medea.num_generations
                  << std::setw(20) << "Pop. Size:" << std::setw(10) << config.medea.population_size << std::endl;
        std::cout << std::left << std::fixed << std::setprecision(2) << "   " 
                  << std::setw(20) << "Imm. Pop. Size:" << std::setw(10) << config.medea.immigrant_population_size
                  << std::setw(20) << "Tournament:" << std::setw(10) << config.medea.use_tournament
                  << std::setw(20) << "Update ERT:" << std::setw(10) << config.medea.update_ert << std::endl;
        std::cout << std::left << std::fixed << std::setprecision(2) << "   "
                  << std::setw(20) << "Random Mut. Prob.:" << std::setw(10) << config.medea.random_mutation_prob
                  << std::setw(20) << "Paral. Mut. Prob.:" << std::setw(10) << config.medea.parallel_mutation_prob
                  << std::setw(20) << "Fill Mut. Prob.:" << std::setw(10) << config.medea.fill_mutation_prob << std::endl;
        std::cout << std::left << std::fixed << std::setprecision(2) << "   " 
                  << std::setw(20) << "Crossover Prob.:" << std::setw(10) << config.medea.crossover_prob
                  << std::setw(20) << "Rand-When-Illegal.:" << std::setw(10) << config.medea.random_when_illegal << std::endl;

        // MOHAM
        auto moham_node = config_node["global-scheduler"];

        config.moham.population_size             = moham_node["population-size"].as<unsigned>(config.moham.population_size);
        config.moham.immigrant_population_size   = moham_node["immigrant-population-size"].as<unsigned>(config.moham.immigrant_population_size);
        config.moham.num_generations             = moham_node["num-generations"].as<unsigned>(config.moham.num_generations);
        config.moham.priority_crossover_prob     = moham_node["priority-crossover-prob"].as<double>(config.moham.priority_crossover_prob);
        config.moham.mapping_crossover_prob      = moham_node["mapping-crossover-prob"].as<double>(config.moham.mapping_crossover_prob);
        config.moham.subacc_crossover_prob       = moham_node["subacc-crossover-prob"].as<double>(config.moham.subacc_crossover_prob);
        config.moham.splitting_mutation_prob     = moham_node["splitting-mutation-prob"].as<double>(config.moham.splitting_mutation_prob);
        config.moham.merging_mutation_prob       = moham_node["merging-mutation-prob"].as<double>(config.moham.merging_mutation_prob);
        config.moham.priority_mutation_prob      = moham_node["priority-mutation-prob"].as<double>(config.moham.priority_mutation_prob);
        config.moham.mapping_mutation_prob       = moham_node["mapping-mutation-prob"].as<double>(config.moham.mapping_mutation_prob);
        config.moham.template_mutation_prob      = moham_node["template-mutation-prob"].as<double>(config.moham.template_mutation_prob);
        config.moham.assignment_mutation_prob    = moham_node["assignment-mutation-prob"].as<double>(config.moham.assignment_mutation_prob);
        config.moham.position_mutation_prob      = moham_node["position-mutation-prob"].as<double>(config.moham.position_mutation_prob);
        config.moham.max_per_workload_mappings   = moham_node["max-per-workload-mappings"].as<unsigned>(config.moham.max_per_workload_mappings);
        config.moham.xu_priority                 = moham_node["xu-priority"].as<bool>(config.moham.xu_priority);
        config.moham.use_tournament              = moham_node["use-tournament"].as<bool>(config.moham.use_tournament);
        config.moham.random_when_illegal         = moham_node["random-when-illegal"].as<bool>(config.moham.random_when_illegal);
        config.moham.stop_on_convergence         = moham_node["stop-on-convergence"].as<bool>(config.moham.stop_on_convergence);

        // Architecture Stuff

        config.moham.max_subaccelerators         = config_yaml["architecture"]["max-subaccelerators"].as<unsigned>(config.moham.max_subaccelerators);
        config.moham.system_bandwidth            = config_yaml["architecture"]["system-bandwidth"].as<double>(config.moham.system_bandwidth);
        config.moham.nip_link_bandwidth          = config_yaml["architecture"]["nip-link-bandwidth"].as<double>(config.moham.nip_link_bandwidth);
        config.moham.nip_hop_energy              = config_yaml["architecture"]["nip-hop-energy"].as<double>(config.moham.nip_link_bandwidth);
        
        const auto& minode = config_yaml["architecture"]["memory-interfaces"];
        for (const auto& am : minode["amount"]) {
            auto amv = am.as<std::vector<unsigned>>();
            assert(amv.size() == 2);
            config.moham.memory_interfaces_amount.emplace_back(amv[0], amv[1]);
        }
        config.moham.memory_interfaces_position  = minode["position"].as<std::string>(config.moham.memory_interfaces_position);
        config.moham.max_memory_interfaces_amount = minode["max-amount"].as<unsigned>(config.moham.max_memory_interfaces_amount);
        
        config.moham.explore_mapping             = moham_node["explore-mapping"].as<bool>(config.moham.explore_mapping);
        config.moham.negotiate_arch              = moham_node["negotiate-arch"].as<bool>(config.moham.negotiate_arch);

        config.moham.random_search               = moham_node["random-search"].as<bool>(config.moham.random_search);

        config.moham.single_obj                  = moham_node["single-objective"].as<bool>(config.moham.single_obj);
        config.moham.prod_obj                    = moham_node["product-objective"].as<bool>(config.moham.prod_obj);
        config.moham.weights_obj                 = moham_node["weights-objective"].as<std::vector<double>>(std::vector<double>());

        // Parsing Summary

        std::cout << "   Moham" << std::endl; 
        std::cout << std::left << std::fixed << std::setprecision(2) << "   " 
                  << std::setw(20) << "Num. Threads:" << std::setw(10) << (config.moham.num_threads ? config.moham.num_threads : omp_get_max_threads())
                  << std::setw(20) << "Num. Generations:" << std::setw(10) << config.moham.num_generations
                  << std::setw(20) << "Pop. Size:" << std::setw(10) << config.moham.population_size << std::endl;
        std::cout << std::left << std::fixed << std::setprecision(2) << "   " 
                  << std::setw(20) << "Imm. Pop. Size:" << std::setw(10) << config.moham.immigrant_population_size
                  << std::setw(20) << "Tournament:" << std::setw(10) << config.moham.use_tournament
                  << std::setw(20) << "Max Sub-acc:" << std::setw(10) << config.moham.max_subaccelerators << std::endl;
        std::cout << std::left << std::fixed << std::setprecision(2) << "   "
                  << std::setw(20) << "Prio. Mut. Prob.:" << std::setw(10) << config.moham.priority_mutation_prob
                  << std::setw(20) << "...:" << std::setw(10) << 0
                  << std::setw(20) << "...:" << std::setw(10) << 0 << std::endl;
        std::cout << std::left << std::fixed << std::setprecision(2) << "   " 
                  << std::setw(20) << "Prio. Cross. Prob.:" << std::setw(10) << config.moham.priority_crossover_prob
                  << std::setw(20) << "Explore Mapping:" << std::setw(10) << bts(config.moham.explore_mapping)
                  << std::setw(20) << "Negotiate Arch.:" << std::setw(10) << bts(config.moham.negotiate_arch) << std::endl;
        
        std::cout << std::endl;

        return config;
    }
}}

#endif // MOHAM_PARSING_H_