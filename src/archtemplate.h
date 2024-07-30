#ifndef MOHAM_ARCHTEMPLATE_H_
#define MOHAM_ARCHTEMPLATE_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "boost/filesystem.hpp"
#include "yaml-cpp/yaml.h"

#include "timeloop.h"
#include "accelergy.h"
#include "layer.h"

namespace fs = boost::filesystem;

namespace moham {

    class ArchTemplate
    {
    public:
        typedef std::size_t ID;
        ID id;

        std::string name;
        
        timeloop::CompoundConfig* config;
        timeloop::EngineSpecs arch_specs;
        timeloop::ArchProperties arch_props;
        timeloop::SparseOptInfo sparse_optimizations;
        bool is_sparse;

        std::vector<LayerType> allowed_layers;
        std::unordered_map<LayerType, timeloop::CompoundConfigNode> mapping_constraints_config; 

        Accelergy accelergy;

    public:
        ArchTemplate() = default;

        ArchTemplate(ID id, std::string name, std::string components_folder, YAML::Node ynode, std::string acc_path = "") 
            : id(id), name(name) 
        {
            // Get architecture template .yaml config files folder
            std::string input_files_folder = ynode["input"].as<std::string>();
            if (input_files_folder.empty()) {
                std::cout << "   " << "Missing \"" << name << "\" architecture config files folder." << std::endl;
                exit(1);
            }
            if (!fs::is_directory(input_files_folder)) {
                std::cerr << "   No such directory: '" << input_files_folder << "'" << std::endl;
                exit(1);
            }

            // Search .yaml files recursively
            std::vector<std::string> input_files;

            fs::path input_files_folder_path = input_files_folder;
            fs::recursive_directory_iterator it(input_files_folder_path), end;
            for (auto& entry : boost::make_iterator_range(it, end))
                if (fs::is_regular(entry) && entry.path().extension() == ".yaml")
                    input_files.push_back(entry.path().native());

            fs::path components_path = components_folder;
            fs::recursive_directory_iterator itc(components_path), endc;
            for (auto& entry : boost::make_iterator_range(itc, endc))
                if (fs::is_regular(entry) && entry.path().extension() == ".yaml")
                    input_files.push_back(entry.path().native());

            // Create Timeloop compound config from all the files
            config = new timeloop::CompoundConfig(input_files);
            auto root_node = config->getRoot();

            // Architecture
            timeloop::CompoundConfigNode arch = root_node.lookup("architecture");
            is_sparse = root_node.exists("sparse_optimizations");
            arch_specs = timeloop::Engine::ParseSpecs(arch, is_sparse);

            // Sparse optimizations
            timeloop::CompoundConfigNode sparse_optimizations_node;
            if (is_sparse) sparse_optimizations_node = root_node.lookup("sparse_optimizations");
            sparse_optimizations = timeloop::SparseOptInfo(timeloop::Sparse::ParseAndConstruct(sparse_optimizations_node, arch_specs));

            // Arch Properties
            arch_props = timeloop::ArchProperties(arch_specs);
            
            // Constraints for different layer types, these must also include arch constraints
            if (root_node.exists("constraints")) 
            {
                auto constraints_config = root_node.lookup("constraints");
                for (int i = 0; i < constraints_config.getLength(); i++)
                {
                    auto constraint_entry = constraints_config[i];
                    std::vector<std::string> target_layers;
                    constraint_entry.lookupArrayValue("layer-types", target_layers);
                    auto config = constraint_entry.lookup("targets");

                    for (auto layer_type : target_layers)
                        mapping_constraints_config[Layer::StringToLayerType(layer_type)] = config;
                }
            }  

            // Accelergy
            accelergy = Accelergy(config, acc_path);

            // Allowed Layers
            auto allowed_layer_strings = ynode["allowed-layers"].as<std::vector<std::string>>();  
            for (auto& layer_type_str : allowed_layer_strings)
                allowed_layers.push_back(Layer::StringToLayerType(layer_type_str));  
        }

        // Copy Constructor
        ArchTemplate(const ArchTemplate& other) :
            id(other.id),
            name(other.name),
            arch_specs(other.arch_specs),
            arch_props(other.arch_props),
            sparse_optimizations(other.sparse_optimizations),
            is_sparse(other.is_sparse),
            allowed_layers(other.allowed_layers)
        {
            config = new timeloop::CompoundConfig(other.config->inFiles);

            mapping_constraints_config.clear();
            for (auto& c: other.mapping_constraints_config) {
                auto cnode = c.second;
                mapping_constraints_config[c.first] = timeloop::CompoundConfigNode(nullptr, cnode.getYNode(), config);
            }

            accelergy = Accelergy(other.accelergy, config);
        }

        ~ArchTemplate() {
            if (config) 
                delete config;
        }


        bool AllowsLayerType(LayerType type) 
        {
            return std::find(allowed_layers.begin(), allowed_layers.end(), type) != allowed_layers.end();
        }
        
    };

}


#endif // MOHAM_ARCHTEMPLATE_H_