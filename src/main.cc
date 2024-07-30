#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "graph.h"
#include "archtemplate.h"
#include "config.h"
#include "parsing.h"

#include "timeloop.h"

#include "medea.h"
#include "moham.h"

using namespace std;
using namespace moham;

namespace po = boost::program_options;
namespace fs = boost::filesystem;


int main(int argc, char **argv)
{
  cout << R"(
   __  __  ___ _____ _  _ 
  |  \/  |/ _ \_   _| || |
  | |\/| | (_) || | | __ |
  |_|  |_|\___/ |_| |_||_|                
  )" << endl;

  po::options_description desc(
    "🦋 MOHAM - Allowed options"
  );
  desc.add_options()
    ("help,h", "Print help information.")
    ("config-file,c", po::value<string>(), "Config .yaml filepath.")
    ("output,o", po::value<string>()->default_value("."), "Output directory.")
  ;

  po::positional_options_description p;
  p.add("config-file", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);
  
  if (vm.count("help"))
  {
    cout << desc << "\n";
    return 0;
  }

  if (!vm.count("config-file"))
  {
    cout << "Missing config file. \n";
    return 0;
  }

  string config_file = vm["config-file"].as<string>();

  Graph graph = parsing::ParseWorkloads(config_file); 
  std::vector<ArchTemplate>* arch_templates = parsing::ParseArchitectureTemplates(config_file);
  Config config = parsing::ParseSearch(config_file, vm["output"].as<string>());

  // Graph Debug
  std::ofstream graph_file(config.out_dir + "/graph.txt");
  graph_file << graph << std::endl;
  graph_file.close();

  // MEDEA
  std::vector<std::unordered_map<ArchTemplate::ID,Medea::Population>> workload_mappings;
  for (auto& workload : graph.GetWorkloads())
  {
    workload_mappings.emplace_back();
    for (auto& arch_template : *arch_templates) 
      if (arch_template.AllowsLayerType(workload.type)) 
      {
        std::cout << std::endl;
        Medea medea(config, workload, arch_template);
        std::string medea_out_path = config.out_dir + "/medea/" + workload.Name() + "/" + arch_template.name + "/";
        
        Medea::Population pareto;

        if ( fs::is_directory(medea_out_path + "pareto") ) {
          pareto = medea.Parse(medea_out_path + "pareto");
          if (pareto.size() > 0) {
            std::cout << "   Skipping mapping." << std::endl;
          } else {
            pareto = medea.Run(medea_out_path);
          }
        } else {
          fs::create_directories(medea_out_path + "pareto");
          pareto = medea.Run(medea_out_path);
        }

        if (config.moham.max_per_workload_mappings == 0 || pareto.size() <= config.moham.max_per_workload_mappings) {
          workload_mappings.back()[arch_template.id] = pareto;
        } else {
          std::vector<Medea::Individual::ID> pareto_energy_sorted_idxs(pareto.size());
          std::iota(pareto_energy_sorted_idxs.begin(), pareto_energy_sorted_idxs.end(), 0);
          std::stable_sort(pareto_energy_sorted_idxs.begin(), pareto_energy_sorted_idxs.end(),
            [&pareto](Medea::Individual::ID i1, Medea::Individual::ID i2) {return pareto[i1].objectives[0] < pareto[i2].objectives[0];});

          std::vector<Medea::Individual::ID> pareto_latency_sorted_idxs(pareto.size());  
          std::iota(pareto_latency_sorted_idxs.begin(), pareto_latency_sorted_idxs.end(), 0);
          std::stable_sort(pareto_latency_sorted_idxs.begin(), pareto_latency_sorted_idxs.end(),
            [&pareto](Medea::Individual::ID i1, Medea::Individual::ID i2) {return pareto[i1].objectives[1] < pareto[i2].objectives[1];});

          std::unordered_set<Medea::Individual::ID> chosen_ids;
          for (Medea::Individual::ID i = 0; i < config.moham.max_per_workload_mappings / 2; i++)
          {
            chosen_ids.insert(pareto_energy_sorted_idxs[i]);
            chosen_ids.insert(pareto_latency_sorted_idxs[i]);
          }
          if (config.moham.max_per_workload_mappings % 2)
            chosen_ids.insert(pareto_energy_sorted_idxs[config.moham.max_per_workload_mappings / 2]);

          workload_mappings.back()[arch_template.id] = Medea::Population();

          if (config.moham.explore_mapping) {
            workload_mappings.back()[arch_template.id].reserve(chosen_ids.size());
            for (Medea::Individual::ID id : chosen_ids)
              workload_mappings.back()[arch_template.id].push_back(pareto[id]);
          } else { 
            int random = rand() % chosen_ids.size();
            workload_mappings.back()[arch_template.id].push_back(pareto[random]);
          }

          //std::cout << "Mappings: " << chosen_ids.size() << std::endl;
        }
        
      } else {
        workload_mappings.back()[arch_template.id] = Medea::Population();
      }
  }

  // MOHAM
  std::cout << std::endl;
  Moham moham(config, &graph, arch_templates, workload_mappings);

  if (!config.moham.random_search) {
    moham.Run();
  } else {
    moham.RunRandom();
  }

  std::cout << std::endl;
  return 0;
}

bool gTerminate = false;
bool gTerminateEval = false;