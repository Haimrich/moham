#ifndef MOHAM_ACCELERGY_H_
#define MOHAM_ACCELERGY_H_

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "yaml-cpp/yaml.h"

#include "timeloop.h"

namespace moham
{
  class Accelergy
  {
  protected:

    timeloop::CompoundConfig* config_;
    std::string fast_accelergy_path_;

    struct Entry
    {
      std::string energy;
      std::string area;
    };

    std::unordered_map<std::string, std::unordered_map<size_t, Entry>> cache_;

    Entry invariant_nodes_;
    std::once_flag invariant_nodes_inizialized_;

  public:

    struct RT
    {
      timeloop::CompoundConfigNode energy;
      timeloop::CompoundConfigNode area;
    };

    Accelergy() = default;

    Accelergy(timeloop::CompoundConfig* config, std::string fast_accelergy_path);

    Accelergy(const Accelergy& other);

    Accelergy(const Accelergy& other, timeloop::CompoundConfig* new_config);
 
    Accelergy& operator=(const Accelergy& other);

    RT GetReferenceTables(std::unordered_map<std::string, uint64_t> &updates, std::string out_prefix);

  protected:

    std::string Run(const char *cmd);

    void InizializeInvariantNodes(YAML::Node &ert_node, YAML::Node &art_node, std::unordered_map<std::string, uint64_t> &updates);

    bool FindInCache(std::unordered_map<std::string, uint64_t> &updates, YAML::Node &ert_node, YAML::Node &art_node);

    void UpdateCache(YAML::Node &ert_node, YAML::Node &art_node, std::unordered_map<std::string, uint64_t> &updates);
  
  };

}

#endif // MOHAM_ACCELERGY_H_