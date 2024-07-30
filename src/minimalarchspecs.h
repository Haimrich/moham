#ifndef MOHAM_MINIMALARCH_H_
#define MOHAM_MINIMALARCH_H_

#include "yaml-cpp/yaml.h"

namespace moham
{
  class MinimalArchSpecs
  {
    struct Level
    {
      std::string name;
      int mesh_x, mesh_y;
      int size;
    };

    std::vector<Level> levels;

  public:

    void Reset(size_t size)
    {
      levels.clear();
      levels.reserve(size);
    }

    void Add(std::string name, int mesh_x, int mesh_y, int size = 0)
    {
      levels.push_back((Level){name, mesh_x, mesh_y, size});
    }

    Level GetLevel(size_t i) const {
      return levels[i];
    }

    bool operator!=(const MinimalArchSpecs& other) 
    {
      if (levels.size() != other.levels.size())
        return true;

      for (size_t i = 0; i < levels.size(); i++) 
        if (levels[i].mesh_x != other.levels[i].mesh_x || levels[i].mesh_y != other.levels[i].mesh_y || levels[i].size != other.levels[i].size) 
          return true;

      return false;
    }

    bool operator==(const MinimalArchSpecs& other) 
    {
      return !(*this != other);
    }


    MinimalArchSpecs& operator&=(const MinimalArchSpecs &other)
    {
      auto bound = levels.size();

      // Do this better
      auto level_tmp = levels;

      levels[bound - 1].mesh_x = std::max(level_tmp[bound - 1].mesh_x, other.levels[bound - 1].mesh_x);
      levels[bound - 1].mesh_y = std::max(level_tmp[bound - 1].mesh_y, other.levels[bound - 1].mesh_y);

      for (int i = bound - 2; i >= 0; i--)
      {
        levels[i].mesh_x = levels[i + 1].mesh_x * std::max(level_tmp[i].mesh_x / level_tmp[i + 1].mesh_x, other.levels[i].mesh_x / other.levels[i + 1].mesh_x);
        levels[i].mesh_y = levels[i + 1].mesh_y * std::max(level_tmp[i].mesh_y / level_tmp[i + 1].mesh_y, other.levels[i].mesh_y / other.levels[i + 1].mesh_y);
      }

      for (size_t i = 0; i < bound; i++)
      {
        levels[i].name = other.levels[i].name;
        levels[i].size = std::max(level_tmp[i].size, other.levels[i].size);
      }

      return *this;
    }

    friend YAML::Emitter &operator<<(YAML::Emitter &out, const MinimalArchSpecs &arch)
    {
      if (arch.levels.empty())
        return out;

      out << YAML::BeginSeq;

      out << YAML::BeginMap;
      out << YAML::Key << "name" << YAML::Value << arch.levels[0].name;
      out << YAML::Key << "mesh_x" << YAML::Value << arch.levels[0].mesh_x;
      out << YAML::Key << "mesh_y" << YAML::Value << arch.levels[0].mesh_y;
      out << YAML::EndMap;

      for (size_t i = 1; i < arch.levels.size(); i++)
      {
        out << YAML::BeginMap;
        out << YAML::Key << "name" << YAML::Value << arch.levels[i].name;
        out << YAML::Key << "mesh_x" << YAML::Value << arch.levels[i].mesh_x;
        out << YAML::Key << "mesh_y" << YAML::Value << arch.levels[i].mesh_y;
        out << YAML::Key << "size" << YAML::Value << arch.levels[i].size;
        out << YAML::EndMap;
      }
      out << YAML::EndSeq;

      return out;
    }


    MinimalArchSpecs(const YAML::Node &yaml) {
      Add(yaml[0]["name"].as<std::string>(), yaml[0]["mesh_x"].as<int>(), yaml[0]["mesh_y"].as<int>());

      for (size_t i=1; i < yaml.size(); i++) 
        Add(yaml[i]["name"].as<std::string>(), yaml[i]["mesh_x"].as<int>(), yaml[i]["mesh_y"].as<int>(), yaml[i]["size"].as<int>());
    }


    MinimalArchSpecs& operator=(const MinimalArchSpecs &other)
    {
      levels = other.levels;
      return *this;
    }

    MinimalArchSpecs(const MinimalArchSpecs &other) : levels(other.levels) {}

    MinimalArchSpecs() = default;

  };

}

#endif // MOHAM_MINIMALARCH_H_
