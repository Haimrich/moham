#include "accelergy.h"

#ifndef BUILD_BASE_DIR
#define BUILD_BASE_DIR "."
#endif

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "yaml-cpp/yaml.h"

#include "common.h"
#include "timeloop.h"

namespace moham
{
    Accelergy::Accelergy(timeloop::CompoundConfig *config, std::string fast_accelergy_path = "")
        : config_(config)
    {
        if (fast_accelergy_path.empty())
        {
            fast_accelergy_path_ = BUILD_BASE_DIR;
            fast_accelergy_path_ += "/../../scripts/fast_accelergy.py";
        }
        else
        {
            fast_accelergy_path_ = fast_accelergy_path;
        }
    }

    Accelergy::RT Accelergy::GetReferenceTables(std::unordered_map<std::string, uint64_t> &updates, std::string out_prefix)
    {
        YAML::Node ert_node = YAML::Node();
        YAML::Node art_node = YAML::Node();

        if (!FindInCache(updates, ert_node, art_node))
        {
            std::vector<std::string> input_files = config_->inFiles;

            std::string cmd = "python3 " + fast_accelergy_path_;

            for (auto input_file : input_files)
                cmd += " " + input_file;

            cmd += " --oprefix " + out_prefix + ". --updates ";

            for (std::unordered_map<std::string, uint64_t>::iterator it = updates.begin(); it != updates.end(); it++)
                cmd += " " + it->first + "," + std::to_string(it->second);

            //std::cout << "CMD: " << cmd.c_str() << std::endl;
            std::string fast_accelergy_out = Run(cmd.c_str());
            if (fast_accelergy_out.length() == 0)
            {
                std::cout << "Failed to run Accelergy. Did you install Accelergy or specify ACCELERGYPATH correctly? Or check accelergy.log to see what went wrong" << std::endl;
                exit(0);
            }

            YAML::Node acc_out = YAML::Load(fast_accelergy_out);
            ert_node = acc_out["ERT"];
            art_node = acc_out["ART"];

            std::call_once(invariant_nodes_inizialized_, [&]
                           { InizializeInvariantNodes(ert_node, art_node, updates); });

            UpdateCache(ert_node, art_node, updates);
        }

        auto art_config = timeloop::CompoundConfigNode(nullptr, art_node, config_);
        auto ert_config = timeloop::CompoundConfigNode(nullptr, ert_node, config_);

        return (RT){.energy = ert_config, .area = art_config};
    }

    std::string Accelergy::Run(const char *cmd)
    {
        std::string result = "";
        char buffer[128];
        FILE *pipe = popen(cmd, "r");
        if (!pipe)
        {
            std::cout << "popen(" << cmd << ") failed" << std::endl;
            exit(0);
        }

        try
        {
            while (fgets(buffer, 128, pipe) != nullptr)
                result += buffer;
        }
        catch (...)
        {
            pclose(pipe);
        }
        pclose(pipe);
        return result;
    }

    void Accelergy::InizializeInvariantNodes(YAML::Node &ert_node, YAML::Node &art_node, std::unordered_map<std::string, uint64_t> &updates)
    {
        YAML::Emitter ytmpe, ytmpa;
        ytmpe << YAML::BeginSeq;
        ytmpa << YAML::BeginSeq;

        for (std::size_t i = 0; i < art_node["tables"].size(); i++)
        {
            auto name = art_node["tables"][i]["name"].as<std::string>();
            name = name.substr(name.find_last_of(".") + 1);

            if (updates.find(name) == updates.end())
            {
                ytmpe << ert_node["tables"][i];
                ytmpa << art_node["tables"][i];
            }
        }

        ytmpe << YAML::EndSeq;
        ytmpa << YAML::EndSeq;

        invariant_nodes_.energy = ytmpe.c_str();
        invariant_nodes_.area = ytmpa.c_str();
    }

    bool Accelergy::FindInCache(std::unordered_map<std::string, uint64_t> &updates, YAML::Node &ert_node, YAML::Node &art_node)
    {
        std::string art, ert;
        art = ert = "version: 0.3\ntables:\n";

        bool cache_hit = true;

        #pragma omp critical(accelergy_cache)
        for (auto it = updates.begin(); cache_hit && it != updates.end(); it++)
        {
            auto buffer_cache = cache_.find(it->first);
            cache_hit = (buffer_cache != cache_.end());
            if (cache_hit)
            {
                auto size_cache = buffer_cache->second.find(it->second);
                cache_hit = (size_cache != buffer_cache->second.end());
                if (cache_hit)
                {
                    Entry &entry = size_cache->second;
                    art += entry.area + "\n";
                    ert += entry.energy + "\n";
                }
            }
        }

        if (!cache_hit) return false;

        art_node = YAML::Load(art + invariant_nodes_.area);
        ert_node = YAML::Load(ert + invariant_nodes_.energy);

        //std::cout << "   Accelergy Cache Hit! " << std::endl;
        return true;
    }

    void Accelergy::UpdateCache(YAML::Node &ert_node, YAML::Node &art_node, std::unordered_map<std::string, uint64_t> &updates)
    {
        //std::cout << art_node["tables"] << std::endl;

        #pragma omp critical(accelergy_cache)
        for (std::size_t i = 0; i < art_node["tables"].size(); i++)
        {
            auto name = art_node["tables"][i]["name"].as<std::string>();

            std::size_t ie = name.find_last_of("]");
            std::size_t is = name.find_last_of(".") + 1;
            if (is > ie) {
                name = name.substr(is);
            } else {
                ie = name.find_last_of("[");
                is = name.find_last_of(".",ie) + 1;
                name =  name.substr(is, ie-is);
            }

            auto size = updates.find(name);
            if (size != updates.end())
            {
                YAML::Emitter ytmpe, ytmpa;

                ytmpe << YAML::BeginSeq << ert_node["tables"][i] << YAML::EndSeq;
                ytmpa << YAML::BeginSeq << art_node["tables"][i] << YAML::EndSeq;

                cache_[name][size->second] = (Entry){
                    .energy = ytmpe.c_str(),
                    .area = ytmpa.c_str()};
            }

        }
    }

    Accelergy::Accelergy(const Accelergy& other)
    {
        config_ = other.config_;
        fast_accelergy_path_ = other.fast_accelergy_path_;
        cache_ = other.cache_;
        invariant_nodes_ = other.invariant_nodes_;
    }

    Accelergy::Accelergy(const Accelergy& other, timeloop::CompoundConfig* new_config)
    {
        config_ = new_config;
        fast_accelergy_path_ = other.fast_accelergy_path_;
        cache_ = other.cache_;
        invariant_nodes_ = other.invariant_nodes_;
    }

    Accelergy& Accelergy::operator=(const Accelergy& other)
    {
        if (this == &other) return *this;
 
        config_ = other.config_;
        fast_accelergy_path_ = other.fast_accelergy_path_;
        cache_ = other.cache_;
        invariant_nodes_ = other.invariant_nodes_;

        return *this;
    }

}