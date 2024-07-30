#ifndef MOHAM_WORKLOAD_H_
#define MOHAM_WORKLOAD_H_

#include <string>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "common.h"
#include "timeloop.h"


namespace moham
{

    class Workload
    {
    public:
        typedef std::size_t ID;
        ID id;
        LayerType type;
        std::unordered_map<std::string, size_t> dimensions;

        timeloop::Workload workload;
        timeloop::Shape shape;

    public:

        Workload() = default;

        Workload(ID id, LayerType type, std::unordered_map<std::string, std::size_t> dimensions) 
            : id(id), type(type), dimensions(dimensions) 
        {
            //std::cout << Yaml() << std::endl;

            timeloop::CompoundConfig tconfig(YAML::Dump(Yaml()), "yml");
            auto config = tconfig.getRoot().lookup("problem");
            timeloop::ParseWorkload(config, workload);

            shape = *timeloop::GetWorkloadShape();
        }

        bool operator==(Workload& other) const
        {
            return type == other.type && dimensions == other.dimensions;
        }

        YAML::Node Yaml()
        {
            auto yaml = YAML::Node();
            if (type == LayerType::CONV)
            {
                yaml = YAML::Load(conv_template);
                auto instance_node = yaml["problem"]["instance"];
                instance_node["C"] = dimensions["C"];
                instance_node["M"] = dimensions["M"];
                instance_node["R"] = dimensions["R"];
                instance_node["S"] = dimensions["S"];
                instance_node["P"] = dimensions["P"];
                instance_node["Q"] = dimensions["Q"];
                instance_node["Wdilation"] = dimensions["WD"];
                instance_node["Hdilation"] = dimensions["HD"];
                instance_node["Wstride"] = dimensions["WS"];
                instance_node["Hstride"] = dimensions["HS"];
                yaml["problem"]["instance"] = instance_node;
            }
            else if (type == LayerType::TRANSPOSED_CONV)
            {
                yaml = YAML::Load(transposed_conv_template);
                auto instance_node = yaml["problem"]["instance"];
                instance_node["C"] = dimensions["C"];
                instance_node["M"] = dimensions["M"];
                instance_node["R"] = dimensions["R"];
                instance_node["S"] = dimensions["S"];
                instance_node["P"] = dimensions["P"];
                instance_node["Q"] = dimensions["Q"];
                instance_node["Wdilation"] = dimensions["WD"];
                instance_node["Hdilation"] = dimensions["HD"];
                instance_node["Wstride"] = dimensions["WS"];
                instance_node["Hstride"] = dimensions["HS"];
                yaml["problem"]["instance"] = instance_node;
            }
            else if (type == LayerType::DENSE)
            {
                yaml = YAML::Load(conv_template);
                auto instance_node = yaml["problem"]["instance"];
                instance_node["C"] = dimensions["C"];
                instance_node["M"] = dimensions["M"];
                yaml["problem"]["instance"] = instance_node;
            }
            else if (type == LayerType::MATMUL)
            {
                yaml = YAML::Load(conv_template);
                auto instance_node = yaml["problem"]["instance"];
                instance_node["N"] = dimensions["N"];
                instance_node["C"] = dimensions["C"];
                instance_node["M"] = dimensions["M"];
                yaml["problem"]["instance"] = instance_node;
            }
            
            return yaml;
        }


        std::string Name() const
        {
            //std::string num = std::to_string(id);
            //std::string name = std::string(3-num.size(), '0') + num;
            std::string name = "WL";

            for (std::size_t i = 0; i < shape.NumFlattenedDimensions; i++)
                name += "_" + shape.FlattenedDimensionIDToName.at(i) + "." + std::to_string(workload.GetFlattenedBound(i));
            
            for (std::size_t i = 0; i < shape.NumCoefficients; i++)
                name += "_" + shape.CoefficientIDToName.at(i) + "." + std::to_string(workload.GetCoefficient(i));
            
            return name;
        }
    private:
        static constexpr const char *conv_template = R"(
        problem:
            shape:
                name: "Convolution"
                dimensions: [ C, M, R, S, N, P, Q ]
                coefficients:
                    -   name: Wstride
                        default: 1
                    -   name: Hstride
                        default: 1
                    -   name: Wdilation
                        default: 1
                    -   name: Hdilation
                        default: 1
                data-spaces:
                    -   name: Weights
                        projection:
                            - [ [C] ]
                            - [ [M] ]
                            - [ [R] ]
                            - [ [S] ]
                    -   name: Inputs
                        projection:
                            - [ [N] ]
                            - [ [C] ]
                            - [ [R, Wdilation], [P, Wstride] ]
                            - [ [S, Hdilation], [Q, Hstride] ]
                    -   name: Outputs
                        projection:
                            - [ [N] ]
                            - [ [M] ]
                            - [ [Q] ]
                            - [ [P] ]
                        read-write: True
            instance:
                N: 1
                C: 1
                M: 1
                R: 1
                S: 1
                P: 1
                Q: 1
                Wdilation: 1
                Hdilation: 1
                Wstride: 1
                Hstride: 1
        )";

        static constexpr const char *transposed_conv_template = R"(
        problem:
            shape:
                name: "Convolution"
                dimensions: [ C, M, R, S, N, P, Q ]
                coefficients:
                    -   name: Wstride
                        default: 1
                    -   name: Hstride
                        default: 1
                    -   name: Wdilation
                        default: 1
                    -   name: Hdilation
                        default: 1
                data-spaces:
                    -   name: Weights
                        projection:
                            - [ [C] ]
                            - [ [M] ]
                            - [ [R] ]
                            - [ [S] ]
                    -   name: Inputs
                        projection:
                            - [ [N] ]
                            - [ [C] ]
                            - [ [Q] ]
                            - [ [P] ]
                    -   name: Outputs
                        projection:
                            - [ [N] ]
                            - [ [M] ]
                            - [ [R, Wdilation], [P, Wstride] ]
                            - [ [S, Hdilation], [Q, Hstride] ]
                        read-write: True
            instance:
                N: 1
                C: 1
                M: 1
                R: 1
                S: 1
                P: 1
                Q: 1
                Wdilation: 1
                Hdilation: 1
                Wstride: 1
                Hstride: 1
        )";
    };

}
#endif // MOHAM_WORKLOAD_H_