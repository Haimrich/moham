#ifndef MOHAM_LAYER_H_
#define MOHAM_LAYER_H_

#include <string>
#include <unordered_map>

#include "common.h"
#include "tensor.h"
#include "workload.h"

namespace moham {

    class Layer;
    
    struct Edge {
        Layer * layer;
        Tensor data;

        bool operator==(const Edge& other) const {
            return layer == other.layer;
        }

        bool operator<(const Edge& other) const {
            return layer < other.layer;
        }
    };

    class Layer
    {
    public:
        std::string name;
        std::string network;

        typedef std::size_t ID;
        ID id;
        Workload::ID workload_id;
        LayerType type;

        std::unordered_map<std::string, Tensor> attributes;
        std::vector<Tensor> parameter_tensors;
        std::vector<Edge> incoming;
        std::vector<Edge> outgoing;
        
    public:

        Layer(
            std::string name,
            std::string network,
            ID id, 
            Workload::ID workload_id, 
            LayerType type, 
            std::unordered_map<std::string, Tensor> attributes,
            std::vector<Tensor> parameter_tensors, 
            std::vector<Edge> incoming
            ) :
            name(name),
            network(network),
            id(id), 
            workload_id(workload_id), 
            type(type), 
            attributes(attributes), 
            parameter_tensors(parameter_tensors), 
            incoming(incoming), 
            outgoing() {}
    
    
        static inline LayerType StringToLayerType(std::string layer_type) {
            std::unordered_map<std::string, LayerType> lookup = {
                {"conv", LayerType::CONV},
                {"depthwise-conv", LayerType::DEPTHWISE_CONV},
                {"separable-conv", LayerType::SEPARABLE_CONV},
                {"transposed-conv", LayerType::TRANSPOSED_CONV},
                {"dense", LayerType::DENSE},
                {"matmul", LayerType::MATMUL}
            };
            return lookup[layer_type];
        }
    };
    
    

}

#endif // MOHAM_LAYER_H_