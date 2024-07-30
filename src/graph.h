#ifndef MOHAM_GRAPH_H_
#define MOHAM_GRAPH_H_

#include <deque>
#include <ostream>
#include <vector>
#include <unordered_map>

#include "layer.h"
#include "workload.h"

namespace moham {

    class Graph
    {
    private:
        std::deque<Layer> layers_;

        std::vector<Workload> workloads_;

    public:

        Layer* AddNode(std::string name, std::string network, LayerType type, std::unordered_map<std::string, Tensor> attributes, std::vector<Tensor> parameters, std::vector<Edge> incoming, std::vector<Tensor> input_tensors, Tensor output_tensor);
        
        Layer* AddNode(std::string name, std::string network, LayerType type, 
                       std::vector<Edge> incoming, 
                       unsigned c, unsigned m,
                       unsigned p, unsigned q,
                       unsigned r, unsigned s,
                       unsigned ws = 1, unsigned hs = 1,
                       unsigned wd = 1, unsigned hd = 1);

        std::size_t NumLayers() const;
        std::size_t NumWorkloads() const;

        std::vector<Workload>& GetWorkloads();

        const Layer& operator[](std::size_t l) const;

        friend std::ostream& operator<<(std::ostream& os, const Graph& g);

    private:

       std::size_t GetLayerWorkloadId(LayerType type, std::unordered_map<std::string,Tensor>& attributes, std::vector<Tensor>& parameters, std::vector<Tensor> input_tensors, Tensor output_tensor);
       std::size_t GetLayerWorkloadId(LayerType type, std::unordered_map<std::string,std::size_t> dimensions);
    };

}


#endif // MOHAM_GRAPH_H_