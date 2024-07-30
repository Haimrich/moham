#include "graph.h"

#include <deque>
#include <unordered_map>
#include <ostream>
#include <string>

#include "common.h"
#include "layer.h"
#include "tensor.h"
#include "workload.h"

namespace moham {

    Layer* Graph::AddNode(std::string name, std::string network, LayerType type, std::unordered_map<std::string, Tensor> attributes, std::vector<Tensor> parameters, std::vector<Edge> incoming, std::vector<Tensor> input_tensors, Tensor output_tensor) {
        std::size_t workload_id = GetLayerWorkloadId(type, attributes, parameters, input_tensors, output_tensor);

        Layer node(name, network, layers_.size(), workload_id, type, attributes, parameters, incoming);
        layers_.push_back(node);

        Layer* layer_ptr = &layers_.back();

        for (auto& dep : node.incoming)
            dep.layer->outgoing.push_back({layer_ptr, dep.data});
        
        return layer_ptr;
    }

    Layer* Graph::AddNode(std::string name, std::string network, LayerType type, 
                   std::vector<Edge> incoming, 
                   unsigned c, unsigned m,
                   unsigned p, unsigned q,
                   unsigned r, unsigned s,
                   unsigned ws, unsigned hs,
                   unsigned wd, unsigned hd) 
        {
        
        std::unordered_map<std::string, std::size_t> dimensions = {
            {"C",c}, {"M",m}, {"P",p}, {"Q",q}, {"R",r}, {"S",s},
            {"WS",ws}, {"HS",hs}, {"WD",wd}, {"HD",hd}, {"N",1}
        };
    
        std::size_t workload_id = GetLayerWorkloadId(type, dimensions);

        // TODO
        std::unordered_map<std::string, Tensor> attributes;
        std::vector<Tensor> parameters;

        Layer node(name, network, layers_.size(), workload_id, type, attributes, parameters, incoming);
        layers_.push_back(node);

        Layer* layer_ptr = &layers_.back();

        for (auto& dep : node.incoming)
            dep.layer->outgoing.push_back({layer_ptr, dep.data});
        
        return layer_ptr;
    }

    std::size_t Graph::NumLayers() const {
        return layers_.size();
    }

    std::size_t Graph::NumWorkloads() const {
        return workloads_.size();
    }

    std::size_t Graph::GetLayerWorkloadId(LayerType type, std::unordered_map<std::string, std::size_t> dimensions) {
        Workload workload(workloads_.size(), type, dimensions);
        
        for (std::size_t i = 0; i < workloads_.size(); i++)
            if (workload == workloads_[i])
                return i;

        workloads_.push_back(workload);
        return workloads_.size() - 1;
    }

    std::size_t Graph::GetLayerWorkloadId(LayerType type, std::unordered_map<std::string,Tensor>& attributes, std::vector<Tensor>& parameters, std::vector<Tensor> input_tensors, Tensor output_tensor) {
        std::unordered_map<std::string, std::size_t> dimensions = {
            {"C",1}, {"M",1}, {"P",1}, {"Q",1}, {"R",1}, {"S",1}, 
            {"WS",1}, {"HS",1}, {"WD",1}, {"HD",1}, {"N",1}
        };

        if (type == LayerType::CONV)
        {
            dimensions["C"] = parameters[0].Shape()[1];
            dimensions["M"] = parameters[0].Shape()[0];
            dimensions["R"] = parameters[0].Shape()[2];
            dimensions["S"] = parameters[0].Shape()[3];
            dimensions["P"] = output_tensor.Shape()[2];
            dimensions["Q"] = output_tensor.Shape()[3];
            dimensions["WD"] = attributes["dilations"].Shape()[0];
            dimensions["HD"] = attributes["dilations"].Shape()[1];
            dimensions["WS"] = attributes["strides"].Shape()[0];
            dimensions["HS"] = attributes["strides"].Shape()[1];
        }
        else if (type == LayerType::TRANSPOSED_CONV)
        {
            dimensions["C"] = parameters[0].Shape()[1];
            dimensions["M"] = parameters[0].Shape()[0];
            dimensions["R"] = parameters[0].Shape()[2];
            dimensions["S"] = parameters[0].Shape()[3];
            dimensions["P"] = input_tensors[0].Shape()[2];
            dimensions["Q"] = input_tensors[0].Shape()[3];
            dimensions["WD"] = attributes["dilations"].Shape()[0];
            dimensions["HD"] = attributes["dilations"].Shape()[1];
            dimensions["WS"] = attributes["strides"].Shape()[0];
            dimensions["HS"] = attributes["strides"].Shape()[1];
        }
        else if (type == LayerType::DENSE)
        {
            dimensions["C"] = parameters[0].Shape()[1];
            dimensions["M"] = parameters[0].Shape()[0];
        }
        else if (type == LayerType::MATMUL)
        {
            assert(input_tensors.size() == 2);
            auto ia = input_tensors[0].Shape();
            auto ib = input_tensors[1].Shape();
            auto o = output_tensor.Shape();
            dimensions["N"] = o[0] * o[1];
            if (o.size() == 4) dimensions["N"] *= o[2];
            assert(ia[ia.size()-1] == ib[ib.size()-2]);
            dimensions["C"] = ia[ia.size()-1];
            dimensions["M"] = o[o.size() - 1];
        }
        else 
        {
            std::cout << "Unsupported layer " << (int)type << " while generating workload." << std::endl;
            exit(1);
        }
        
        return GetLayerWorkloadId(type, dimensions);
    }


    std::vector<Workload>& Graph::GetWorkloads() {
        return workloads_;
    }

    const Layer& Graph::operator[](std::size_t l) const {
        return layers_[l];
    }

    std::ostream& operator<<(std::ostream& os, const Graph& g)
    {
        os << std::endl << "|==========< LAYER-WORKLOAD LOOKUP >==========|" << std::endl;
        for (auto& l : g.layers_) {
            os << l.network << " " << l.name << " " << l.id << " " << l.workload_id << std::endl;
        }

        os << std::endl << "|==================< GRAPH >==================|" << std::endl;

        for (auto& l : g.layers_) {
            os << l.id << " " << l.name << " " << l.network << std::endl;

            os << "  Attributes: " << std::endl;
            for (auto p : l.attributes) {
                os << "    " << p.first << ": ";
                for (auto i : p.second.Shape())
                    os << i << " ";
                os << std::endl;
            }

            os << "  Parameters: " << std::endl;
            for (auto p : l.parameter_tensors)
                os << "    " << p.Name() << std::endl;

            os << "  Input: " << std::endl;
            for (auto p : l.incoming)
                os << "    " << p.data.Name() << " " << p.layer->name << " " << p.layer->id << std::endl;

            os << "  Output: " << std::endl;
            for (auto p : l.outgoing)
                os << "    " << p.data.Name() << " " << p.layer->name << " " << p.layer->id << std::endl;
        }

        os << "[";
        for (auto& l : g.layers_) {
            os << l.id << ",";
        }
        os << "]" << std::endl;

        os << "[";
        for (auto& l : g.layers_) 
            for (auto& e : l.outgoing)
                os << "(" << l.id << "," << e.layer->id << "),";
        os << "]" << std::endl;

        os << "[";
        for (auto& l : g.layers_) 
            for (auto& e : l.incoming)
                os << "(" << e.layer->id << "," << l.id<< "),";
        os << "]" << std::endl;


        os << std::endl << "|================< WORKLOADS >================|" << std::endl;
        for (Layer::ID l = 0; l < g.NumLayers(); l++)
            os << std::setw(3) << l << " : " << g.workloads_[g.layers_[l].workload_id].Name() << std::endl;
        
        return os;
    }

    
}