/*
    Benchmark network models, porting of
    https://github.com/he-actlab/planaria.code/blob/main/src/benchmarks/benchmarks.py
*/

#ifndef MOHAM_BENCHMARKS_H_
#define MOHAM_BENCHMARKS_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "graph.h"
#include "layer.h"

namespace moham 
{
    namespace benchmarks 
    {
        unsigned GoogleNet(Graph& graph, std::vector<std::string> skipped_layers) {
            const std::string nn = "GoogleNet";
            std::vector<Edge> prev;

            prev = {{graph.AddNode("conv1", nn, LayerType::CONV, prev, 3, 64, 112, 112, 7, 7, 2, 2), Tensor()}};
            prev = {{graph.AddNode("pool1", nn, LayerType::CONV, prev, 64, 64, 56, 56, 3, 3, 2, 2), Tensor()}};
            prev = {{graph.AddNode("conv2_3x3_reduce", nn, LayerType::CONV, prev, 64, 64, 56, 56, 1, 1), Tensor()}};
            prev = {{graph.AddNode("pooconv2_3x3l1", nn, LayerType::CONV, prev, 64, 192, 56, 56, 3, 3), Tensor()}};
            prev = {{graph.AddNode("conv2_3x3", nn, LayerType::CONV, prev, 64, 192, 56, 56, 3, 3), Tensor()}};
            prev = {{graph.AddNode("pool2", nn, LayerType::CONV, prev, 192, 192, 28, 28, 3, 3, 2, 2), Tensor()}};

            auto AddInception = [&](std::string incp_id, unsigned sfmap, unsigned nfmaps_in, unsigned nfmaps_1, unsigned nfmaps_3r,
                                    unsigned nfmaps_3, unsigned nfmaps_5r, unsigned nfmaps_5, unsigned nfmaps_pool, std::vector<Edge> prev) 
            { 
                // Add an inception module to the network.
                std::string pfx = "inception_" + incp_id +  "_";
                std::vector<Edge> nprev;
                std::vector<Edge> tprev;

                // 1x1 branch.
                nprev.push_back({graph.AddNode(pfx + "1x1", nn, LayerType::CONV, prev, nfmaps_in, nfmaps_1, sfmap, sfmap, 1, 1), Tensor()});
                
                // 3x3 branch.
                tprev = {{graph.AddNode(pfx + "3x3_reduce", nn, LayerType::CONV, prev, nfmaps_in, nfmaps_3r, sfmap, sfmap, 1, 1), Tensor()}};
                nprev.push_back({graph.AddNode(pfx + "3x3", nn, LayerType::CONV, tprev, nfmaps_3r, nfmaps_3, sfmap, sfmap, 3, 3), Tensor()});

                // 5x5 branch.
                tprev = {{graph.AddNode(pfx + "5x5_reduce", nn, LayerType::CONV, prev, nfmaps_in, nfmaps_5r, sfmap, sfmap, 1, 1), Tensor()}};
                nprev.push_back({graph.AddNode(pfx + "5x5", nn, LayerType::CONV, tprev, nfmaps_5r, nfmaps_5, sfmap, sfmap, 5, 5), Tensor()});

                // Pooling branch.
                nprev.push_back({graph.AddNode(pfx + "pool_proj", nn, LayerType::CONV, prev, nfmaps_in, nfmaps_pool, sfmap, sfmap, 1, 1), Tensor()});

                // Merge branches.
                return nprev;
            };

            // Inception 3
            prev = AddInception("3a", 28, 192, 64, 96, 128, 16, 32, 32, prev);
            prev = AddInception("3b", 28, 256, 128, 128, 192, 32, 96, 64, prev);

            prev = {{graph.AddNode("pool3", nn, LayerType::CONV, prev, 480, 480, 14, 14, 3, 3, 2, 2), Tensor()}};

            // Inception 4
            prev = AddInception("4a", 14, 480, 192, 96, 208, 16, 48, 64, prev);
            prev = AddInception("4b", 14, 512, 160, 112, 224, 24, 64, 64, prev);
            prev = AddInception("4c", 14, 512, 128, 128, 256, 24, 64, 64, prev);
            prev = AddInception("4d", 14, 512, 112, 144, 288, 32, 64, 64, prev);
            prev = AddInception("4e", 14, 528, 256, 160, 320, 32, 128, 128, prev);

            prev = {{graph.AddNode("pool4", nn, LayerType::CONV, prev, 832, 832, 7, 7, 3, 3, 2, 2), Tensor()}};

            // Inception 5
            prev = AddInception("5a", 7, 832, 256, 160, 320, 32, 128, 128, prev);
            prev = AddInception("5b", 7, 832, 384, 192, 384, 48, 128, 128, prev);

            prev = {{graph.AddNode("pool5", nn, LayerType::CONV, prev, 1024, 1024, 1, 1, 7, 7), Tensor()}};
            prev = {{graph.AddNode("fc", nn, LayerType::CONV, prev, 1024, 1000, 1, 1, 1, 1), Tensor()}};
            
            (void) skipped_layers;
            return 64;
        }

        unsigned ResNet50(Graph& graph, std::vector<std::string> skipped_layers) {
            const std::string nn = "ResNet50";
            std::vector<Edge> prev;

            prev = {{graph.AddNode("conv1", nn, LayerType::CONV, prev, 3, 64, 112, 112, 7, 7, 2, 2), Tensor()}};
            prev = {{graph.AddNode("pool1", nn, LayerType::CONV, prev, 64, 64, 56, 56, 3, 3, 2, 2), Tensor()}};
            
            Edge res;
            for (unsigned i = 1; i < 4; i++)
            {
                prev = {{graph.AddNode("conv2_"+std::to_string(i)+"_a", nn, LayerType::CONV, prev, (i == 1) ? 64 : 256, 64, 56, 56, 1, 1), Tensor()}};
                prev = {{graph.AddNode("conv2_"+std::to_string(i)+"_b", nn, LayerType::CONV, prev, 64, 64, 56, 56, 3, 3), Tensor()}};
                prev = {{graph.AddNode("conv2_"+std::to_string(i)+"_c", nn, LayerType::CONV, prev, 64, 256, 56, 56, 1, 1), Tensor()}};

                if (i != 1) prev.push_back(res);
                res = prev[0];
            }

            for (unsigned i = 1; i < 5; i++)
            {
                unsigned ifs = (i == 1) ? 256 : 512;
                unsigned sd = (i == 1) ? 2 : 1;

                prev = {{graph.AddNode("conv3_"+std::to_string(i)+"_a", nn, LayerType::CONV, prev, ifs, 128, 28, 28, 1, 1, sd, sd), Tensor()}};
                prev = {{graph.AddNode("conv3_"+std::to_string(i)+"_b", nn, LayerType::CONV, prev, 128, 128, 28, 28, 3, 3), Tensor()}};
                prev = {{graph.AddNode("conv3_"+std::to_string(i)+"_c", nn, LayerType::CONV, prev, 128, 512, 28, 28, 1, 1), Tensor()}};

                if (i != 1) prev.push_back(res);
                res = prev[0];
            }

            for (unsigned i = 1; i < 7; i++)
            {
                unsigned ifs = (i == 1) ? 512 : 1024;
                unsigned sd = (i == 1) ? 2 : 1;
                
                prev = {{graph.AddNode("conv4_"+std::to_string(i)+"_a", nn, LayerType::CONV, prev, ifs, 256, 14, 14, 1, 1, sd, sd), Tensor()}};
                prev = {{graph.AddNode("conv4_"+std::to_string(i)+"_b", nn, LayerType::CONV, prev, 256, 256, 14, 14, 3, 3), Tensor()}};
                prev = {{graph.AddNode("conv4_"+std::to_string(i)+"_c", nn, LayerType::CONV, prev, 256, 1024, 14, 14, 1, 1), Tensor()}};

                if (i != 1) prev.push_back(res);
                res = prev[0];
            }

            for (unsigned i = 1; i < 4; i++)
            {
                unsigned ifs = (i == 1) ? 1024 : 2048;
                unsigned sd = (i == 1) ? 2 : 1;
                
                prev = {{graph.AddNode("conv5_"+std::to_string(i)+"_a", nn, LayerType::CONV, prev, ifs, 512, 7, 7, 1, 1, sd, sd), Tensor()}};
                prev = {{graph.AddNode("conv5_"+std::to_string(i)+"_b", nn, LayerType::CONV, prev, 512, 512, 7, 7, 3, 3), Tensor()}};
                prev = {{graph.AddNode("conv5_"+std::to_string(i)+"_c", nn, LayerType::CONV, prev, 512, 2048, 7, 7, 1, 1), Tensor()}};

                if (i != 1) prev.push_back(res);
                res = prev[0];
            }
            
            (void) skipped_layers;
            return 50;
        }


        unsigned DebugLayer(Graph& graph, std::vector<std::string> skipped_layers) {
            const std::string nn = "GoogleNetFirstLayer";
            std::vector<Edge> prev;
            graph.AddNode("conv1", nn, LayerType::CONV, prev, 3, 64, 112, 112, 7, 7, 2, 2);

            (void) skipped_layers;
            return 1;
        }

        const std::unordered_map<std::string, std::function<unsigned(moham::Graph&,std::vector<std::string>)>> benchmarksFunc =
        {
            {"GoogleNet", GoogleNet}, 
            {"ResNet50", ResNet50},
            // {"MobileNet-v1", MobileNetV1},
            // {"Tiny-YOLO", TinyYolo},
            // {"SSD-ResNet-34", SSDResNet34},
            // {"SSD-MobileNet-v1", SSDMobileNetV1},
            // {"GNMT", GNMT},
            // {"YOLOv3", YoloV3}
            {"Debug", DebugLayer}, 
        };
    }
    
}

#endif // MOHAM_BENCHMARKS_H_