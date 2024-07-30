#ifndef MOHAM_TENSOR_H_
#define MOHAM_TENSOR_H_

#include <string>
#include <vector>

namespace moham {

    class Tensor
    {
    private:
        std::string name_;
        std::vector<size_t> shape_;
        
    public:
        Tensor() = default;
        
        Tensor(std::string name, std::vector<size_t> shape);

        const std::string Name() const;
        const std::vector<size_t> Shape() const;

        bool operator==(const Tensor& other) const;
    };

}

#endif // MOHAM_TENSOR_H_