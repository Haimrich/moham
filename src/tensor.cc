#include "tensor.h"

#include <vector>
#include <string>

namespace moham {

Tensor::Tensor(std::string name, std::vector<size_t> shape) : name_(name), shape_(shape) {}

const std::string Tensor::Name() const {
    return name_;
}

const std::vector<size_t> Tensor::Shape() const {
    return shape_;
}

bool Tensor::operator==(const Tensor& other) const {
    return std::equal(shape_.begin(), other.shape_.begin(), other.shape_.end())
            && name_ == other.name_;
}

}