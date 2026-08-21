#include "inference/tensor.h"

#include <stdexcept>
#include <utility>

namespace inference {
    Tensor::Tensor(TensorShape shape) :
        shape_(std::move(shape)), data_(shape_.numel())
    {}
    Tensor::Tensor(TensorShape shape, std::vector<float> data) :
        shape_(std::move(shape)),
        data_(std::move(data))
    {
        if (data_.size() != shape_.numel()) {
            throw std::invalid_argument("Data does not fit Tensor shape");
        }
    }


    [[nodiscard]] const TensorShape& Tensor::shape() const {
        return shape_;
    }

    [[nodiscard]] const std::vector<float>& Tensor::data() const {
        return data_;
    }

    [[nodiscard]] std::size_t Tensor::numel() const{
        return shape_.numel();
    }

    void Tensor::set_data(std::vector<float> new_data) {
        if (new_data.size() != data_.size()) throw std::invalid_argument("Cannot set data of incompatible size");
        data_ = std::move(new_data);
    }

    const float& Tensor::operator[](const std::size_t index) const {
        return data_[index];
    }

    float& Tensor::operator[](std::size_t index) {
        return data_[index];
    }

    [[nodiscard]] const std::vector<std::size_t>& Tensor::dims() const {
        return shape_.dims();
    }




















}
