#pragma once

#include <cstddef>
#include <vector>

#include "tensor_shape.h"

namespace inference {

    class Tensor {
    public:
        explicit Tensor(TensorShape shape);
        Tensor(TensorShape shape, std::vector<float> data);

        [[nodiscard]] const TensorShape& shape() const;
        [[nodiscard]] std::size_t numel() const;

        //Immutable (read-only) reference to tensor's data.
        [[nodiscard]] const std::vector<float>& data() const;

        void set_data(std::vector<float> new_data);

        /* Write to data at index
         * Overload the vector[] operator. Returns mutable reference.
         * Works OK, unlike vector mutable reference, since a float is always just one element
         */
        float& operator[](std::size_t index);

        /* Read data at index
         * Overload the vector[] operator. Returns immutable (read-only) reference
         */
        const float& operator[](std::size_t index) const;

        [[nodiscard]] const std::vector<std::size_t>& dims() const;

    private:
        TensorShape shape_;
        std::vector<float> data_;
    };



}  // namespace inference
