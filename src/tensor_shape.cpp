#include "inference/tensor_shape.h"

#include <utility>

namespace inference {
    //Member-initialization constructor
    TensorShape::TensorShape(std::vector<std::size_t> shape) : shape_(std::move(shape)) {}

    std::size_t TensorShape::numel() const {
        std::size_t result{1};

        if (rank() == 0) return result; //Scalar tensor, has one element

        for (const std::size_t dimension : shape_) {
            if (dimension == 0) return 0; // If one dimension is 0, numel = 0
            result *= dimension;
        }

        return result;
    }

    std::size_t TensorShape::rank() const {
        return shape_.size();
    }

    const std::vector<std::size_t>& TensorShape::dims() const {
        return shape_;
    }

    // Compare shapes by rank and per-dimension extent.
    bool TensorShape::operator==(const TensorShape& shape) const {
        return dims() == shape.dims();
    }




}
