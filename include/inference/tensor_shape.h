#pragma once

#include <cstddef>
#include <vector>

namespace inference {

    class TensorShape {
    public:
        explicit TensorShape(std::vector<std::size_t> shape);

        [[nodiscard]] std::size_t numel() const;

        [[nodiscard]] std::size_t rank() const;

        [[nodiscard]] const std::vector<std::size_t>& dims() const;

        [[nodiscard]] bool operator==(const TensorShape& shape) const;

    private:
        const std::vector<std::size_t> shape_;
    };



}  // namespace inference
