#include "inference/cpu_ops.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace inference {
namespace {

    void require_rank(const Tensor& tensor, std::size_t rank) {
        if (tensor.shape().rank() != rank) throw std::invalid_argument("Invalid tensor rank");
    }

    void require_same_shape(const Tensor& a, const Tensor& b) {
        if (!(a.shape() == b.shape())) throw std::invalid_argument("Tensors must be of equal shape");
    }

}

    Tensor add(const Tensor& a, const Tensor& b) {
        require_same_shape(a, b);
        const TensorShape shape(a.shape().dims());
        auto c = Tensor(shape);

        for (std::size_t i = 0; i < c.data().size(); ++i) {
            c[i] = a[i] + b[i];
        }

        return c;
    }

    Tensor relu(Tensor tensor) {
        for (std::size_t i = 0; i < tensor.data().size(); ++i) {
            tensor[i] = std::max(0.0F, tensor[i]);
        }

        return tensor;
    }

    Tensor matvec(const Tensor& matrix, const Tensor& vector) {
        require_rank(matrix, 2);
        require_rank(vector, 1);
        if ( matrix.shape().dims()[1] != vector.shape().dims()[0] ) {
            throw std::invalid_argument("Matrix row length must equal vector length");
        }

        const std::size_t rows = matrix.shape().dims()[0];
        const std::size_t cols = matrix.shape().dims()[1];
        Tensor result(TensorShape({rows}));

        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                result[row] += matrix[row * cols + col] * vector[col];
            }
        }

        return result;
    }

    Tensor matmul(const Tensor& a, const Tensor& b) {
        require_rank(a, 2);
        require_rank(b, 2);

        if ( a.dims()[1] != b.dims()[0] ) {
            throw std::invalid_argument("Inner matrix dimensions must match");
        }
        /* Shape rule:
         * a: {m, k}
         * b: {k, n}
         * c: {m, n}
         */
        std::size_t m = a.dims()[0];
        std::size_t n = b.dims()[1];
        std::size_t k = a.dims()[1];

        Tensor result(TensorShape({m, n}));

        /* For each element in matrix a (loop through rows and columns):
            * For each element in the b's matching column
                * c += a * b
         */

        for (std::size_t row = 0; row < m; ++row) {
            for (std::size_t col = 0; col < n; ++col) {
                float sum = 0.0F;

                for (std::size_t inner = 0; inner < k; ++inner) {
                    sum += a[row * k + inner] * b[inner * n + col];
                }

                result[row * n + col] = sum;
            }
        }

        return result;
    }

}
