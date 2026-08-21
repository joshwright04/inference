#include <cstddef>
#include <gtest/gtest.h>
#include <vector>
#include "inference/tensor_shape.h"

namespace inference {
namespace {

TEST(TensorShapeTest, ComputesRankAndElementCount) {
    TensorShape shape({2, 3, 4});
    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape.numel(), 24);
    EXPECT_EQ(shape.dims(), std::vector<std::size_t>({2, 3, 4}));
}

TEST(TensorShapeTest, ScalarHasRankZeroAndOneElement) {
    TensorShape shape({});
    EXPECT_EQ(shape.rank(), 0);
    EXPECT_EQ(shape.numel(), 1);
}

TEST(TensorShapeTest, ZeroDimensionsHasZeroElements) {
    TensorShape shape({2, 0, 4});
    EXPECT_EQ(shape.numel(), 0);
}

TEST(TensorShapeTest, EmptyVectorDimensionsAreAllowed) {
    TensorShape shape({2, 0, 4});
    EXPECT_EQ(shape.numel(), 0);
}

}  // namespace
}  // namespace inference
