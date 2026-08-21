#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "inference/tensor.h"
#include "inference/tensor_shape.h"

namespace inference {
    namespace {

        TEST(TensorTest, HoldsCorrectTensorShape) {
            TensorShape shape({2, 3, 4});
            Tensor tensor(shape);
            EXPECT_EQ(tensor.shape(), shape);
            EXPECT_EQ(tensor.shape().numel(), shape.numel());
            EXPECT_EQ(tensor.shape().rank(), shape.rank());
        }

        TEST(TensorTest, AllocatesOneFloatPerElement) {
            TensorShape shape({2, 2});
            Tensor tensor(shape);

            EXPECT_EQ(tensor.data().size(), 4);
            EXPECT_EQ(tensor.data().size(), tensor.numel());
        }

        TEST(TensorTest, DataIsZeroInitialized) {
            TensorShape shape({2, 2});
            Tensor tensor(shape);

            const std::vector<float> expected_data = {0, 0, 0, 0};
            EXPECT_EQ(expected_data, tensor.data());
        }

        TEST(TensorTest, MutableDataCanBeWritten) {
            TensorShape shape({2, 2});
            Tensor tensor(shape, {1.2F, 3.1F, 5.5F, 6.0F});

            const std::vector<float> expected_data = {1.2F, 3.1F, 5.5F, 6.0F};
            EXPECT_EQ(tensor.data(), expected_data);
        }


        TEST(TensorTest, ScalarTensorStoresOneFloat) {
            TensorShape shape({});
            Tensor tensor(shape);

            EXPECT_EQ(tensor.data().size(), 1);
        }

        TEST(TensorTest, ZeroElementTensorStoresNoData) {
            TensorShape shape({1, 0 , 5});
            Tensor tensor(shape);

            EXPECT_TRUE(tensor.data().empty());
        }

        TEST(TensorTest, ConstructsWithInitialData) {
            const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
            const TensorShape shape({2, 2});
            const Tensor tensor(shape, data);
            EXPECT_EQ(tensor.data(), data);
        }

        TEST(TensorTest, ConstructWithIncompatibleDataFails) {
            const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
            const TensorShape shape({2, 1});
            EXPECT_THROW(Tensor(shape, data), std::invalid_argument);
        }

        TEST(TensorTest, CanAccessDataAtIndex) {
            const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
            const TensorShape shape({2, 2});
            const auto tensor = Tensor(shape, data);
            EXPECT_EQ(tensor[0], 1.0F);
        }

        TEST(TensorTest, CanModifyDataAtIndex) {
            const std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
            const TensorShape shape({2, 2});
            auto tensor = Tensor(shape, data);
            tensor[0] = 5.0F;
            EXPECT_EQ(tensor[0], 5.0F);
        }

    }
}
