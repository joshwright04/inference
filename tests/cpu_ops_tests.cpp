#include <cstddef>
#include <gtest/gtest.h>
#include <vector>
#include "inference/cpu_ops.h"

namespace inference {
    namespace {
        TEST(CpuOpsTest, AddsElementwise) {
            const auto shape = TensorShape({2, 2});
            Tensor a(shape, {1.0F, 2.0F, 3.0F, 4.0F});
            Tensor b(shape, {2.0F, 3.0F, 4.0F, 5.0F});

            const std::vector expected_result = {3.0F, 5.0F, 7.0F, 9.0F};

            EXPECT_EQ(add(a, b).data(), expected_result);
        }

        TEST(CpuOpsTest, PreservesShape) {
            const auto shape = TensorShape({2, 2});
            Tensor a(shape);
            Tensor b(shape);

            Tensor result = add(a, b);

            EXPECT_EQ(result.shape(), shape);
        }

        TEST(CpuOpsTest, AddsScalarTensors) {
            const auto shape = TensorShape({});
            Tensor a(shape, {1.0F});
            Tensor b(shape, {2.0F});

            std::vector<float> expected = {3.0F};
            EXPECT_EQ(add(a, b).data(), expected);
        }

        TEST(CpuOpsTest, ThrowsForMismatchedShapes) {
            const auto shape1 = TensorShape({});
            const auto shape2 = TensorShape({2, 4});
            const Tensor a(shape1);
            const Tensor b(shape2);
            EXPECT_THROW(add(a, b), std::invalid_argument);

        }

        TEST(CpuOpsTest, HandlesZeroElementTensors) {
            const auto shape = TensorShape({2, 0});
            const Tensor a(shape);
            const Tensor b(shape);
            const Tensor c = add(a, b);
            EXPECT_TRUE(c.data().empty());
        }

        TEST(CpuOpsTest, AddDoesNotModifyTensors) {
            const auto shape = TensorShape({});
            const Tensor a(shape);
            const Tensor b(shape);
            const auto data1 = a.data();
            const auto data2 = b.data();
            const Tensor c = add(a, b);
            EXPECT_EQ(a.data(), data1);
            EXPECT_EQ(b.data(), data2);
        }

        TEST(CpuOpsTest, SameNumelDifferentShapeFails) {
            const auto shape1 = TensorShape({1, 2});
            const auto shape2 = TensorShape({2, 1});
            const Tensor a(shape1);
            const Tensor b(shape2);
            EXPECT_THROW(add(a, b), std::invalid_argument);
        }

        TEST(CpuOpsTest, ReluKeepsPositiveValues) {
            const Tensor tensor(TensorShape({1, 2}), {2.0F, -5.0F});
            Tensor result = relu(tensor);
            EXPECT_TRUE(result[0] == 2.0F);
        }

        TEST(CpuOpsTest, ReluClampsNegatives) {
            const Tensor tensor(TensorShape({1, 2}), {2.0F, -5.0F});
            Tensor result = relu(tensor);
            EXPECT_TRUE(result[1] == 0.0F);
        }

        TEST(CpuOpsTest, ReluPreservesShape) {
            const Tensor tensor(TensorShape({1, 2}), {2.0F, -5.0F});
            Tensor result = relu(tensor);
            EXPECT_TRUE(result.shape() == tensor.shape());
        }

        TEST(CpuOpsTest, ReluHandlesScalarTensor) {
            const Tensor tensor(TensorShape({}), {-5.0F});
            Tensor result = relu(tensor);
            EXPECT_EQ(result[0], 0.0F);
        }

        TEST(CpuOpsTest, ReluHandlesZeroElementTensor) {
            const Tensor tensor(TensorShape({2, 0}));
            Tensor result = relu(tensor);
            EXPECT_EQ(result.data(), tensor.data());
        }

        TEST(CpuOpsTest, ReluDoesNotModifyInput) {
            const Tensor tensor(TensorShape({}), {-5.0F});
            Tensor result = relu(tensor);
            EXPECT_TRUE(tensor[0] == -5.0F);
        }

        TEST(CpuOpsTest, MatVecComputesOutput) {
            const Tensor matrix(TensorShape({2, 3}), {
                1.0F, 2.0F, 3.0F,
                4.0F, 5.0F, 6.0F,
            });
            const Tensor vector(TensorShape({3}), {10.0F, 20.0F, 30.0F});

            const Tensor result = matvec(matrix, vector);

            EXPECT_EQ(result.data(), std::vector<float>({140.0F, 320.0F}));
        }

        TEST(CpuOpsTest, MatVecPreservesExpectedOutputShape) {
            const Tensor matrix(TensorShape({2, 3}));
            const Tensor vector(TensorShape({3}));

            const Tensor result = matvec(matrix, vector);

            EXPECT_EQ(result.shape(), TensorShape({2}));
        }

        TEST(CpuOpsTest, MatVecThrowsForNonMatrixInput) {
            const Tensor matrix(TensorShape({3}));
            const Tensor vector(TensorShape({3}));

            EXPECT_THROW(matvec(matrix, vector), std::invalid_argument);
        }

        TEST(CpuOpsTest, MatVecThrowsForNonVectorInput) {
            const Tensor matrix(TensorShape({2, 3}));
            const Tensor vector(TensorShape({3, 1}));

            EXPECT_THROW(matvec(matrix, vector), std::invalid_argument);
        }

        TEST(CpuOpsTest, MatVecThrowsForIncompatibleDimensions) {
            const Tensor matrix(TensorShape({2, 3}));
            const Tensor vector(TensorShape({2}));

            EXPECT_THROW(matvec(matrix, vector), std::invalid_argument);
        }

        TEST(CpuOpsTest, MatVecHandlesZeroRows) {
            const Tensor matrix(TensorShape({0, 3}));
            const Tensor vector(TensorShape({3}));

            const Tensor result = matvec(matrix, vector);

            EXPECT_EQ(result.shape(), TensorShape({0}));
            EXPECT_TRUE(result.data().empty());
        }

        TEST(CpuOpsTest, MatMulComputesOutput) {
            const Tensor a(TensorShape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
            const Tensor b(TensorShape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

            EXPECT_EQ(matmul(a, b).data(), std::vector<float>({58.0F, 64.0F, 139.0F, 154.0F}));
        }

        TEST(CpuOpsTest, MatMulPreservesOutputShape) {
            const Tensor a(TensorShape({2, 3}), {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F});
            const Tensor b(TensorShape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

            EXPECT_EQ(matmul(a, b).dims(), std::vector<std::size_t>({2, 2}));
        }

        TEST(CpuOpsTest, MatMulThrowsForNonMatrixLeftInput) {
            const Tensor a(TensorShape({1}));
            const Tensor b(TensorShape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});

            EXPECT_THROW(matmul(a, b), std::invalid_argument);
        }

        TEST(CpuOpsTest, MatMulThrowsForNonMatrixRightInput) {
            const Tensor a(TensorShape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
            const Tensor b(TensorShape({1}));

            EXPECT_THROW(matmul(a, b), std::invalid_argument);
        }

        TEST(CpuOpsTest, MatMulThrowsForIncompatibleDimensions) {
            const Tensor a(TensorShape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
            const Tensor b(TensorShape({3, 2}));

            EXPECT_THROW(matmul(a, b), std::invalid_argument);
        }

        TEST(CpuOpsTest, MatMulHandlesZeroRows) {
            const Tensor a(TensorShape({0, 2}));
            const Tensor b(TensorShape({2, 3}), {
                1.0F, 2.0F, 3.0F,
                4.0F, 5.0F, 6.0F,
            });

            const Tensor result = matmul(a, b);

            EXPECT_EQ(result.shape(), TensorShape({0, 3}));
            EXPECT_TRUE(result.data().empty());
        }

        TEST(CpuOpsTest, MatMulHandlesZeroColumns) {
            const Tensor a(TensorShape({2, 3}), {
                1.0F, 2.0F, 3.0F,
                4.0F, 5.0F, 6.0F,
            });
            const Tensor b(TensorShape({3, 0}));

            const Tensor result = matmul(a, b);

            EXPECT_EQ(result.shape(), TensorShape({2, 0}));
            EXPECT_TRUE(result.data().empty());
        }

        TEST(CpuOpsTest, MatMulDoesNotModifyInputs) {
            const Tensor a(TensorShape({3, 2}), {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F});
            const std::vector<float> data = a.data();
            const Tensor b(TensorShape({2, 3}));
            const Tensor c = matmul(a, b);
            EXPECT_EQ(a.data(), data);
        }





    }  // namespace
}  // namespace inference
