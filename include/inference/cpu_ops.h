#pragma once

#include "inference/tensor.h"
namespace inference {
    //const reference: avoids expensive copy, but can't modify a and b
    Tensor add(const Tensor& a, const Tensor& b);

    Tensor relu(Tensor tensor);

    // Matrix-vector multiply
    Tensor matvec(const Tensor& matrix, const Tensor& vector);

    Tensor matmul(const Tensor& a, const Tensor& b);








}


