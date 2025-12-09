#pragma once
#include "tensor.hpp"

void silu(const Tensor& x, Tensor& out);
void embedding(const Tensor& idx, const Tensor& w, Tensor& out);
void softmax(const Tensor& x, Tensor& out);