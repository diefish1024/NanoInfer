#pragma once
#include "tensor.hpp"

void silu(const Tensor& x, Tensor& out);
void embedding(const Tensor& idx, const Tensor& w, Tensor& out);
void softmax(const Tensor& x, Tensor& out);
void rope(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin, int start_pos);