// Launcher declarations shared by the .cu translation units and bindings.cpp.
// Including this from the .cu files turns a signature drift into a compile
// error instead of an LNK2019 at link time.
#pragma once

#include <torch/extension.h>

#include <optional>

torch::Tensor add_one(torch::Tensor x);

// y = x @ W^T for a packed W. x is fp16 [M, K] with 1 <= M <= 8; returns fp16 [M, N].
torch::Tensor gemv(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
                   std::optional<torch::Tensor> zeros, int64_t bits, int64_t group_size,
                   bool symmetric, bool interleaved);

// Materialize the packed weight. Returns fp16 [N, K], bit-exact with packer.dequantize.
torch::Tensor dequantize(torch::Tensor qweight, torch::Tensor scales,
                         std::optional<torch::Tensor> zeros, int64_t bits, int64_t group_size,
                         bool symmetric, bool interleaved, int64_t K);
