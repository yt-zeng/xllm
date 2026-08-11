/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/py_executor_impl.h"

#include <Python.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/mtp_topk_state.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/runtime/py_attention_metadata.h"
#include "models/llm/py_causal_lm.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace py = pybind11;

namespace xllm {
namespace {

class PyMtpTopkState final : public MtpTopkState {
 public:
  explicit PyMtpTopkState(torch::Tensor topk_indices)
      : topk_indices_(std::move(topk_indices)) {
    CHECK(topk_indices_.defined())
        << "Python MTP DSA top-k indices must be defined.";
    CHECK_GE(topk_indices_.dim(), 1)
        << "Python MTP DSA top-k indices must have at least one dimension.";
  }

  const torch::Tensor& topk_indices() const { return topk_indices_; }

  int64_t num_rows() const override { return topk_indices_.size(0); }

  torch::Device device() const override { return topk_indices_.device(); }

  MtpTopkStatePtr to(const torch::Device& device) const override {
    return std::make_shared<PyMtpTopkState>(topk_indices_.to(
        topk_indices_.options().device(device), /*non_blocking=*/true));
  }

  MtpTopkStatePtr index_select_rows(const torch::Tensor& index) const override {
    return std::make_shared<PyMtpTopkState>(
        topk_indices_.index_select(/*dim=*/0, index));
  }

 private:
  torch::Tensor topk_indices_;
};

py::object optional_tensor(const torch::Tensor& tensor) {
  return tensor.defined() ? py::cast(tensor) : py::none();
}

void clear_python_object(py::object& object) {
  if (!object) {
    return;
  }
  if (!Py_IsInitialized()) {
    (void)object.release();
    return;
  }
  py::gil_scoped_acquire gil;
  object = py::object();
}

}  // namespace

PYBIND11_EMBEDDED_MODULE(xllm_runtime, m) {
  register_attention_metadata_views(m);

#if defined(USE_NPU)
  py::class_<NPULayerSynchronizerImpl,
             std::shared_ptr<NPULayerSynchronizerImpl>>(m, "LayerSynchronizer")
      .def("record_event",
           [](NPULayerSynchronizerImpl& self, int64_t layer_id) {
             int32_t device_id = static_cast<int32_t>(
                 c10_npu::getCurrentNPUStream().device_index());
             return self.record_event(layer_id, device_id);
           });
#endif
}

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_causal_lm_(dynamic_cast<PyCausalLM*>(model)),
      args_(args),
      device_(device),
      options_(options),
      enable_mla_(args.enable_mla()) {
  CHECK(py_causal_lm_ != nullptr) << "PyExecutorImpl requires PyCausalLM";

  py::gil_scoped_acquire gil;
  py::module_::import("xllm_runtime");
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  int32_t graph_max_seqs_per_batch = options_.max_seqs_per_batch();
#if defined(USE_NPU)
  graph_max_seqs_per_batch = std::min(
      graph_max_seqs_per_batch,
      std::max<int32_t>(
          1,
          ExecutionConfig::get_instance().acl_graph_decode_batch_size_limit()));
#endif
  py_executor_ =
      executor_module.attr("ModelExecutor")(py_causal_lm_->python_model(),
                                            py_causal_lm_->config_dict(),
                                            graph_max_seqs_per_batch);
}

PyExecutorImpl::~PyExecutorImpl() { clear_python_object(py_executor_); }

ForwardInput PyExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }

  py::gil_scoped_acquire gil;

  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py::list kv_caches_py;
    for (auto& kv : kv_caches) {
      // Slot order must match ``LayerCache`` on the Python side.
      const std::optional<torch::Tensor> indexer_cache_scale =
          kv.get_indexer_cache_scale();
      py::object indexer_cache_scale_py =
          indexer_cache_scale.has_value()
              ? py::cast(indexer_cache_scale.value())
              : py::none();
      kv_caches_py.append(py::make_tuple(optional_tensor(kv.get_k_cache()),
                                         optional_tensor(kv.get_v_cache()),
                                         optional_tensor(kv.get_index_cache()),
                                         optional_tensor(kv.get_conv_cache()),
                                         optional_tensor(kv.get_ssm_cache()),
                                         indexer_cache_scale_py));
    }
    py_executor_.attr("bind_kv_caches")(kv_caches_py);
    kv_bound_ = true;
    kv_layer_count_ = num_layers;
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
  }

  py::object py_metadata =
      py::cast(PyAttentionMetadataView(attn_metadata, params));
  py::object input_embedding = params.embedding.input_embedding.defined()
                                   ? py::cast(params.embedding.input_embedding)
                                   : py::none();
  py::object mtp_topk_state = py::none();
  if (params.mtp_topk_state != nullptr) {
    const auto state =
        std::dynamic_pointer_cast<const PyMtpTopkState>(params.mtp_topk_state);
    CHECK(state != nullptr)
        << "Python MTP model received an incompatible top-k state.";
    mtp_topk_state = py::cast(state->topk_indices());
  }

  py::object py_sync = py::none();
#if defined(USE_NPU)
  if (params.parallel.layer_synchronizer) {
    py_sync = py::cast(params.parallel.layer_synchronizer);
  }
#endif

  // Execute: one C++ -> Python call per step.
  py::object result = py_executor_.attr("execute")(
      tokens, positions, py_metadata, input_embedding, mtp_topk_state, py_sync);
  if (!py::isinstance<py::tuple>(result)) {
    return ModelOutput(result.cast<torch::Tensor>());
  }

  py::tuple output_tuple = result.cast<py::tuple>();
  CHECK_EQ(output_tuple.size(), 2)
      << "Python model execute tuple must contain hidden states and MTP top-k.";
  ModelOutput output(output_tuple[0].cast<torch::Tensor>());
  if (!output_tuple[1].is_none()) {
    output.mtp_topk_state =
        std::make_shared<PyMtpTopkState>(output_tuple[1].cast<torch::Tensor>());
  }
  return output;
}

}  // namespace xllm
