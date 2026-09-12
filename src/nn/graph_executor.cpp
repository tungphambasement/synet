/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#include "nn/graph_executor.hpp"

#include <fmt/core.h>

#include <chrono>
#include <ostream>

#include "device/device_allocator.hpp"
#include "device/device_manager.hpp"
#include "device/pool_allocator.hpp"
#include "nn/edge.hpp"
#include "nn/edge_profile.hpp"
#include "nn/execution_plan.hpp"
#include "nn/graph.hpp"
#include "nn/macro_solver.hpp"
#include "nn/memory_packer.hpp"
#include "nn/tensor_bundle.hpp"
#include "tensor/ops.hpp"
#include "tensor/tensor.hpp"

namespace tunx {

namespace {

class ParameterStateGuard {
public:
  ParameterStateGuard(const Vec<Param> &params, Device &device, stream s)
      : stream_(s),
        params_(params) {
    auto &allocator = PoolAllocator::instance(device, s);
    for (const auto &param : params) {
      if (!param || param.size() == 0) {
        snapshots_.push_back({});
        continue;
      }

      Snapshot snapshot;
      snapshot.data = Tensor(param.shape(), param.dtype(), allocator);
      copy(param.data(), snapshot.data, stream_);
      if (param.grad()) {
        snapshot.grad = Tensor(param.shape(), param.dtype(), allocator);
        copy(param.grad(), snapshot.grad, stream_);
      }
      snapshots_.push_back(std::move(snapshot));
    }
    stream_.sync();
  }

  ~ParameterStateGuard() {
    for (size_t i = 0; i < snapshots_.size(); ++i) {
      if (!snapshots_[i].data) continue;
      copy(snapshots_[i].data, params_[i].data(), stream_);
      if (snapshots_[i].grad) {
        copy(snapshots_[i].grad, params_[i].grad(), stream_);
      }
    }
    stream_.sync();
  }

  ParameterStateGuard(const ParameterStateGuard &) = delete;
  ParameterStateGuard &operator=(const ParameterStateGuard &) = delete;

private:
  struct Snapshot {
    Tensor data;
    Tensor grad;
  };

  stream stream_;
  Vec<Param> params_;
  Vec<Snapshot> snapshots_;
};

}  // namespace

GraphExecutor::GraphExecutor(Graph &graph, bool bootstrap_offload)
    : graph_(graph),
      bootstrap_offload_(bootstrap_offload),
      host_allocator_(std::make_unique<OffloadAllocator>(&DeviceAllocator::instance(getHost()))) {
  for (const auto &edge : graph_.edges()) {
    for (const auto &producer : edge->producers()) {
      ++data_ref_counts_[producer];
    }
    for (const auto &consumer : edge->consumers()) {
      ++grad_ref_counts_[consumer];
    }
  }
  for (const auto &output : graph_.outputs()) {
    ++data_ref_counts_[output];
  }
  for (const auto &input : graph_.inputs()) {
    ++grad_ref_counts_[input];
  }
}

const Tensor &GraphExecutor::data(const Node &node) const { return data_.at(node).tensor; }

void GraphExecutor::set_data(const Node &node, const Tensor &tensor, int ref_count) {
  data_[node] = {tensor, ref_count};
}

void GraphExecutor::release_data(const Node &node) {
  auto it = data_.find(node);
  if (it != data_.end() && it->second.ref_count > 0 && --it->second.ref_count == 0) {
    data_.erase(it);
  }
}

const Tensor &GraphExecutor::grad(const Node &node) const { return grads_.at(node).tensor; }

void GraphExecutor::set_grad(const Node &node, const Tensor &tensor, int ref_count) {
  grads_[node] = {tensor, ref_count};
}

void GraphExecutor::accumulate_grad(const Node &node, const Tensor &tensor, int ref_count) {
  auto [it, inserted] = grads_.try_emplace(node, Entry{tensor, ref_count});
  if (!inserted) {
    add(it->second.tensor, tensor, it->second.tensor, graph_.handle().get_stream());
  }
}

void GraphExecutor::release_grad(const Node &node) {
  auto it = grads_.find(node);
  if (it != grads_.end() && it->second.ref_count > 0 && --it->second.ref_count == 0) {
    grads_.erase(it);
  }
}

void GraphExecutor::cleanup_released(std::map<Node, Entry> &entries) {
  for (auto it = entries.begin(); it != entries.end();) {
    if (it->second.ref_count == 0) {
      it = entries.erase(it);
    } else {
      ++it;
    }
  }
}

const BuiltPlan &GraphExecutor::build_plans(TensorBundle &input_map, SolverOptions options) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  bool is_training = false;
  for (const auto &edge : graph_.edges()) {
    if (edge->layer() && edge->layer()->is_training()) {
      is_training = true;
      break;
    }
  }

  PlanKey key;
  key.enable_naive = options.enable_naive;
  key.enable_linear = options.enable_linear;
  key.enable_branching = options.enable_branching;
  key.enable_joining = options.enable_joining;
  key.is_training = is_training;
  key.device = graph_.device().get_id();

  for (const auto &[uid, tensor] : input_map) {
    auto node = uid_to_node[uid];
    key.input_shapes[node] = tensor.shape();
    key.input_dtypes[node] = tensor.dtype();
  }
  if (built_plans_.count(key)) {
    return built_plans_.at(key);
  }

  ParameterStateGuard parameter_state_guard(graph_.params(), graph_.device(),
                                            graph_.handle().get_stream());

  auto start_total = std::chrono::high_resolution_clock::now();
  MacroSolver planner(graph_, os_, options);
  auto start_extraction_fw = std::chrono::high_resolution_clock::now();
  auto [output_map, forward_edge_profiles, node_profiles] = profile_edges_forward(input_map, true);
  auto end_extraction_fw = std::chrono::high_resolution_clock::now();
  double extraction_time_ms =
      std::chrono::duration<double, std::milli>(end_extraction_fw - start_extraction_fw).count();

  auto start_scheduling_fw = std::chrono::high_resolution_clock::now();
  ExecutionPlan forward_plan;
  if (options.enable_naive) {
    forward_plan.order = graph_.edges();
  } else {
    forward_plan = planner.find_forward_order(forward_edge_profiles);
  }
  auto end_scheduling_fw = std::chrono::high_resolution_clock::now();
  double scheduling_time_ms =
      std::chrono::duration<double, std::milli>(end_scheduling_fw - start_scheduling_fw).count();

  output_map.clear();

  auto start_extraction_fw2 = std::chrono::high_resolution_clock::now();
  for (const auto &[uid, tensor] : input_map) {
    auto node = uid_to_node[uid];
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(node, device_tensor, data_ref_counts_[node]);
  }

  bool should_offload = is_training && bootstrap_offload_;

  auto s = graph_.handle().get_stream();
  for (const Edge &edge : forward_plan.order) {
    forward_edge(edge);

    if (should_offload) {
      std::unordered_map<void *, Tensor> offloaded_aliases;
      residuals_[edge].apply_tensors([&](Tensor &tensor) {
        if (!tensor) return;
        if (tensor.device() != getHost()) {
          void *source = tensor.data_as<void>();
          auto found = offloaded_aliases.find(source);
          if (found != offloaded_aliases.end()) {
            tensor = found->second;
            return;
          }
          Tensor offloaded(tensor.shape(), tensor.dtype(), *host_allocator_);
          copy(tensor, offloaded, s);
          s.sync();
          offloaded_aliases.emplace(source, offloaded);
          tensor = std::move(offloaded);
        }
      });
    }

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }
  cleanup_released(data_);
  auto end_extraction_fw2 = std::chrono::high_resolution_clock::now();
  extraction_time_ms +=
      std::chrono::duration<double, std::milli>(end_extraction_fw2 - start_extraction_fw2).count();

  TensorBundle output_grad_map;
  auto &grad_allocator = PoolAllocator::instance(graph_.device(), graph_.handle().get_stream());
  for (const auto &[uid, tensor] : output_map) {
    Tensor t(tensor.shape(), tensor.dtype(), grad_allocator);
    // zero init so that it does not corrupt param gradients.
    fill(t, 0.0, graph_.handle().get_stream());
    output_grad_map.set(uid, t);
  }

  output_map.clear();

  graph_.workspace_allocator()->evict_unused();

  std::map<Edge, EdgeProfile> backward_edge_profiles;
  ExecutionPlan backward_plan;
  if (is_training) {
    auto start_extraction_bw = std::chrono::high_resolution_clock::now();
    auto profile_res = profile_edges_backward(output_grad_map, should_offload);
    backward_edge_profiles = std::move(profile_res.second);
    profile_res.first.clear();
    graph_.workspace_allocator()->evict_unused();
    auto end_extraction_bw = std::chrono::high_resolution_clock::now();
    extraction_time_ms +=
        std::chrono::duration<double, std::milli>(end_extraction_bw - start_extraction_bw).count();

    auto start_scheduling_bw = std::chrono::high_resolution_clock::now();
    if (options.enable_naive) {
      for (auto it = graph_.edges().rbegin(); it != graph_.edges().rend(); ++it) {
        backward_plan.order.push_back(*it);
      }
    } else {
      backward_plan = planner.find_backward_order(backward_edge_profiles);

      // The MacroSolver may schedule edges in an order that differs from the naive
      // reverse-topological order used above to classify backward buffer roles
      // (BufferRole::GradientOutput vs Workspace). That classification is order-sensitive
      // at multi-producer (fan-out/accumulate) nodes such as residual/skip connections: it
      // decides which edge "owns" the persistent buffer that the gradient is accumulated
      // into. If the memory packer (pack_memory) later lays out buffers using
      // backward_plan.order but roles computed under a different order, it can free/reuse
      // a gradient's memory before all of its contributions have actually been written,
      // silently corrupting gradients. Re-profile backward buffer roles using the real
      // scheduled order so they stay consistent with what pack_memory will assume.
      //
      // profile_edges_backward() consumes (erases) residuals_ as it runs, so residuals
      // must be regenerated via a fresh forward pass before re-profiling. Any parameter
      // gradients accumulated by this extra pass are discarded the same way the first
      // profiling pass's gradients are: callers are expected to zero gradients after the
      // graph is compiled (i.e. after the first forward() call) and before the first real
      // backward() call.
      data_.clear();
      for (const auto &[uid, tensor] : input_map) {
        auto node_it = uid_to_node.find(uid);
        if (node_it == uid_to_node.end()) continue;
        Tensor device_tensor = tensor;
        if (tensor.device() != graph_.device()) {
          device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
          copy(tensor, device_tensor, graph_.handle().get_stream());
        }
        set_data(node_it->second, device_tensor, data_ref_counts_[node_it->second]);
      }
      for (const Edge &edge : forward_plan.order) {
        forward_edge(edge);
        for (const Node &consumer : edge->consumers()) {
          if (graph_.is_output(consumer)) {
            release_data(consumer);
          }
        }
      }
      cleanup_released(data_);
      graph_.workspace_allocator()->evict_unused();

      auto reprofile_res =
          profile_edges_backward(output_grad_map, should_offload, &backward_plan.order);
      backward_edge_profiles = std::move(reprofile_res.second);
      reprofile_res.first.clear();
      graph_.workspace_allocator()->evict_unused();
    }
    auto end_scheduling_bw = std::chrono::high_resolution_clock::now();
    scheduling_time_ms +=
        std::chrono::duration<double, std::milli>(end_scheduling_bw - start_scheduling_bw).count();
  }

  BuiltPlan plan{forward_plan,
                 backward_plan,
                 std::move(forward_edge_profiles),
                 std::move(backward_edge_profiles),
                 std::move(node_profiles),
                 nullptr};

  clear_residuals();
  host_allocator_->evict_unused();

  pack_memory(plan, input_map, output_grad_map);

  output_grad_map.clear();

  plan.extraction_time_ms = extraction_time_ms;
  plan.scheduling_time_ms = scheduling_time_ms;
  auto end_total = std::chrono::high_resolution_clock::now();
  plan.total_time_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

  auto [it, inserted] = built_plans_.emplace(key, std::move(plan));

  graph_.workspace_allocator()->evict_unused();

  return it->second;
}

TensorBundle GraphExecutor::forward(TensorBundle &input_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  PlanKey key;
  key.enable_naive = false;
  key.enable_linear = true;
  key.enable_branching = true;
  key.enable_joining = true;
  key.is_training = (graph_.mode() == ExecutionMode::TRAIN);
  key.device = graph_.device().get_id();
  for (const auto &[uid, tensor] : input_map) {
    auto &node = uid_to_node.at(uid);
    key.input_shapes[node] = tensor.shape();
  }

  auto it = built_plans_.find(key);
  if (it == built_plans_.end()) {
    it = built_plans_.emplace(key, build_plans(input_map)).first;
  }
  active_built_plan_ = it->second;
  active_built_plan_.packed_allocator->set_backend_allocator(graph_.workspace_allocator());

  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }
  for (auto &edge : graph_.edges()) {
    if (edge->layer())
      edge->layer()->set_workspace_allocator(active_built_plan_.packed_allocator.get());
  }

  TensorBundle output_map;
  for (Edge &edge : active_built_plan_.forward_plan.order) {
    active_built_plan_.packed_allocator->set_current_edge(edge->uid() + "_fw");
    forward_edge(edge);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  cleanup_released(data_);
  active_built_plan_.packed_allocator->evict_unused();  // return memory back to backend allocator
  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(graph_.workspace_allocator());
  }
  return output_map;
}

TensorBundle GraphExecutor::backward(TensorBundle &output_grad_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }

  active_built_plan_.packed_allocator->set_backend_allocator(graph_.workspace_allocator());

  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }
  TensorBundle grad_input_map;

  for (auto &edge : graph_.edges()) {
    if (edge->layer())
      edge->layer()->set_workspace_allocator(active_built_plan_.packed_allocator.get());
  }

  for (Edge &edge : active_built_plan_.backward_plan.order) {
    active_built_plan_.packed_allocator->set_current_edge(edge->uid() + "_bw");
    backward_edge(edge);
    residuals_.erase(edge);
  }

  for (const auto &input : graph_.inputs()) {
    grad_input_map.set(input->uid(), grad(input));
    release_grad(input);
  }

  cleanup_released(grads_);
  active_built_plan_.packed_allocator->evict_unused();  // return memory back to backend allocator
  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(graph_.workspace_allocator());
  }
  return grad_input_map;
}

std::tuple<TensorBundle, std::map<Edge, EdgeProfile>, std::map<Node, size_t>>
GraphExecutor::profile_edges_forward(TensorBundle &input_map, bool discard_residuals) {
  std::map<Edge, EdgeProfile> edge_profiles;
  std::map<Node, size_t> node_profiles;
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
    node_profiles[it->second] = device_tensor.num_bytes();
  }
  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  // assuming sorted topologically
  for (const Edge &edge : graph_.edges()) {
    EdgeProfile profile = profile_edge_forward(edge);
    edge_profiles[edge] = profile;
    for (const Node &consumer : edge->consumers()) {
      node_profiles[consumer] = data(consumer).num_bytes();
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
    if (discard_residuals) {
      residuals_.erase(edge);
    }
  }
  cleanup_released(data_);
  return {output_map, edge_profiles, node_profiles};
}

ExecutionPlanStats GraphExecutor::profile_forward_plan(TensorBundle &input_map,
                                                       const ExecutionPlan &plan) {
  ParameterStateGuard parameter_state_guard(graph_.params(), graph_.device(),
                                            graph_.handle().get_stream());

  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }
  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  auto *allocator = graph_.workspace_allocator();
  size_t peak_usage = 0;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });

  ExecutionPlanStats stats;

  for (auto &edge : graph_.edges()) {
    if (edge->layer()) edge->layer()->set_workspace_allocator(allocator);
  }

  PackedAllocator *packed_alloc = dynamic_cast<PackedAllocator *>(allocator);

  // assuming sorted topologically
  for (const Edge &edge : plan.order) {
    if (packed_alloc) {
      packed_alloc->set_current_edge(edge->uid() + "_fw");
    }
    EdgeProfile profile = profile_edge_forward(edge);

    int64_t last_workspace_mem =
        stats.edge_stats.empty() ? 0 : stats.edge_stats.back().workspaces_mem;

    EdgeMemStats edge_stat;
    edge_stat.layer_name = edge->layer()->name();
    edge_stat.allocated_mem = allocator->allocated();
    edge_stat.reserved_mem = allocator->reserved();
    edge_stat.peak_mem = peak_usage;
    edge_stat.fragmented_mem =
        allocator->reserved() - (allocator->allocated() + allocator->unused());
    edge_stat.cached_mem = allocator->unused();
    edge_stat.activations_mem = allocator->allocated() + host_allocator_->allocated();
    edge_stat.gradients_mem = 0;
    edge_stat.host_mem = host_allocator_->allocated();
    edge_stat.workspaces_mem = std::max(last_workspace_mem, profile.workspace_mem);
    stats.edge_stats.push_back(edge_stat);

    last_workspace_mem = std::max(last_workspace_mem, profile.workspace_mem);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  allocator->remove_allocation_hook(hook_id);

  stats.peak_mem = stats.edge_stats.size() > 0 ? stats.edge_stats.back().peak_mem : 0;

  // free memory for residuals since this is just profiling
  clear_residuals();
  output_map.clear();
  cleanup_released(data_);
  allocator->evict_unused();

  return stats;
}

std::pair<TensorBundle, std::map<Edge, EdgeProfile>> GraphExecutor::profile_edges_backward(
    TensorBundle &output_grad_map, bool prefetch_residuals, const Vec<Edge> *edge_order) {
  std::map<Edge, EdgeProfile> edge_profiles;
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }
  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }
  TensorBundle grad_input_map;

  auto *allocator = graph_.workspace_allocator();

  Vec<Edge> naive_order;
  if (!edge_order) {
    naive_order.assign(graph_.edges().rbegin(), graph_.edges().rend());
    edge_order = &naive_order;
  }

  for (const Edge &edge : *edge_order) {
    if (prefetch_residuals) {
      residuals_[edge].apply_tensors([this](Tensor &tensor) {
        if (!tensor) return;
        if (tensor.device() != graph_.device()) {
          Tensor device_tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
          copy(tensor, device_tensor, graph_.handle().get_stream());
          tensor = device_tensor;
        }
      });
    }

    EdgeProfile profile = profile_edge_backward(edge);
    edge_profiles[edge] = profile;
  }

  for (const auto &input : graph_.inputs()) {
    grad_input_map.set(input->uid(), grad(input));
    release_grad(input);
  }

  cleanup_released(grads_);
  allocator->evict_unused();

  return {grad_input_map, edge_profiles};
}

ExecutionPlanStats GraphExecutor::profile_backward_plan(TensorBundle &input_map,
                                                        const ExecutionPlan &forward_plan,
                                                        const ExecutionPlan &backward_plan) {
  ParameterStateGuard parameter_state_guard(graph_.params(), graph_.device(),
                                            graph_.handle().get_stream());

  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) {
    uid_to_node[node->uid()] = node;
  }

  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Input UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_data(it->second, device_tensor, data_ref_counts_[it->second]);
  }
  TensorBundle output_map;  // placeholder to ensure outputs arent prematurely deallocated

  auto *allocator = graph_.workspace_allocator();
  tunx::PackedAllocator *packed_alloc_fw = dynamic_cast<tunx::PackedAllocator *>(allocator);
  for (auto &edge : graph_.edges()) {
    edge->layer()->set_workspace_allocator(allocator);
  }

  for (const Edge &edge : forward_plan.order) {
    if (packed_alloc_fw) {
      packed_alloc_fw->set_current_edge(edge->uid() + "_fw");
    }
    forward_edge(edge);

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        output_map.set(consumer->uid(), data(consumer));
        release_data(consumer);
      }
    }
  }

  auto &mem_pool = PoolAllocator::instance(graph_.device(), graph_.handle().get_stream());
  TensorBundle output_grad_map;
  for (const auto &[uid, tensor] : output_map) {
    output_grad_map.set(uid, Tensor(tensor.shape(), tensor.dtype(), mem_pool));
  }

  output_map.clear();

  graph_.workspace_allocator()->evict_unused();

  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it == uid_to_node.end()) {
      throw std::runtime_error("Output UID not found in graph: " + uid);
    }
    Tensor device_tensor = tensor;
    if (tensor.device() != graph_.device()) {
      device_tensor = Tensor(tensor.shape(), tensor.dtype(), *graph_.workspace_allocator());
      copy(tensor, device_tensor, graph_.handle().get_stream());
    }
    set_grad(it->second, device_tensor, grad_ref_counts_[it->second]);
  }
  output_grad_map.clear();

  TensorBundle grad_input_map;

  size_t peak_usage = 0;
  const size_t hook_id = allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });

  ExecutionPlanStats stats;

  int64_t last_workspace_mem = 0;
  int64_t last_activations_mem = allocator->allocated();
  int64_t last_gradients_mem = 0;
  int64_t param_gradients_mem = 0;
  for (const auto &param : graph_.params()) {
    if (param.grad()) {
      param_gradients_mem += param.grad().num_bytes();
    }
  }

  tunx::PackedAllocator *packed_alloc = dynamic_cast<tunx::PackedAllocator *>(allocator);

  for (const Edge &edge : backward_plan.order) {
    if (packed_alloc) {
      packed_alloc->set_current_edge(edge->uid() + "_bw");
    }

    EdgeProfile profile = profile_edge_backward(edge);

    last_gradients_mem += profile.output_mem - profile.input_mem;
    // profile.net_mem in backward = outputs - inputs - residuals_freed;
    // and activations freed = residuals_freed;
    // so net activations = -(outputs-inputs-profile.net_mem)
    last_activations_mem -= profile.output_mem - profile.input_mem - profile.net_mem;

    EdgeMemStats edge_stat;
    edge_stat.layer_name = edge->layer()->name();
    edge_stat.allocated_mem = allocator->allocated();
    edge_stat.reserved_mem = allocator->reserved();
    edge_stat.peak_mem = peak_usage;
    edge_stat.fragmented_mem =
        allocator->reserved() - (allocator->allocated() + allocator->unused());
    edge_stat.cached_mem = allocator->unused();
    edge_stat.activations_mem = last_activations_mem + host_allocator_->allocated();
    edge_stat.gradients_mem = param_gradients_mem + last_gradients_mem;
    edge_stat.host_mem = host_allocator_->allocated();
    edge_stat.workspaces_mem = std::max(last_workspace_mem, profile.workspace_mem);
    stats.edge_stats.push_back(edge_stat);

    last_workspace_mem = std::max(last_workspace_mem, profile.workspace_mem);

    for (const auto &producer : edge->producers()) {
      if (graph_.is_input(producer)) {
        grad_input_map.set(producer->uid(), grad(producer));
        release_grad(producer);
      }
    }
  }

  allocator->remove_allocation_hook(hook_id);

  stats.peak_mem = stats.edge_stats.size() > 0 ? stats.edge_stats.back().peak_mem : 0;

  grad_input_map.clear();
  cleanup_released(grads_);
  allocator->evict_unused();

  return stats;
}

void GraphExecutor::forward_edge(const Edge &edge) {
  Vec<Tensor> inputs;
  for (const auto &producer : edge->producers()) {
    try {
      inputs.push_back(data(producer));
    } catch (const std::out_of_range &) {
      throw std::runtime_error("out_of_range for producer data: " + producer->uid());
    }
    release_data(producer);
  }
  Residuals residuals;
  Vec<Tensor> output_data;
  output_data = edge->layer()->forward(inputs, residuals);

  residuals_[edge] = std::move(residuals);
  for (size_t index = 0; index < edge->consumers().size(); ++index) {
    const Node &consumer = edge->consumers()[index];
    set_data(consumer, output_data[index], data_ref_counts_[consumer]);
  }

  if (forward_hook_) {
    std::map<std::string, Tensor> consumers_snap;
    for (const auto &consumer : edge->consumers()) {
      auto it = data_.find(consumer);
      if (it != data_.end() && it->second.ref_count > 0) {
        consumers_snap[consumer->uid()] = it->second.tensor;
      }
    }
    forward_hook_(edge, consumers_snap);
  }
}

void GraphExecutor::backward_edge(const Edge &edge) {
  // Capture consumer grads before releasing — these are the gradients flowing INTO this layer
  // (PyTorch's grad_output), corresponding to d_loss/d_output of the forward layer.
  std::map<std::string, Tensor> consumers_snap;
  if (backward_grad_hook_) {
    for (const auto &consumer : edge->consumers()) {
      auto it = grads_.find(consumer);
      if (it != grads_.end() && it->second.ref_count > 0) {
        consumers_snap[consumer->uid()] = it->second.tensor;
      }
    }
  }

  Vec<Tensor> grad_outputs;
  for (const auto &consumer : edge->consumers()) {
    try {
      grad_outputs.push_back(grad(consumer));
    } catch (const std::out_of_range &) {
      throw std::runtime_error("out_of_range for consumer grad: " + consumer->uid());
    }
    release_grad(consumer);
  }
  auto residuals_it = residuals_.find(edge);
  if (residuals_it == residuals_.end()) {
    throw std::runtime_error("Residuals not found for the given graph executor");
  }
  Vec<Tensor> grad_inputs = edge->layer()->backward(grad_outputs, residuals_it->second);

  for (size_t index = 0; index < edge->producers().size(); ++index) {
    const Node &producer = edge->producers()[index];
    accumulate_grad(producer, grad_inputs[index], grad_ref_counts_[producer]);
  }

  if (backward_grad_hook_) {
    // Capture producer grads after accumulating — these are d_loss/d_input (PyTorch's grad_input).
    std::map<std::string, Tensor> producers_snap;
    for (const auto &producer : edge->producers()) {
      auto it = grads_.find(producer);
      if (it != grads_.end() && it->second.ref_count > 0) {
        producers_snap[producer->uid()] = it->second.tensor;
      }
    }
    backward_grad_hook_(edge, consumers_snap, producers_snap);
  }
}

class EdgeProfilingAllocator : public IAllocator {
public:
  EdgeProfilingAllocator(IAllocator *backend)
      : backend_(backend) {}

  void set_current_edge(const std::string &uid) {
    current_edge_uid_ = uid;
    current_alloc_index_ = 0;
  }

  dptr allocate(size_t size) override {
    if (size == 0) return dptr(nullptr);
    size_t aligned_size = (size + 255) & ~255;
    std::string iid = current_edge_uid_ + "_" + std::to_string(current_alloc_index_++);
    dptr orig = backend_->allocate(aligned_size);
    void *ptr = orig.get<void>();
    allocations_.push_back({iid, aligned_size, ptr});
    return orig;
  }

  void clear() override { backend_->clear(); }
  void ensure(size_t size) override { backend_->ensure(size); }
  size_t reserved() const override { return backend_->reserved(); }
  size_t allocated() const override { return backend_->allocated(); }
  size_t unused() const override { return backend_->unused(); }
  void evict_unused() override { backend_->evict_unused(); }
  size_t add_allocation_hook(std::function<void(size_t)> hook) override {
    return backend_->add_allocation_hook(hook);
  }
  bool remove_allocation_hook(size_t hook_id) override {
    return backend_->remove_allocation_hook(hook_id);
  }
  Device &device() const override { return backend_->device(); }

  struct AllocRecord {
    std::string iid;
    size_t size;
    void *ptr;
  };

  const std::vector<AllocRecord> &allocations() const { return allocations_; }

private:
  IAllocator *backend_;
  std::string current_edge_uid_;
  int current_alloc_index_ = 0;
  std::vector<AllocRecord> allocations_;
};

EdgeProfile GraphExecutor::profile_edge_forward(const Edge &edge) {
  // keep a copy to check cached inputs
  std::map<Node, Tensor> inputs;
  std::map<Node, Tensor> outputs;
  for (const Node &producer : edge->producers()) {
    inputs[producer] = data(producer);
  }
  auto *backend_allocator = graph_.workspace_allocator();
  EdgeProfilingAllocator tracker(backend_allocator);
  tracker.set_current_edge(edge->uid() + "_fw");

  const size_t usage_before = backend_allocator->allocated();
  size_t peak_usage = usage_before;
  const size_t hook_id = backend_allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
  auto stream = graph_.handle().get_stream();
  ParameterStateGuard parameter_state_guard(edge->layer()->params(), graph_.device(), stream);
  auto start = std::chrono::high_resolution_clock::now();

  edge->layer()->set_workspace_allocator(&tracker);
  forward_edge(edge);
  edge->layer()->set_workspace_allocator(backend_allocator);

  stream.sync();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  const size_t usage_after = backend_allocator->allocated();
  backend_allocator->remove_allocation_hook(hook_id);
  for (const Node &consumer : edge->consumers()) {
    outputs[consumer] = data(consumer);
  }
  auto &residuals = residuals_.at(edge);

  EdgeProfile profile;
  profile.exec_time = elapsed.count();
  profile.total_mem = peak_usage - usage_before;
  profile.workspace_mem = peak_usage - usage_after;
  profile.output_mem = 0;
  for (const auto &[node, tensor] : outputs) {
    profile.output_mem += tensor.num_bytes();
    for (const auto &[name, residual] : residuals.tensors()) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        profile.cached_nodes.push_back(node->uid());
        break;
      }
    }
  }

  profile.input_mem = 0;
  for (const auto &[node, tensor] : inputs) {
    profile.input_mem += tensor.num_bytes();
  }

  for (const auto &alloc : tracker.allocations()) {
    bool found = false;
    for (const auto &[node, tensor] : outputs) {
      if (tensor.data_as<void>() == alloc.ptr) {
        profile.forward_buffers.push_back(
            {alloc.iid, BufferRole::Output, alloc.size, 256, node->uid()});
        found = true;
        break;
      }
    }
    if (found) continue;

    for (const auto &[name, residual] : residuals.tensors()) {
      if (residual.data_as<void>() == alloc.ptr) {
        profile.forward_buffers.push_back(
            {alloc.iid, BufferRole::SecondaryResidual, alloc.size, 256, std::nullopt});
        profile.secondary_mem += alloc.size;
        found = true;
        break;
      }
    }
    if (found) continue;

    profile.forward_buffers.push_back(
        {alloc.iid, BufferRole::Workspace, alloc.size, 256, std::nullopt});
  }

  // Now, explicitly identify SavedInputAlias and SavedOutputAlias from residuals
  for (const auto &[name, residual] : residuals.tensors()) {
    bool matched = false;
    for (const auto &[node, tensor] : inputs) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        profile.forward_buffers.push_back(
            {"", BufferRole::SavedInputAlias, residual.num_bytes(), 256, node->uid()});
        matched = true;
        break;
      }
    }
    if (matched) continue;

    for (const auto &[node, tensor] : outputs) {
      if (tensor.data_as<void>() == residual.data_as<void>()) {
        profile.forward_buffers.push_back(
            {"", BufferRole::SavedOutputAlias, residual.num_bytes(), 256, node->uid()});
        break;
      }
    }
  }
  inputs.clear();   // free to see real net mem
  outputs.clear();  // free to see real net mem
  profile.net_mem = backend_allocator->allocated() - usage_before;

  residuals_[edge] = std::move(residuals);
  return profile;
}

EdgeProfile GraphExecutor::profile_edge_backward(const Edge &edge) {
  std::map<Node, Tensor> grad_outputs;
  std::map<Node, Tensor> grad_inputs;
  for (const auto &consumer : edge->consumers()) {
    grad_outputs[consumer] = grad(consumer);
  }

  std::map<Node, bool> is_first_to_init;
  for (const auto &producer : edge->producers()) {
    is_first_to_init[producer] = grads_.find(producer) == grads_.end();
  }

  auto *backend_allocator = graph_.workspace_allocator();
  EdgeProfilingAllocator tracker(backend_allocator);
  tracker.set_current_edge(edge->uid() + "_bw");

  const size_t usage_before = backend_allocator->allocated();
  size_t peak_usage = usage_before;
  const size_t hook_id = backend_allocator->add_allocation_hook(
      [&peak_usage](size_t usage) { peak_usage = std::max(peak_usage, usage); });
  auto stream = graph_.handle().get_stream();
  ParameterStateGuard parameter_state_guard(edge->layer()->params(), graph_.device(), stream);
  auto start = std::chrono::high_resolution_clock::now();

  edge->layer()->set_workspace_allocator(&tracker);
  backward_edge(edge);
  edge->layer()->set_workspace_allocator(backend_allocator);

  size_t usage_after = backend_allocator->allocated();
  backend_allocator->remove_allocation_hook(hook_id);
  stream.sync();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  for (const auto &producer : edge->producers()) {
    grad_inputs[producer] = grad(producer);
  }

  EdgeProfile profile;
  profile.exec_time = elapsed.count();
  profile.total_mem = peak_usage - usage_before;
  profile.workspace_mem = peak_usage - usage_after;
  profile.input_mem = 0;
  for (const auto &[node, grad_output] : grad_outputs) {
    profile.input_mem += grad_output.num_bytes();
  }
  profile.output_mem = 0;
  for (const auto &alloc : tracker.allocations()) {
    bool found = false;
    for (const auto &[node, grad_input] : grad_inputs) {
      if (grad_input.data_as<void>() == alloc.ptr) {
        if (is_first_to_init[node]) {
          profile.backward_buffers.push_back(
              {alloc.iid, BufferRole::GradientOutput, alloc.size, 256, node->uid()});
        } else {
          profile.backward_buffers.push_back(
              {alloc.iid, BufferRole::GradientContribution, alloc.size, 256, node->uid()});
        }
        found = true;
        break;
      }
    }
    if (found) continue;
    profile.backward_buffers.push_back(
        {alloc.iid, BufferRole::Workspace, alloc.size, 256, std::nullopt});
  }

  profile.secondary_mem = 0;  // none since backward
  grad_outputs.clear();       // clear to see real net memory
  grad_inputs.clear();        // clear to see real net memory
  residuals_.erase(edge);
  profile.net_mem = backend_allocator->allocated() - usage_before;
  return profile;
}

class LifetimeTrace {
public:
  void allocate(const std::string &id, size_t bytes, size_t alignment, size_t step,
                int ref_count = 1) {
    live_[id] = records_.size();
    records_.push_back({id, bytes, (int)step, -1});
    ref_counts_[id] = ref_count;
  }

  void add_ref(const std::string &id) {
    if (ref_counts_.count(id)) {
      ref_counts_[id]++;
    }
  }

  void release(const std::string &id, size_t step) {
    if (ref_counts_.count(id)) {
      if (--ref_counts_[id] == 0) {
        records_[live_[id]].end_step = step;
        live_.erase(id);
        ref_counts_.erase(id);
      }
    }
  }

  const std::vector<TensorAllocation> &records() const { return records_; }

  void release_all(size_t step) {
    for (auto it = live_.begin(); it != live_.end();) {
      records_[it->second].end_step = step;
      ref_counts_.erase(it->first);
      it = live_.erase(it);
    }
  }

private:
  std::unordered_map<std::string, size_t> live_;
  std::unordered_map<std::string, int> ref_counts_;
  std::vector<TensorAllocation> records_;
};

void GraphExecutor::pack_memory(BuiltPlan &plan, TensorBundle &input_map,
                                TensorBundle &output_grad_map) {
  std::map<std::string, Node> uid_to_node;
  for (const auto &node : graph_.nodes()) uid_to_node[node->uid()] = node;

  auto start_lifetime = std::chrono::high_resolution_clock::now();
  LifetimeTrace trace;
  size_t step = 0;

  std::map<std::string, std::string> node_to_iid;
  std::map<std::string, std::string> node_to_grad_iid;

  for (const auto &[uid, tensor] : input_map) {
    auto it = uid_to_node.find(uid);
    if (it != uid_to_node.end()) {
      trace.allocate(uid, tensor.num_bytes(), 256, step, data_ref_counts_[it->second]);
      node_to_iid[uid] = uid;
    }
  }

  for (const Edge &edge : plan.forward_plan.order) {
    const EdgeProfile &profile = plan.forward_edge_profiles.at(edge);
    step++;

    for (const auto &buf : profile.forward_buffers) {
      if (buf.role == BufferRole::Output) {
        auto node = uid_to_node.at(buf.alias_node_uid.value());
        node_to_iid[node->uid()] = buf.local_id;
        trace.allocate(buf.local_id, buf.bytes, buf.alignment, step, data_ref_counts_[node]);
      } else if (buf.role == BufferRole::SecondaryResidual) {
        trace.allocate(buf.local_id, buf.bytes, buf.alignment, step, 1);
      } else if (buf.role == BufferRole::SavedInputAlias ||
                 buf.role == BufferRole::SavedOutputAlias) {
        trace.add_ref(node_to_iid[buf.alias_node_uid.value()]);
      } else if (buf.role == BufferRole::Workspace) {
        trace.allocate(buf.local_id, buf.bytes, buf.alignment, step, 1);
        trace.release(buf.local_id, step);
      }
    }

    for (const auto &producer : edge->producers()) {
      trace.release(node_to_iid[producer->uid()], step);
    }

    for (const Node &consumer : edge->consumers()) {
      if (graph_.is_output(consumer)) {
        trace.add_ref(node_to_iid[consumer->uid()]);  // Held by output_map
      }
    }
  }

  for (const auto &[uid, tensor] : output_grad_map) {
    auto it = uid_to_node.find(uid);
    if (it != uid_to_node.end() && tensor) {
      trace.allocate(uid + "/grad", tensor.num_bytes(), 256, step, grad_ref_counts_[it->second]);
      node_to_grad_iid[uid] = uid + "/grad";
    }
  }

  if (plan.backward_plan.order.size() > 0) {
    for (const Edge &edge : plan.backward_plan.order) {
      const EdgeProfile &profile = plan.backward_edge_profiles.at(edge);
      step++;

      for (const auto &buf : profile.backward_buffers) {
        if (buf.role == BufferRole::GradientOutput) {
          std::string node_uid = buf.alias_node_uid.value();
          auto node = uid_to_node.at(node_uid);
          node_to_grad_iid[node_uid] = buf.local_id;
          trace.allocate(buf.local_id, buf.bytes, buf.alignment, step, grad_ref_counts_[node]);
        } else if (buf.role == BufferRole::GradientContribution) {
          trace.allocate(buf.local_id, buf.bytes, buf.alignment, step, 1);
          trace.release(buf.local_id, step);
        } else if (buf.role == BufferRole::Workspace) {
          trace.allocate(buf.local_id, buf.bytes, buf.alignment, step, 1);
          trace.release(buf.local_id, step);
        }
      }

      for (const auto &consumer : edge->consumers()) {
        trace.release(node_to_grad_iid[consumer->uid()], step);
      }

      const EdgeProfile &fwd_profile = plan.forward_edge_profiles.at(edge);
      for (const auto &buf : fwd_profile.forward_buffers) {
        if (buf.role == BufferRole::SecondaryResidual) {
          trace.release(buf.local_id, step);
        } else if (buf.role == BufferRole::SavedInputAlias ||
                   buf.role == BufferRole::SavedOutputAlias) {
          trace.release(node_to_iid[buf.alias_node_uid.value()], step);
        }
      }

      for (const auto &producer : edge->producers()) {
        if (graph_.is_input(producer)) {
          trace.add_ref(node_to_grad_iid[producer->uid()]);  // Held by grad_input_map
        }
      }
    }
  }

  trace.release_all(step + 1);
  auto end_lifetime = std::chrono::high_resolution_clock::now();
  plan.lifetime_time_ms =
      std::chrono::duration<double, std::milli>(end_lifetime - start_lifetime).count();

  auto start_packing = std::chrono::high_resolution_clock::now();
  auto pack = MemoryPacker::pack(trace.records());
  if (os_) {
    *os_ << "Memory Packer: Peak memory packed size = " << pack.peak_memory << " bytes\n";
  }
  plan.packed_allocator = PackedAllocator::create(pack.peak_memory, pack.offsets);
  plan.packed_allocator->set_backend_allocator(graph_.workspace_allocator());
  auto end_packing = std::chrono::high_resolution_clock::now();
  plan.packing_time_ms =
      std::chrono::duration<double, std::milli>(end_packing - start_packing).count();
}

}  // namespace tunx