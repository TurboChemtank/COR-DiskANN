// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <sstream>
#include <typeinfo>
#include <unordered_map>

#include "omp.h"
#include "defaults.h"

namespace diskann
{

class IndexWriteParameters

{
  public:
    const uint32_t search_list_size; // L
    const uint32_t max_degree;       // R
    const bool saturate_graph;
    const uint32_t max_occlusion_size; // C
    const float alpha;
    const uint32_t num_threads;
    const uint32_t filter_list_size; // Lf
    // 【新增参数】查询时扩展相关标签的数量（K），默认0表示不扩展
    const uint32_t num_correlated_labels_to_expand;
    // 兼容保留参数：旧版用于低维投影，当前实现已忽略该值
    const uint32_t label_projection_dim;
    // 标签元数据懒更新触发阈值：达到 max(nd*ratio, min_ops) 时批量刷新
    const float filter_lazy_update_ratio;
    const uint32_t filter_lazy_update_min_ops;

    // 【修改构造函数 - 中文说明】移除 beta 相关参数（相关度改为剪枝后统计，无法用于建图阶段）
    IndexWriteParameters(const uint32_t search_list_size, const uint32_t max_degree, const bool saturate_graph,
                         const uint32_t max_occlusion_size, const float alpha, const uint32_t num_threads,
                         const uint32_t filter_list_size, const uint32_t num_correlated_labels_to_expand,
                         const uint32_t label_projection_dim,
                         const float filter_lazy_update_ratio, const uint32_t filter_lazy_update_min_ops)
        : search_list_size(search_list_size), max_degree(max_degree), saturate_graph(saturate_graph),
          max_occlusion_size(max_occlusion_size), alpha(alpha), num_threads(num_threads),
          filter_list_size(filter_list_size), num_correlated_labels_to_expand(num_correlated_labels_to_expand),
          label_projection_dim(label_projection_dim),
          filter_lazy_update_ratio(filter_lazy_update_ratio),
          filter_lazy_update_min_ops(filter_lazy_update_min_ops)
    {
    }

    friend class IndexWriteParametersBuilder;
};

class IndexSearchParams
{
  public:
    IndexSearchParams(const uint32_t initial_search_list_size, const uint32_t num_search_threads)
        : initial_search_list_size(initial_search_list_size), num_search_threads(num_search_threads)
    {
    }
    const uint32_t initial_search_list_size; // search L
    const uint32_t num_search_threads;       // search threads
};

class IndexWriteParametersBuilder
{
    /**
     * Fluent builder pattern to keep track of the 7 non-default properties
     * and their order. The basic ctor was getting unwieldy.
     */
  public:
    IndexWriteParametersBuilder(const uint32_t search_list_size, // L
                                const uint32_t max_degree        // R
                                )
        : _search_list_size(search_list_size), _max_degree(max_degree)
    {
    }

    IndexWriteParametersBuilder &with_max_occlusion_size(const uint32_t max_occlusion_size)
    {
        _max_occlusion_size = max_occlusion_size;
        return *this;
    }

    IndexWriteParametersBuilder &with_saturate_graph(const bool saturate_graph)
    {
        _saturate_graph = saturate_graph;
        return *this;
    }

    IndexWriteParametersBuilder &with_alpha(const float alpha)
    {
        _alpha = alpha;
        return *this;
    }

    IndexWriteParametersBuilder &with_num_threads(const uint32_t num_threads)
    {
        _num_threads = num_threads == 0 ? omp_get_num_procs() : num_threads;
        return *this;
    }

    IndexWriteParametersBuilder &with_filter_list_size(const uint32_t filter_list_size)
    {
        _filter_list_size = filter_list_size == 0 ? _search_list_size : filter_list_size;
        return *this;
    }

    // 【新增Builder方法】设置查询时标签扩展的K
    IndexWriteParametersBuilder &with_num_correlated_labels_to_expand(const uint32_t k)
    {
        _num_correlated_labels_to_expand = k;
        return *this;
    }

    // 兼容保留接口：旧版用于低维投影，当前实现仅透传该值而不再使用
    IndexWriteParametersBuilder &with_label_projection_dim(const uint32_t dim)
    {
        _label_projection_dim = dim;
        return *this;
    }

    // 设置标签元数据懒更新阈值比例（建议 0~1）
    IndexWriteParametersBuilder &with_filter_lazy_update_ratio(const float ratio)
    {
        _filter_lazy_update_ratio = ratio;
        return *this;
    }

    // 设置标签元数据懒更新阈值下限
    IndexWriteParametersBuilder &with_filter_lazy_update_min_ops(const uint32_t min_ops)
    {
        _filter_lazy_update_min_ops = min_ops;
        return *this;
    }

    IndexWriteParameters build() const
    {
        // 【修改build - 中文说明】移除 beta 相关参数
        return IndexWriteParameters(_search_list_size, _max_degree, _saturate_graph, _max_occlusion_size, _alpha,
                                    _num_threads, _filter_list_size, _num_correlated_labels_to_expand,
                                    _label_projection_dim,
                                    _filter_lazy_update_ratio, _filter_lazy_update_min_ops);
    }

    IndexWriteParametersBuilder(const IndexWriteParameters &wp)
        : _search_list_size(wp.search_list_size), _max_degree(wp.max_degree),
          _max_occlusion_size(wp.max_occlusion_size), _saturate_graph(wp.saturate_graph), _alpha(wp.alpha),
          _filter_list_size(wp.filter_list_size), _num_correlated_labels_to_expand(wp.num_correlated_labels_to_expand),
          _label_projection_dim(wp.label_projection_dim),
          _filter_lazy_update_ratio(wp.filter_lazy_update_ratio),
          _filter_lazy_update_min_ops(wp.filter_lazy_update_min_ops)
    {
    }
    IndexWriteParametersBuilder(const IndexWriteParametersBuilder &) = delete;
    IndexWriteParametersBuilder &operator=(const IndexWriteParametersBuilder &) = delete;

  private:
    uint32_t _search_list_size{};
    uint32_t _max_degree{};
    uint32_t _max_occlusion_size{defaults::MAX_OCCLUSION_SIZE};
    bool _saturate_graph{defaults::SATURATE_GRAPH};
    float _alpha{defaults::ALPHA};
    uint32_t _num_threads{defaults::NUM_THREADS};
    uint32_t _filter_list_size{defaults::FILTER_LIST_SIZE};
    uint32_t _num_correlated_labels_to_expand{0};
    uint32_t _label_projection_dim{32};
    float _filter_lazy_update_ratio{0.01f};
    uint32_t _filter_lazy_update_min_ops{128};
};

} // namespace diskann
