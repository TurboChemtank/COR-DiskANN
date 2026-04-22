// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <omp.h>
#include <queue>
#include <sstream>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "distance.h"
#include "index.h"
#include "index_factory.h"
#include "memory_mapper.h"
#include "defaults.h"
#include "parameters.h"
#include "program_options_utils.hpp"
#include "utils.h"

namespace po = boost::program_options;

// 中文说明：逐行解析查询标签文件；每行是一个查询的 OR 标签集合，逗号分隔。
std::vector<std::vector<std::string>> parse_query_filters_file(const std::string &filename)
{
    std::vector<std::vector<std::string>> query_filters;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw diskann::ANNException("Failed to open query filters file", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<std::string> labels;
        std::stringstream ss(line);
        std::string label;
        while (std::getline(ss, label, ','))
        {
            label.erase(0, label.find_first_not_of(" \t\n\r\f\v"));
            label.erase(label.find_last_not_of(" \t\n\r\f\v") + 1);
            if (!label.empty())
            {
                labels.push_back(label);
            }
        }
        query_filters.push_back(labels);
    }
    file.close();
    return query_filters;
}

struct SearchRunStats
{
    double displayed_qps = 0.0;
    float avg_cmps = 0.0f;
    float mean_latency = 0.0f;
    float p999_latency = 0.0f;
    bool valid = false;
};

inline void save_groundtruth_as_one_file(const std::string &filename, uint32_t *data, float *distances, size_t npts,
                                         size_t ndims)
{
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    int npts_i32 = (int)npts, ndims_i32 = (int)ndims;
    writer.write((char *)&npts_i32, sizeof(int));
    writer.write((char *)&ndims_i32, sizeof(int));
    std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, npts*dim dist-matrix) with npts = "
              << npts << ", dim = " << ndims << std::endl;
    writer.write((char *)data, npts * ndims * sizeof(uint32_t));
    writer.write((char *)distances, npts * ndims * sizeof(float));
    writer.close();
    std::cout << "Finished writing truthset to " << filename << std::endl;
}

template <typename T>
void build_dynamic_groundtruth(diskann::AbstractIndex &index, diskann::Metric metric, const T *query, size_t query_num,
                               size_t query_dim, size_t query_aligned_dim,
                               const std::vector<std::vector<std::string>> &query_label_sets, uint32_t recall_at,
                               uint32_t num_threads, std::vector<uint32_t> &gt_ids,
                               std::vector<float> &gt_dists)
{
    using TagT = uint32_t;
    tsl::robin_set<TagT> active_tag_set;
    index.get_active_tags(active_tag_set);
    std::vector<TagT> active_tags(active_tag_set.begin(), active_tag_set.end());
    std::sort(active_tags.begin(), active_tags.end());

    std::cout << "Building dynamic ground truth from current active index..." << std::endl;
    std::cout << "Active tags: " << active_tags.size() << std::endl;

    std::vector<T> active_vectors(active_tags.size() * query_dim);
    for (size_t i = 0; i < active_tags.size(); i++)
    {
        TagT tag = active_tags[i];
        if (index.get_vector_by_tag(tag, active_vectors.data() + i * query_dim) != 0)
        {
            throw diskann::ANNException("Failed to fetch vector by tag while building dynamic ground truth", -1);
        }
    }

    std::unique_ptr<diskann::Distance<T>> distance_fn(diskann::get_distance_function<T>(metric));
    gt_ids.assign(query_num * recall_at, 0U);
    gt_dists.assign(query_num * recall_at, std::numeric_limits<float>::infinity());

    omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t query_idx = 0; query_idx < (int64_t)query_num; query_idx++)
    {
        using Candidate = std::pair<float, uint32_t>;
        std::priority_queue<Candidate> topk;
        const T *query_vec = query + query_idx * query_aligned_dim;
        const auto &query_labels = query_label_sets[(size_t)query_idx];

        for (size_t active_idx = 0; active_idx < active_tags.size(); active_idx++)
        {
            const TagT tag = active_tags[active_idx];
            if (!index.matches_any_labels(tag, query_labels))
            {
                continue;
            }

            const float dist = distance_fn->compare(query_vec, active_vectors.data() + active_idx * query_dim,
                                                    (uint32_t)query_dim);
            if (topk.size() < recall_at)
            {
                topk.emplace(dist, static_cast<uint32_t>(tag));
            }
            else if (dist < topk.top().first)
            {
                topk.pop();
                topk.emplace(dist, static_cast<uint32_t>(tag));
            }
        }

        std::vector<Candidate> ordered;
        ordered.reserve(topk.size());
        while (!topk.empty())
        {
            ordered.push_back(topk.top());
            topk.pop();
        }
        std::sort(ordered.begin(), ordered.end(),
                  [](const Candidate &lhs, const Candidate &rhs) { return lhs.first < rhs.first; });

        for (size_t i = 0; i < ordered.size(); i++)
        {
            gt_ids[(size_t)query_idx * recall_at + i] = ordered[i].second;
            gt_dists[(size_t)query_idx * recall_at + i] = ordered[i].first;
        }
    }

    std::cout << "Dynamic ground truth built." << std::endl;
}

// 中文说明：把一次子搜索结果 merge 到总结果里；若同一 id 出现多次，只保留更优距离。
inline void merge_partial_results(const uint32_t *ids, const float *dists, size_t k, diskann::Metric metric,
                                  std::unordered_map<uint32_t, float> &best_dist_by_id)
{
    for (size_t i = 0; i < k; i++)
    {
        if (!std::isfinite(dists[i]))
        {
            continue;
        }

        const float sortable_dist = metric == diskann::Metric::INNER_PRODUCT ? -dists[i] : dists[i];
        auto it = best_dist_by_id.find(ids[i]);
        if (it == best_dist_by_id.end() || sortable_dist < it->second)
        {
            best_dist_by_id[ids[i]] = sortable_dist;
        }
    }
}

// 中文说明：把 merge 完的候选按距离排序后写回最终 top-k。
inline void write_merged_topk(const std::unordered_map<uint32_t, float> &best_dist_by_id, size_t k,
                              diskann::Metric metric, uint32_t *out_ids, float *out_dists)
{
    std::vector<std::pair<float, uint32_t>> ordered;
    ordered.reserve(best_dist_by_id.size());
    for (const auto &kv : best_dist_by_id)
    {
        ordered.emplace_back(kv.second, kv.first);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second); });

    size_t pos = 0;
    for (; pos < k && pos < ordered.size(); pos++)
    {
        out_ids[pos] = ordered[pos].second;
        out_dists[pos] = metric == diskann::Metric::INNER_PRODUCT ? -ordered[pos].first : ordered[pos].first;
    }
    for (; pos < k; pos++)
    {
        out_ids[pos] = 0;
        out_dists[pos] = std::numeric_limits<float>::infinity();
    }
}

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::string &query_filters_file, const float fail_if_recall_below,
                        const uint32_t expand_labels_k, const uint32_t label_projection_dim)
{
    using TagT = uint32_t;
    using ConcreteIndex = diskann::Index<T, TagT, LabelT>;

    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    std::vector<uint32_t> dynamic_gt_ids;
    std::vector<float> dynamic_gt_dists;
    bool need_dynamic_groundtruth = false;
    std::string dynamic_gt_output_file;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
        diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
        diskann::cout << "Ground truth file loaded. Recall will be calculated." << std::endl;
    }
    else if (dynamic && tags)
    {
        calc_recall_flag = true;
        gt_num = query_num;
        gt_dim = recall_at;
        need_dynamic_groundtruth = true;
        dynamic_gt_output_file = result_path_prefix + "_dynamic_gt.bin";
        if (truthset_file != std::string("null"))
        {
            diskann::cout << "Truthset file " << truthset_file
                          << " not found. Will build brute-force ground truth from current active dynamic index."
                          << std::endl;
        }
        else
        {
            diskann::cout << "Dynamic+tags mode without gt_file: will build brute-force ground truth from current "
                             "active index."
                          << std::endl;
        }
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

    auto query_label_sets = parse_query_filters_file(query_filters_file);
    if (query_label_sets.size() != query_num)
    {
        throw diskann::ANNException("Number of queries in query file and query filters file do not match", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
    }
    std::cout << "Query filters file loaded." << std::endl;

    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);
    auto write_params =
        diskann::IndexWriteParametersBuilder(diskann::defaults::SEARCH_LIST_SIZE, diskann::defaults::MAX_DEGREE)
            .with_num_threads(num_threads)
            .with_filter_list_size(diskann::defaults::FILTER_LIST_SIZE)
            .with_num_correlated_labels_to_expand(expand_labels_k)
            .with_label_projection_dim(label_projection_dim)
            .build();

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .with_index_write_params(write_params)
                      .is_dynamic_index(dynamic)
                      .is_enable_tags(tags)
                      .is_concurrent_consolidate(false)
                      .is_pq_dist_build(false)
                      .is_use_opq(false)
                      .is_filtered(false)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;
    index->set_expand_labels_k(expand_labels_k);

    auto *typed_index = dynamic_cast<ConcreteIndex *>(index.get());
    if (typed_index == nullptr)
    {
        throw diskann::ANNException("Failed to cast abstract index to concrete memory index", -1, __FUNCSIG__,
                                    __FILE__, __LINE__);
    }

    if (metric == diskann::FAST_L2)
    {
        index->optimize_index_layout();
    }

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
              << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
    table_width += 4 + 12 + 18 + 20 + 15;

    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats(query_num, 0);
    std::vector<SearchRunStats> run_stats(Lvec.size());

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        const uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            const T *query_vec = query + i * query_aligned_dim;
            std::vector<std::string> current_query_labels = query_label_sets[(size_t)i];
            std::sort(current_query_labels.begin(), current_query_labels.end());
            current_query_labels.erase(std::unique(current_query_labels.begin(), current_query_labels.end()),
                                       current_query_labels.end());

            const uint32_t threshold = index->get_filter_frequency_threshold();
            const double low_cutoff = static_cast<double>(threshold) * 0.01;
            std::vector<std::string> high_freq_labels;
            std::vector<std::string> mid_freq_labels;
            std::vector<std::string> low_freq_labels;
            high_freq_labels.reserve(current_query_labels.size());
            mid_freq_labels.reserve(current_query_labels.size());
            low_freq_labels.reserve(current_query_labels.size());

            for (const auto &label : current_query_labels)
            {
                const uint32_t freq = index->get_filter_frequency(label);
                if (freq > threshold)
                {
                    high_freq_labels.emplace_back(label);
                }
                else if (static_cast<double>(freq) > low_cutoff)
                {
                    mid_freq_labels.emplace_back(label);
                }
                else
                {
                    low_freq_labels.emplace_back(label);
                }
            }

            std::unordered_map<uint32_t, float> merged_best;
            merged_best.reserve(std::max<size_t>(1, current_query_labels.size() * recall_at));
            std::vector<uint32_t> temp_ids(recall_at, 0);
            std::vector<TagT> temp_tags(recall_at, TagT{});
            std::vector<float> temp_dists(recall_at, std::numeric_limits<float>::infinity());
            std::vector<T *> temp_res_vectors;
            uint32_t current_cmp_stats = 0;

            auto merge_ids = [&]() {
                merge_partial_results(temp_ids.data(), temp_dists.data(), recall_at, metric, merged_best);
            };
            auto merge_tags = [&]() {
                merge_partial_results(temp_tags.data(), temp_dists.data(), recall_at, metric, merged_best);
            };

            // 中文说明：高频标签单独做一次普通过滤搜索，不做标签扩展。
            for (const auto &label : high_freq_labels)
            {
                try
                {
                    if (dynamic && tags)
                    {
                        auto retval = typed_index->search_with_filter_label_group_tags(
                            query_vec, recall_at, L, temp_tags.data(), temp_dists.data(), temp_res_vectors, {label}, 0,
                            true);
                        current_cmp_stats += retval.second;
                        merge_tags();
                    }
                    else
                    {
                        auto retval = typed_index->search_with_filter_label_group(
                            query_vec, {label}, recall_at, L, temp_ids.data(), temp_dists.data(), 0, true);
                        current_cmp_stats += retval.second;
                        merge_ids();
                    }
                }
                catch (const std::exception &)
                {
                    // 中文说明：若常规过滤搜索因为缺少 medoid 等原因失败，则回退到倒排爆搜保证可用性。
                    if (dynamic && tags)
                    {
                        auto retval = typed_index->brute_force_search_filter_label_tags(
                            query_vec, label, recall_at, temp_tags.data(), temp_dists.data(), temp_res_vectors);
                        current_cmp_stats += retval.second;
                        merge_tags();
                    }
                    else
                    {
                        auto retval = typed_index->brute_force_search_filter_label(
                            query_vec, label, recall_at, temp_ids.data(), temp_dists.data());
                        current_cmp_stats += retval.second;
                        merge_ids();
                    }
                }
            }

            // 中文说明：中频标签合并成一次 OR 搜索，起点只放各标签 medoid，并允许相关标签扩展。
            if (!mid_freq_labels.empty())
            {
                try
                {
                    if (dynamic && tags)
                    {
                        auto retval = typed_index->search_with_filter_label_group_tags(
                            query_vec, recall_at, L, temp_tags.data(), temp_dists.data(), temp_res_vectors,
                            mid_freq_labels, (int)expand_labels_k, false);
                        current_cmp_stats += retval.second;
                        merge_tags();
                    }
                    else
                    {
                        auto retval = typed_index->search_with_filter_label_group(
                            query_vec, mid_freq_labels, recall_at, L, temp_ids.data(), temp_dists.data(),
                            (int)expand_labels_k, false);
                        current_cmp_stats += retval.second;
                        merge_ids();
                    }
                }
                catch (const std::exception &)
                {
                    // 中文说明：合并搜索失败时，退化成逐标签倒排爆搜，保证结果仍然完整。
                    for (const auto &label : mid_freq_labels)
                    {
                        if (dynamic && tags)
                        {
                            auto retval = typed_index->brute_force_search_filter_label_tags(
                                query_vec, label, recall_at, temp_tags.data(), temp_dists.data(), temp_res_vectors);
                            current_cmp_stats += retval.second;
                            merge_tags();
                        }
                        else
                        {
                            auto retval = typed_index->brute_force_search_filter_label(
                                query_vec, label, recall_at, temp_ids.data(), temp_dists.data());
                            current_cmp_stats += retval.second;
                            merge_ids();
                        }
                    }
                }
            }

            // 中文说明：超低频标签直接走倒排索引，把 posting 全部拿出来爆搜。
            for (const auto &label : low_freq_labels)
            {
                if (dynamic && tags)
                {
                    auto retval = typed_index->brute_force_search_filter_label_tags(
                        query_vec, label, recall_at, temp_tags.data(), temp_dists.data(), temp_res_vectors);
                    current_cmp_stats += retval.second;
                    merge_tags();
                }
                else
                {
                    auto retval =
                        typed_index->brute_force_search_filter_label(query_vec, label, recall_at, temp_ids.data(),
                                                                     temp_dists.data());
                    current_cmp_stats += retval.second;
                    merge_ids();
                }
            }

            write_merged_topk(merged_best, recall_at, metric, query_result_ids[test_id].data() + i * recall_at,
                              query_result_dists[test_id].data() + i * recall_at);
            cmp_stats[i] = current_cmp_stats;

            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[(size_t)i] = (float)(diff.count() * 1000000);
        }

        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;
        double displayed_qps = query_num / diff.count();
        if (show_qps_per_thread)
        {
            displayed_qps /= num_threads;
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        const double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<double>(query_num);
        const float avg_cmps = static_cast<float>(std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0ULL) /
                                                  static_cast<double>(query_num));

        run_stats[test_id].displayed_qps = displayed_qps;
        run_stats[test_id].avg_cmps = avg_cmps;
        run_stats[test_id].mean_latency = (float)mean_latency;
        run_stats[test_id].p999_latency = latency_stats[(uint64_t)(0.999 * query_num)];
        run_stats[test_id].valid = true;
    }

    if (need_dynamic_groundtruth)
    {
        build_dynamic_groundtruth<T>(*index, metric, query, query_num, query_dim, query_aligned_dim, query_label_sets,
                                     recall_at, num_threads, dynamic_gt_ids, dynamic_gt_dists);
        gt_ids = dynamic_gt_ids.data();
        gt_dists = dynamic_gt_dists.data();
        save_groundtruth_as_one_file(dynamic_gt_output_file, gt_ids, gt_dists, query_num, recall_at);
    }

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        const uint32_t L = Lvec[test_id];
        if (L < recall_at || !run_stats[test_id].valid)
        {
            continue;
        }

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::cout << std::setw(4) << L << std::setw(12) << run_stats[test_id].displayed_qps << std::setw(18)
                  << run_stats[test_id].avg_cmps << std::setw(20) << run_stats[test_id].mean_latency << std::setw(15)
                  << run_stats[test_id].p999_latency;
        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(best_recall, recall);
        }
        std::cout << std::endl;
    }

    std::cout << "Done searching. Now saving results " << std::endl;
    for (size_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        const auto L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        const std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);
        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);
    }

    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, label_type,
        query_filters_file;
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;
    uint32_t expand_labels_k = 0;
    uint32_t label_projection_dim = 32;

    po::options_description desc{
        program_options_utils::make_program_description("test_or", "Searches in-memory DiskANN indexes with new OR logic")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);
        required_configs.add_options()(
            "query_filters_file", po::value<std::string>(&query_filters_file)->required(),
            "Path to query filters file. Each line contains comma-separated labels for the corresponding query.");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()(
            "dynamic", po::value<bool>(&dynamic)->default_value(false),
            "Whether the index is dynamic. Dynamic indices must have associated tags. Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("expand_labels_k", po::value<uint32_t>(&expand_labels_k)->default_value(0),
                                       "Expand to Top-K correlated labels at query time (default 0)");
        optional_configs.add_options()("label_projection_dim",
                                       po::value<uint32_t>(&label_projection_dim)->default_value(32),
                                       "Deprecated compatibility option. Kept for CLI compatibility but ignored.");

        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in the output table");

        desc.add(required_configs).add(optional_configs).add(output_controls);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/cosine are supported in general, and "
                     "mips/fast_l2 only for floating point data."
                  << std::endl;
        return -1;
    }

    if (dynamic && !tags)
    {
        std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
        return -1;
    }

    if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0)
    {
        std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
        return -1;
    }

    try
    {
        if (label_type == "ushort")
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters_file, fail_if_recall_below,
                    expand_labels_k, label_projection_dim);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters_file, fail_if_recall_below,
                    expand_labels_k, label_projection_dim);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters_file, fail_if_recall_below,
                    expand_labels_k, label_projection_dim);
            }
        }
        else
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                   num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                   show_qps_per_thread, query_filters_file, fail_if_recall_below,
                                                   expand_labels_k, label_projection_dim);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, query_filters_file, fail_if_recall_below,
                                                    expand_labels_k, label_projection_dim);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, query_filters_file, fail_if_recall_below,
                                                  expand_labels_k, label_projection_dim);
            }
        }

        std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
        return -1;
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
