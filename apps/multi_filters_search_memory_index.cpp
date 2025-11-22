// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <sstream>
#include <vector>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"
#include "parameters.h"
#include "defaults.h"

namespace po = boost::program_options;

/*
 * 文件目的（工具概述）
 * -----------------------------------------------------------------------------
 * 该可执行程序用于“多标签 OR 逻辑”的过滤检索评测：对于每个查询，读取其标签集合
 * （每行逗号分隔），对集合中的每一个标签分别调用一次单标签过滤检索
 *   index->search_with_filters(query, single_label, K, L, ids, dists)
 * 然后把所有单标签的结果做并集（聚合、去重、按距离排序），再取前 K 输出。
 *
 * 与单标签版本的主要差异：
 * - 单标签版本：每个查询只执行一次 search_with_filters。
 * - 本工具：每个查询对“标签集合”中的每个标签各执行一次，再聚合。
 *
 * 注意与风险点（导致崩溃/异常的常见原因）
 * -----------------------------------------------------------------------------
 * 1) search_with_filters 内部若找不到该标签对应的 medoid（_label_to_start_id 缺失），
 *    会抛出 ANNException（如“No filtered medoid found...”）；在 OpenMP 线程内未捕获会导致
 *    terminate called recursively → 程序中止。解决：在内层单标签调用处加 try/catch 并跳过。
 *
 * 2) 查询标签与索引标签映射不一致：
 *    - 若 query_filters_file 中的某些标签字符串在构建索引时从未出现，则索引侧没有映射/medoid，
 *      该标签的检索应当跳过并不影响其他标签的 OR 聚合。
 *
 * 3) MIPS 路径下的距离符号：
 *    - 内积越大越好，代码中统一用“越小越好”的距离模型，于是以 -dot 作为排序距离；
 *      输出（或聚合）时要注意还原/转换的语义（本文件中对 MIPS 用负号做了统一处理）。
 *
 * 4) 召回评测的 ground truth
 *    - 必须使用“多标签 OR 逻辑”的真值（compute_groundtruth_for_multi_filters 生成）；
 *      若错用单标签真值，会导致评测结果不对（Recall 计算不匹配）。
 */
// 新增：辅助函数，用于解析逗号分隔的多标签文件
// (与 compute_groundtruth_for_multi_filters.cpp 中的函数相同)
std::vector<std::vector<std::string>> parse_query_filters_file(const std::string &filename)
{
    // 逐行读取，每行表示一个查询的标签集合；每行中逗号分隔多个标签字符串
    // 例如：label_A,label_B,label_C
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

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::string &query_filters_file, // 修改：接收文件路径
                        const float fail_if_recall_below, const uint32_t expand_labels_k,
                        const bool use_label_correlation, const float beta_strength)
{
    using TagT = uint32_t;
    using IdType = uint32_t; // 内存索引通常使用 uint32_t 作为ID

    // 加载查询文件
    // - load_aligned_bin 会把查询向量按对齐维度加载到对齐内存，便于后续 SIMD 距离计算
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    // 加载真值文件
    // - 若提供了“OR 真值”文件，将在末尾打印 Recall 指标
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
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

    // --- 加载索引 ---
    // - 读取图头，确定冻点数量等；随后 load() 载入数据与图
    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

    // 写入参数：用于启用 β 与查询时标签扩展K
    auto write_params =
        diskann::IndexWriteParametersBuilder(diskann::defaults::SEARCH_LIST_SIZE, diskann::defaults::MAX_DEGREE)
            .with_num_threads(num_threads)
            .with_filter_list_size(diskann::defaults::FILTER_LIST_SIZE)
            .with_use_label_correlation(use_label_correlation)
            .with_beta_strength(beta_strength)
            .with_num_correlated_labels_to_expand(expand_labels_k)
            .build();

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0) // 在加载时通常设为0
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
                      .is_filtered(true)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;

    // 设置查询阶段标签扩展K（可覆盖构造时值）
    index->set_expand_labels_k(expand_labels_k);

    // --- 加载查询标签（字符串），不在此处转换为 LabelT ---
    // - 这里直接把字符串作为 raw_label 传入 search_with_filters；
    //   内部会调用索引的标签映射，若找不到映射/medoid 会抛异常（需要上层注意处理）
    auto query_label_sets = parse_query_filters_file(query_filters_file);
    if (query_label_sets.size() != query_num)
    {
        throw diskann::ANNException("Number of queries in query file and query filters file do not match", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
    }
    std::cout << "Query filters file loaded." << std::endl;

    // ... (优化的索引布局逻辑保持不变)
    if (metric == diskann::FAST_L2)
        index->optimize_index_layout();

    // --- 准备输出 ---
    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    // (省略了 'tags' 的特殊表格格式，专注于过滤搜索)
    std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps" << std::setw(20)
              << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
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

    double best_recall = 0.0;

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);

        // --- 核心修改：并行搜索循环 ---
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            // 每个查询独立：对其“多标签集合”逐个执行单标签检索，并聚合结果
            auto qs = std::chrono::high_resolution_clock::now(); // 开始计时

            const T *query_vec = query + i * query_aligned_dim;
            const auto &current_query_labels = query_label_sets[i];
            uint32_t current_cmp_stats = 0;

            // 1. 创建聚合池 (自动排序和去重)
            std::set<diskann::Neighbor> aggregated_results;

            // 2. 创建线程本地的临时缓冲区
            std::vector<IdType> temp_ids(recall_at);
            std::vector<float> temp_dists(recall_at);

            // 3. 执行“OR”查询（新内循环）
            for (const auto &single_label_str : current_query_labels)
            {
                // 3a. 执行单标签搜索
                // 显式将查询向量封装为 std::any，避免 any_cast 类型不匹配导致的异常；
                // 同时用 try/catch 防御 OpenMP 线程内异常导致的进程终止。
                try
                {
                    std::any any_query = static_cast<const T *>(query_vec);
                    auto retval = index->search_with_filters(any_query, single_label_str, recall_at, L, temp_ids.data(),
                                                             temp_dists.data());
                    current_cmp_stats += retval.second; // 累加比较次数
                }
                catch (const std::bad_any_cast &)
                {
                    // 该标签检索失败（类型擦除传递失败），跳过本标签
                    continue;
                }
                catch (const std::exception &)
                {
                    // 包含：标签无 medoid（No filtered medoid found...）等场景，均跳过该标签
                    continue;
                }

                // 3b. 聚合结果
                for (size_t j = 0; j < recall_at; ++j)
                {
                    if (temp_dists[j] != std::numeric_limits<float>::infinity())
                    {
                        // 使用 MIPS 调整后的距离（如果适用）进行排序
                        float dist_for_comp =
                            (metric == diskann::Metric::INNER_PRODUCT) ? -temp_dists[j] : temp_dists[j];
                        aggregated_results.insert(diskann::Neighbor(temp_ids[j], dist_for_comp));
                    }
                }
            }

            // 4. 提取最终 Top-K
            // - set<Neighbor> 默认按 distance 从小到大排序（见 Neighbor::operator< 实现）
            // - 这里顺序遍历即得到最小的 K 个；MIPS 已在插入时做了负号处理，回写时再还原
            size_t pos = 0;
            for (const auto &neighbor : aggregated_results)
            {
                if (pos >= recall_at) // 只需要 K (recall_at) 个
                    break;

                query_result_ids[test_id][i * recall_at + pos] = neighbor.id;
                // 转换回原始距离（为MIPS还原符号）
                query_result_dists[test_id][i * recall_at + pos] =
                    (metric == diskann::Metric::INNER_PRODUCT) ? -neighbor.distance : neighbor.distance;
                pos++;
            }
            // 填充不足 K 的部分
            while (pos < recall_at)
            {
                query_result_ids[test_id][i * recall_at + pos] = 0;
                query_result_dists[test_id][i * recall_at + pos] = std::numeric_limits<float>::infinity();
                pos++;
            }

            auto qe = std::chrono::high_resolution_clock::now(); // 停止计时
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
            cmp_stats[i] = current_cmp_stats;
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        // --- 性能计算和输出 (基本不变) ---
        double displayed_qps = query_num / diff.count();
        if (show_qps_per_thread)
            displayed_qps /= num_threads;

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

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps << std::setw(20)
                  << (float)mean_latency << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];

        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
        }
        std::cout << std::endl;
    }

    // --- 保存结果 (不变) ---
    std::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        // 为每个 L 输出一份结果（ids 与 dists 分别保存）
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }
        std::string cur_result_path_prefix = result_path_prefix + "_" + std::to_string(L);

        std::string cur_result_path = cur_result_path_prefix + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = cur_result_path_prefix + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id].data(), query_num, recall_at);

        test_id++;
    }

    diskann::aligned_free(query);
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path, query_file, gt_file, label_type,
        query_filters_file; // 移除了 filter_label
    uint32_t num_threads, K;
    std::vector<uint32_t> Lvec;
    bool print_all_recalls, dynamic, tags, show_qps_per_thread;
    float fail_if_recall_below = 0.0f;
    uint32_t expand_labels_k = 0;
    bool use_label_correlation = false;
    float beta_strength = 1.0f;

    po::options_description desc{program_options_utils::make_program_description(
        "multi_filters_search_memory_index", "Searches in-memory DiskANN indexes with multi-label 'OR' logic")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
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
        // --- 修改：移除 filter_label，使 query_filters_file 成为必需 ---
        required_configs.add_options()(
            "query_filters_file", po::value<std::string>(&query_filters_file)->required(),
            "Path to query filters file. Each line contains comma-separated labels for the corresponding query.");

        // Optional parameters
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
            "Whether the index is dynamic. Dynamic indices must have associated tags.  Default false.");
        optional_configs.add_options()("tags", po::value<bool>(&tags)->default_value(false),
                                       "Whether to search with external identifiers (tags). Default false.");
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        // 标签相关性与查询扩展
        optional_configs.add_options()("expand_labels_k", po::value<uint32_t>(&expand_labels_k)->default_value(0),
                                       "Expand to Top-K correlated labels at query time (default 0)");
        optional_configs.add_options()("use_label_correlation", po::bool_switch(&use_label_correlation),
                                       "Enable beta-based label correlation during search (default off)");
        optional_configs.add_options()("beta_strength", po::value<float>(&beta_strength)->default_value(1.0f),
                                       "Strength of beta factor for label correlation (default 1.0)");

        // Output controls
        po::options_description output_controls("Output controls");
        output_controls.add_options()("print_all_recalls", po::bool_switch(&print_all_recalls),
                                      "Print recalls at all positions, from 1 up to specified "
                                      "recall_at value");
        output_controls.add_options()("print_qps_per_thread", po::bool_switch(&show_qps_per_thread),
                                      "Print overall QPS divided by the number of threads in "
                                      "the output table");

        // Merge required and optional parameters
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

    // ... (metric, dynamic/tags, fail_if_recall_below 检查保持不变) ...
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
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

    if (dynamic && not tags)
    {
        std::cerr << "Tags must be enabled while searching dynamically built indices" << std::endl;
        return -1;
    }

    if (fail_if_recall_below < 0.0 || fail_if_recall_below >= 100.0)
    {
        std::cerr << "fail_if_recall_below parameter must be between 0 and 100%" << std::endl;
        return -1;
    }

    // --- 移除 filter_label 和 query_filters_file 的冲突检查 ---
    // (因为 query_filters_file 是唯一且必需的)

    try
    {
        // 根据标签和数据类型调用模板函数
        if (label_type == "ushort")
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters_file, fail_if_recall_below, expand_labels_k,
                    use_label_correlation, beta_strength);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters_file, fail_if_recall_below, expand_labels_k,
                    use_label_correlation, beta_strength);
            }
            else if (data_type == std::string("float"))
            {
                std::cout << "check1" << std::endl;
                return search_memory_index<float, uint16_t>(
                    metric, index_path_prefix, result_path, query_file, gt_file, num_threads, K, print_all_recalls,
                    Lvec, dynamic, tags, show_qps_per_thread, query_filters_file, fail_if_recall_below, expand_labels_k,
                    use_label_correlation, beta_strength);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
        else // 默认为 uint (uint32_t)
        {
            if (data_type == std::string("int8"))
            {
                return search_memory_index<int8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                   num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                   show_qps_per_thread, query_filters_file, fail_if_recall_below,
                                                   expand_labels_k, use_label_correlation, beta_strength);
            }
            else if (data_type == std::string("uint8"))
            {
                return search_memory_index<uint8_t>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                    num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                    show_qps_per_thread, query_filters_file, fail_if_recall_below,
                                                    expand_labels_k, use_label_correlation, beta_strength);
            }
            else if (data_type == std::string("float"))
            {
                return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
                                                  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
                                                  show_qps_per_thread, query_filters_file, fail_if_recall_below,
                                                  expand_labels_k, use_label_correlation, beta_strength);
            }
            else
            {
                std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
                return -1;
            }
        }
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}