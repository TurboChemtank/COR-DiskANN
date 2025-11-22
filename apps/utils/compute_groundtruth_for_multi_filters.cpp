// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <boost/program_options.hpp>

#include "index.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "abstract_data_store.h"

namespace po = boost::program_options;

// 计算两向量距离（与 DiskANN 一致的最小化目标）
template <typename T>
static inline float compute_distance_scalar(const T *a, const T *b, uint32_t dim, diskann::Metric metric)
{
    if (metric == diskann::Metric::L2)
    {
        double sum = 0.0;
        for (uint32_t d = 0; d < dim; ++d)
        {
            double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
            sum += diff * diff;
        }
        return static_cast<float>(sum);
    }
    else if (metric == diskann::Metric::COSINE)
    {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (uint32_t d = 0; d < dim; ++d)
        {
            double va = static_cast<double>(a[d]);
            double vb = static_cast<double>(b[d]);
            dot += va * vb;
            na += va * va;
            nb += vb * vb;
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom <= 0.0)
            return 1.0f; // 最大角距离
        double cos_sim = dot / denom;
        return static_cast<float>(1.0 - cos_sim); // 作为“距离”越小越好
    }
    else if (metric == diskann::Metric::INNER_PRODUCT)
    {
        double dot = 0.0;
        for (uint32_t d = 0; d < dim; ++d)
        {
            dot += static_cast<double>(a[d]) * static_cast<double>(b[d]);
        }
        return static_cast<float>(-dot); // 最大化点积 <=> 最小化 -dot
    }
    // 默认回退到 L2
    double sum = 0.0;
    for (uint32_t d = 0; d < dim; ++d)
    {
        double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        sum += diff * diff;
    }
    return static_cast<float>(sum);
}

// 新增：辅助函数，用于解析逗号分隔的多标签文件
// 返回一个二维向量，外层索引是查询ID（行号），内层向量是该查询的标签字符串
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
            // 移除可能存在的空格
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

// 复制自 compute_groundtruth_for_filters.cpp 的简化版：读取基础数据的标签文件
// 将每一行按逗号解析为字符串标签列表，行号对应 point_id
static inline void parse_label_file_into_vec(size_t &line_cnt, const std::string &map_file,
                                             std::vector<std::vector<std::string>> &pts_to_labels)
{
    std::ifstream infile(map_file);
    std::string line, token;
    std::set<std::string> labels;
    infile.clear();
    infile.seekg(0, std::ios::beg);
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<std::string> lbls(0);

        getline(iss, token, '\t');
        std::istringstream new_iss(token);
        while (getline(new_iss, token, ','))
        {
            token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
            token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
            lbls.push_back(token);
            labels.insert(token);
        }
        std::sort(lbls.begin(), lbls.end());
        pts_to_labels.push_back(lbls);
    }
    line_cnt = pts_to_labels.size();
}

// 新增：辅助函数，检查两个标签集之间是否存在“OR”匹配（即交集非空）
template <typename LabelT>
bool check_label_set_intersection(const std::vector<LabelT> &query_labels, const std::vector<LabelT> &base_labels,
                                  const LabelT &universal_label, bool use_universal_label)
{
    if (use_universal_label)
    {
        // 如果基础点包含通用标签，则匹配
        for (const auto &lbl : base_labels)
        {
            if (lbl == universal_label)
                return true;
        }
        // 如果查询包含通用标签，则匹配
        for (const auto &lbl : query_labels)
        {
            if (lbl == universal_label)
                return true;
        }
    }

    // 检查是否有任何一个查询标签存在于基础点标签中
    for (const auto &q_lbl : query_labels)
    {
        for (const auto &b_lbl : base_labels)
        {
            if (q_lbl == b_lbl)
            {
                return true; // 找到交集
            }
        }
    }
    return false; // 没有交集
}

template <typename T>
void compute_groundtruth(const std::string &data_type, const std::string &dist_fn, const std::string &base_file,
                         const std::string &query_file, const std::string &gt_file, const uint32_t K,
                         const std::string &label_file, const std::string &universal_label_str,
                         const std::string &query_filters_file)
{
    diskann::Metric metric;
    if (dist_fn == "l2")
        metric = diskann::Metric::L2;
    else if (dist_fn == "mips")
        metric = diskann::Metric::INNER_PRODUCT;
    else if (dist_fn == "cosine")
        metric = diskann::Metric::COSINE;
    else
        throw diskann::ANNException("Unknown distance function", -1, __FUNCSIG__, __FILE__, __LINE__);

    // 加载基础数据
    T *base_data = nullptr;
    size_t npts, ndim;
    diskann::load_bin<T>(base_file, base_data, npts, ndim);

    // 加载查询数据
    T *query_data = nullptr;
    size_t nqueries, qdim;
    diskann::load_bin<T>(query_file, query_data, nqueries, qdim);

    if (ndim != qdim)
    {
        throw diskann::ANNException("Dimensions of base and query do not match", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // 加载基础数据标签（字符串形式）
    size_t num_points_labels = 0;
    std::vector<std::vector<std::string>> location_to_labels_str;
    parse_label_file_into_vec(num_points_labels, label_file, location_to_labels_str);
    if (num_points_labels != npts)
    {
        diskann::cerr << "Warning: labels lines != number of base points, proceeding with min(npts, lines)"
                      << std::endl;
    }
    diskann::cout << "Loaded base label file" << std::endl;

    // 处理通用标签
    bool use_universal_label = false;
    std::string universal_label = universal_label_str;
    if (!universal_label_str.empty())
        use_universal_label = true;

    // 加载查询标签集
    auto query_filters_string_sets = parse_query_filters_file(query_filters_file);
    if (query_filters_string_sets.size() != nqueries)
    {
        throw diskann::ANNException("Number of queries in query file and query filters file do not match", -1,
                                    __FUNCSIG__, __FILE__, __LINE__);
    }

    diskann::cout << "Loaded query filters file" << std::endl;

    // 分配GT结果空间
    uint32_t *gt_ids = new uint32_t[nqueries * K];
    float *gt_dists = new float[nqueries * K];

    diskann::cout << "Computing ground truth..." << std::endl;

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t)nqueries; i++)
    {
        // 获取当前查询的标签集
        const auto &current_query_labels = query_filters_string_sets[i];
        if (current_query_labels.empty())
        {
            // 如果一个查询没有任何有效标签，则GT为空
            for (uint32_t j = 0; j < K; j++)
            {
                gt_ids[i * K + j] = 0; // 或者使用 std::numeric_limits<uint32_t>::max()
                gt_dists[i * K + j] = std::numeric_limits<float>::infinity();
            }
            continue;
        }

        // 使用最小堆来维护 Top-K (pair<distance, id>)
        std::priority_queue<std::pair<float, uint32_t>> top_k;

        for (uint32_t j = 0; j < npts; j++)
        {
            // 关键：执行“OR”逻辑检查
            const std::vector<std::string> &base_point_labels =
                (j < location_to_labels_str.size()) ? location_to_labels_str[j] : std::vector<std::string>();
            bool matches_filter = check_label_set_intersection<std::string>(current_query_labels, base_point_labels,
                                                                            universal_label, use_universal_label);

            if (matches_filter)
            {
                // 如果标签匹配，则计算距离
                float dist =
                    compute_distance_scalar<T>(query_data + i * qdim, base_data + j * ndim, (uint32_t)ndim, metric);

                if (top_k.size() < K)
                {
                    top_k.push(std::make_pair(dist, j));
                }
                else if (dist < top_k.top().first)
                {
                    top_k.pop();
                    top_k.push(std::make_pair(dist, j));
                }
            }
        }

        // 填充结果 (从堆中取出，顺序是反的)
        int32_t k = (int32_t)top_k.size() - 1;
        while (!top_k.empty())
        {
            gt_ids[i * K + k] = top_k.top().second;
            gt_dists[i * K + k] = top_k.top().first;
            top_k.pop();
            k--;
        }
        // 填充剩余的 (如果找到的少于K个)
        for (; k >= 0; k--)
        {
            gt_ids[i * K + k] = 0; // 或者 std::numeric_limits<uint32_t>::max()
            gt_dists[i * K + k] = std::numeric_limits<float>::infinity();
        }
    }

    // 保存GT文件
    diskann::cout << "Saving ground truth to " << gt_file << std::endl;
    std::ofstream writer(gt_file, std::ios::binary);
    uint32_t nq_u32 = (uint32_t)nqueries;
    writer.write((char *)&nq_u32, sizeof(uint32_t));
    writer.write((char *)&K, sizeof(uint32_t));
    writer.write((char *)gt_ids, sizeof(uint32_t) * nqueries * K);
    writer.write((char *)gt_dists, sizeof(float) * nqueries * K);
    writer.close();

    diskann::cout << "Ground truth computation complete." << std::endl;

    delete[] gt_ids;
    delete[] gt_dists;
    diskann::aligned_free(base_data);
    diskann::aligned_free(query_data);
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, base_file, query_file, gt_file, label_file, universal_label, query_filters_file,
        label_type;
    uint32_t K;

    po::options_description desc{program_options_utils::make_program_description(
        "compute_groundtruth_for_multi_filters", "Computes ground truth for multi-label 'OR' filtered queries.")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("base_file", po::value<std::string>(&base_file)->required(),
                                       "File containing the base vectors in binary format");
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("label_file", po::value<std::string>(&label_file)->required(),
                                       "Path to label file for base data");
        required_configs.add_options()(
            "query_filters_file", po::value<std::string>(&query_filters_file)->required(),
            "Path to query filters file. Each line contains comma-separated labels for the corresponding query.");

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);

        desc.add(required_configs).add(optional_configs);

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

    try
    {
        // label_type 在本工具中不再影响流程（使用字符串标签比对）
        {
            if (data_type == "float")
                compute_groundtruth<float>(data_type, dist_fn, base_file, query_file, gt_file, K, label_file,
                                           universal_label, query_filters_file);
            else if (data_type == "int8")
                compute_groundtruth<int8_t>(data_type, dist_fn, base_file, query_file, gt_file, K, label_file,
                                            universal_label, query_filters_file);
            else if (data_type == "uint8")
                compute_groundtruth<uint8_t>(data_type, dist_fn, base_file, query_file, gt_file, K, label_file,
                                             universal_label, query_filters_file);
            else
            {
                std::cerr << "Unsupported data type" << std::endl;
                return -1;
            }
        }
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}