#include <boost/program_options.hpp>
#include <fstream>
#include <future>
#include <numeric>
#include <omp.h>
#include <sstream>
#include <string.h>
#include <time.h>

#include "abstract_index.h"
#include "filter_utils.h"
#include "index_factory.h"
#include "memory_mapper.h"
#include "program_options_utils.hpp"
#include "timer.h"
#include "utils.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace po = boost::program_options;

template <typename T>
inline void load_aligned_bin_part(const std::string &bin_file, T *data, size_t offset_points, size_t points_to_read)
{
    diskann::Timer timer;
    std::ifstream reader;
    reader.exceptions(std::ios::failbit | std::ios::badbit);
    reader.open(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    int npts_i32, dim_i32;
    reader.read((char *)&npts_i32, sizeof(int));
    reader.read((char *)&dim_i32, sizeof(int));
    size_t npts = (uint32_t)npts_i32;
    size_t dim = (uint32_t)dim_i32;

    size_t expected_actual_file_size = npts * dim * sizeof(T) + 2 * sizeof(uint32_t);
    if (actual_file_size != expected_actual_file_size)
    {
        std::stringstream stream;
        stream << "Error. File size mismatch. Actual size is " << actual_file_size << " while expected size is "
               << expected_actual_file_size << " npts = " << npts << " dim = " << dim << " size of <T>= " << sizeof(T)
               << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (offset_points + points_to_read > npts)
    {
        std::stringstream stream;
        stream << "Error. Not enough points in file. Requested " << offset_points << " offset and " << points_to_read
               << " points, but have only " << npts << " points" << std::endl;
        std::cout << stream.str();
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    reader.seekg(2 * sizeof(uint32_t) + offset_points * dim * sizeof(T));

    const size_t rounded_dim = ROUND_UP(dim, 8);
    for (size_t i = 0; i < points_to_read; i++)
    {
        reader.read((char *)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }
    reader.close();

    const double elapsed_seconds = timer.elapsed() / 1000000.0;
    std::cout << "Read " << points_to_read << " points using non-cached reads in " << elapsed_seconds << std::endl;
}

std::string get_save_filename(const std::string &save_path, size_t points_to_skip, size_t points_deleted,
                              size_t last_point_threshold)
{
    std::string final_path = save_path;
    if (points_to_skip > 0)
    {
        final_path += "skip" + std::to_string(points_to_skip) + "-";
    }

    final_path += "del" + std::to_string(points_deleted) + "-";
    final_path += std::to_string(last_point_threshold);
    return final_path;
}

template <typename T, typename TagT, typename LabelT>
void insert_till_next_checkpoint(diskann::AbstractIndex &index, size_t start, size_t end, int32_t thread_count, T *data,
                                 size_t aligned_dim, const std::vector<std::vector<LabelT>> &location_to_labels)
{
    diskann::Timer insert_timer;
    size_t failed = 0;

#pragma omp parallel for num_threads(thread_count) schedule(dynamic) reduction(+ : failed)
    for (int64_t j = (int64_t)start; j < (int64_t)end; j++)
    {
        int rc = 0;
        if (!location_to_labels.empty())
        {
            rc = index.insert_point(&data[(j - start) * aligned_dim], 1 + static_cast<TagT>(j), location_to_labels[j]);
        }
        else
        {
            rc = index.insert_point(&data[(j - start) * aligned_dim], 1 + static_cast<TagT>(j));
        }
        if (rc != 0)
        {
            failed++;
        }
    }

    const double elapsed_seconds = insert_timer.elapsed() / 1000000.0;
    std::cout << "Insertion time " << elapsed_seconds << " seconds (" << (end - start) / elapsed_seconds
              << " points/second overall, " << (end - start) / elapsed_seconds / thread_count << " per thread)";
    if (failed > 0)
    {
        std::cout << " failed=" << failed;
    }
    std::cout << std::endl;
}

template <typename TagT>
void delete_from_beginning(diskann::AbstractIndex &index, diskann::IndexWriteParameters &delete_params,
                           size_t points_to_skip, size_t points_to_delete_from_beginning)
{
    try
    {
        diskann::Timer delete_timer;
        std::cout << std::endl
                  << "Lazy deleting points " << points_to_skip << " to "
                  << points_to_skip + points_to_delete_from_beginning << "... ";
        for (size_t i = points_to_skip; i < points_to_skip + points_to_delete_from_beginning; ++i)
        {
            index.lazy_delete(static_cast<TagT>(i + 1));
        }
        std::cout << "done." << std::endl;

        auto report = index.consolidate_deletes(delete_params);
        const double elapsed_seconds = delete_timer.elapsed() / 1000000.0;
        std::cout << "#active points: " << report._active_points << std::endl
                  << "max points: " << report._max_points << std::endl
                  << "empty slots: " << report._empty_slots << std::endl
                  << "deletes processed: " << report._slots_released << std::endl
                  << "latest delete size: " << report._delete_set_size << std::endl
                  << "delete+consolidate wall time: " << elapsed_seconds << " seconds" << std::endl
                  << "rate: (" << points_to_delete_from_beginning / report._time << " points/second overall, "
                  << points_to_delete_from_beginning / report._time / delete_params.num_threads << " per thread)"
                  << std::endl;
    }
    catch (std::system_error &e)
    {
        std::cout << "Exception caught in deletion thread: " << e.what() << std::endl;
    }
}

template <typename T, typename LabelT>
void build_incremental_index(diskann::Metric metric, const std::string &data_path, diskann::IndexWriteParameters &params,
                             size_t points_to_skip, size_t max_points_to_insert, size_t beginning_index_size,
                             float start_point_norm, uint32_t num_start_pts, size_t points_per_checkpoint,
                             size_t checkpoints_per_snapshot, const std::string &save_path,
                             size_t points_to_delete_from_beginning, size_t start_deletes_after, bool concurrent,
                             const std::string &label_file, const std::string &universal_label, bool save_final_index)
{
    using TagT = uint32_t;

    size_t dim, aligned_dim;
    size_t num_points;
    diskann::get_bin_metadata(data_path, num_points, dim);
    aligned_dim = ROUND_UP(dim, 8);
    bool has_labels = !label_file.empty();

    size_t current_point_offset = points_to_skip;

    if (points_to_skip > num_points)
    {
        throw diskann::ANNException("Asked to skip more points than in data file", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (max_points_to_insert == 0)
    {
        max_points_to_insert = num_points - points_to_skip;
    }

    if (points_to_skip + max_points_to_insert > num_points)
    {
        max_points_to_insert = num_points - points_to_skip;
        std::cerr << "WARNING: Reducing max_points_to_insert to " << max_points_to_insert
                  << " points since the data file has only that many" << std::endl;
    }

    if (beginning_index_size > max_points_to_insert)
    {
        beginning_index_size = max_points_to_insert;
        std::cerr << "WARNING: Reducing beginning index size to " << beginning_index_size << std::endl;
    }

    if (checkpoints_per_snapshot > 0 && beginning_index_size > points_per_checkpoint)
    {
        beginning_index_size = points_per_checkpoint;
        std::cerr << "WARNING: Reducing beginning index size to " << beginning_index_size << std::endl;
    }

    if (points_to_delete_from_beginning > max_points_to_insert)
    {
        points_to_delete_from_beginning = max_points_to_insert;
        std::cerr << "WARNING: Reducing points_to_delete_from_beginning to " << points_to_delete_from_beginning
                  << std::endl;
    }

    const size_t last_point_threshold = points_to_skip + max_points_to_insert;
    const auto save_path_inc =
        get_save_filename(save_path + ".after-delete-", points_to_skip, points_to_delete_from_beginning, last_point_threshold);

    std::vector<std::vector<LabelT>> location_to_labels;
    std::string initial_build_label_file = label_file;
    if (has_labels)
    {
        std::string labels_file_to_use = save_path_inc + "_label_formatted.txt";
        std::string mem_labels_int_map_file = save_path_inc + "_labels_map.txt";
        convert_labels_string_to_int(label_file, labels_file_to_use, mem_labels_int_map_file, universal_label);
        auto parse_result = diskann::parse_formatted_label_file<LabelT>(labels_file_to_use);
        location_to_labels = std::get<0>(parse_result);
        if (beginning_index_size > 0)
        {
            initial_build_label_file = save_path_inc + "_initial_build_raw_labels.txt";
            std::ofstream subset_label_writer(initial_build_label_file);
            if (!subset_label_writer.is_open())
            {
                throw diskann::ANNException("Failed to open initial build label file for writing", -1, __FUNCSIG__,
                                            __FILE__, __LINE__);
            }
            for (size_t i = 0; i < beginning_index_size; ++i)
            {
                for (size_t j = 0; j < location_to_labels[i].size(); ++j)
                {
                    if (j > 0)
                    {
                        subset_label_writer << ",";
                    }
                    subset_label_writer << location_to_labels[i][j];
                }
                subset_label_writer << std::endl;
            }
            subset_label_writer.close();
        }
    }

    auto index_search_params = diskann::IndexSearchParams(params.search_list_size, params.num_threads);
    auto index_config = diskann::IndexConfigBuilder()
                            .with_metric(metric)
                            .with_dimension(dim)
                            .with_max_points(max_points_to_insert)
                            .is_dynamic_index(true)
                            .with_index_write_params(params)
                            .with_index_search_params(index_search_params)
                            .with_data_type(diskann_type_to_name<T>())
                            .with_tag_type(diskann_type_to_name<TagT>())
                            .with_label_type(diskann_type_to_name<LabelT>())
                            .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                            .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                            .is_enable_tags(true)
                            // 中文说明：COR-DiskANN 的动态图默认构建为纯 ANNS 图；
                            // 标签仅用于维护元数据、过滤搜索和相关属性扩展。
                            .is_filtered(false)
                            .with_num_frozen_pts(num_start_pts)
                            .is_concurrent_consolidate(concurrent)
                            .build();

    diskann::IndexFactory index_factory(index_config);
    auto index = index_factory.create_instance();

    if (!universal_label.empty())
    {
        LabelT u_label = 0;
        index->set_universal_label(u_label);
    }

    T *data = nullptr;
    diskann::alloc_aligned(
        (void **)&data, std::max(points_per_checkpoint, std::max<size_t>(beginning_index_size, 1)) * aligned_dim * sizeof(T),
        8 * sizeof(T));

    std::vector<TagT> tags(beginning_index_size);
    std::iota(tags.begin(), tags.end(), 1 + static_cast<TagT>(current_point_offset));

    if (beginning_index_size > 0)
    {
        load_aligned_bin_part(data_path, data, current_point_offset, beginning_index_size);
    }

    diskann::Timer init_timer;
    if (beginning_index_size > 0)
    {
        if (has_labels)
        {
            auto filter_params = diskann::IndexFilterParamsBuilder()
                                     .with_save_path_prefix(save_path_inc)
                                     .with_label_file(initial_build_label_file)
                                     .with_universal_label(universal_label)
                                     .with_post_build_label_processing(true)
                                     .build();
            std::cout << "Building pure ANNS graph with label metadata post-processing." << std::endl;
            index->build(data_path, beginning_index_size, filter_params);
        }
        else
        {
            index->build(data, beginning_index_size, tags);
        }
    }
    else
    {
        index->set_start_points_at_random(static_cast<T>(start_point_norm));
    }
    const double init_elapsed_seconds = init_timer.elapsed() / 1000000.0;
    std::cout << "Initial build time for " << beginning_index_size << " points took " << init_elapsed_seconds
              << " seconds";
    if (beginning_index_size > 0)
    {
        std::cout << " (" << beginning_index_size / init_elapsed_seconds << " points/second)";
    }
    std::cout << std::endl;

    current_point_offset += beginning_index_size;

    if (concurrent)
    {
        const auto save_path_inc = get_save_filename(save_path + ".after-concurrent-delete-", points_to_skip,
                                                     points_to_delete_from_beginning, last_point_threshold);
        int32_t sub_threads = (params.num_threads + 1) / 2;
        bool delete_launched = false;
        std::future<void> delete_task;
        diskann::Timer total_timer;

        for (size_t start = current_point_offset; start < last_point_threshold;
             start += points_per_checkpoint, current_point_offset += points_per_checkpoint)
        {
            const size_t end = std::min(start + points_per_checkpoint, last_point_threshold);
            std::cout << std::endl << "Inserting from " << start << " to " << end << std::endl;

            auto insert_task = std::async(std::launch::async, [&]() {
                load_aligned_bin_part(data_path, data, start, end - start);
                insert_till_next_checkpoint<T, TagT, LabelT>(*index, start, end, sub_threads, data, aligned_dim,
                                                             location_to_labels);
            });
            insert_task.wait();

            if (!delete_launched && points_to_delete_from_beginning > 0 && end >= start_deletes_after &&
                end >= points_to_skip + points_to_delete_from_beginning)
            {
                delete_launched = true;
                auto delete_params = diskann::IndexWriteParametersBuilder(params).with_num_threads(sub_threads).build();
                delete_task = std::async(std::launch::async, [&]() {
                    delete_from_beginning<TagT>(*index, delete_params, points_to_skip, points_to_delete_from_beginning);
                });
            }
        }

        if (delete_launched)
        {
            delete_task.wait();
        }

        std::cout << "Total elapsed " << total_timer.elapsed() / 1000 << "ms" << std::endl;
        if (save_final_index)
        {
            index->save(save_path_inc.c_str(), true);
        }
    }
    else
    {
        const auto save_path_inc = get_save_filename(save_path + ".after-delete-", points_to_skip,
                                                     points_to_delete_from_beginning, last_point_threshold);
        size_t last_snapshot_points_threshold = 0;
        size_t num_checkpoints_till_snapshot = checkpoints_per_snapshot;
        diskann::Timer total_timer;

        for (size_t start = current_point_offset; start < last_point_threshold;
             start += points_per_checkpoint, current_point_offset += points_per_checkpoint)
        {
            const size_t end = std::min(start + points_per_checkpoint, last_point_threshold);
            std::cout << std::endl << "Inserting from " << start << " to " << end << std::endl;

            load_aligned_bin_part(data_path, data, start, end - start);
            insert_till_next_checkpoint<T, TagT, LabelT>(*index, start, end, (int32_t)params.num_threads, data,
                                                         aligned_dim, location_to_labels);

            if (checkpoints_per_snapshot > 0 && --num_checkpoints_till_snapshot == 0)
            {
                diskann::Timer save_timer;
                const auto snapshot_path =
                    get_save_filename(save_path + ".inc-", points_to_skip, points_to_delete_from_beginning, end);
                index->save(snapshot_path.c_str(), false);
                const double elapsed_seconds = save_timer.elapsed() / 1000000.0;
                const size_t points_saved = end - points_to_skip;

                std::cout << "Saved " << points_saved << " points in " << elapsed_seconds << " seconds ("
                          << points_saved / elapsed_seconds << " points/second)" << std::endl;

                num_checkpoints_till_snapshot = checkpoints_per_snapshot;
                last_snapshot_points_threshold = end;
            }

            std::cout << "Number of points in the index post insertion " << end << std::endl;
        }

        if (checkpoints_per_snapshot > 0 && last_snapshot_points_threshold != last_point_threshold)
        {
            const auto snapshot_path = get_save_filename(save_path + ".inc-", points_to_skip, points_to_delete_from_beginning,
                                                         last_point_threshold);
            std::cout << "Final snapshot path would be " << snapshot_path << std::endl;
        }

        if (points_to_delete_from_beginning > 0)
        {
            auto delete_params = diskann::IndexWriteParametersBuilder(params).with_num_threads(params.num_threads).build();
            delete_from_beginning<TagT>(*index, delete_params, points_to_skip, points_to_delete_from_beginning);
        }

        std::cout << "Total elapsed " << total_timer.elapsed() / 1000 << "ms" << std::endl;
        if (save_final_index)
        {
            index->save(save_path_inc.c_str(), true);
        }
    }

    diskann::aligned_free(data);
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t num_threads, R, L, num_start_pts;
    float alpha, start_point_norm, filter_lazy_update_ratio;
    size_t points_to_skip, max_points_to_insert, beginning_index_size, points_per_checkpoint, checkpoints_per_snapshot,
        points_to_delete_from_beginning, start_deletes_after;
    bool concurrent, save_final_index;

    std::string label_file, label_type, universal_label;
    uint32_t Lf, unique_labels_supported, filter_lazy_update_min_ops, expand_labels_k, label_projection_dim;

    po::options_description desc{program_options_utils::make_program_description(
        "test_incremental_filtered_updates", "Test incremental insert/delete timing for modified filtered index")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("data_path", po::value<std::string>(&data_path)->required(),
                                       program_options_utils::INPUT_DATA_PATH);
        required_configs.add_options()("points_to_skip", po::value<uint64_t>(&points_to_skip)->required(),
                                       "Skip these first set of points from file");
        required_configs.add_options()("beginning_index_size", po::value<uint64_t>(&beginning_index_size)->required(),
                                       "Batch build will be called on these set of points");
        required_configs.add_options()("points_per_checkpoint", po::value<uint64_t>(&points_per_checkpoint)->required(),
                                       "Insertions are done in batches of points_per_checkpoint");
        required_configs.add_options()("checkpoints_per_snapshot",
                                       po::value<uint64_t>(&checkpoints_per_snapshot)->required(),
                                       "Save the index to disk every few checkpoints");
        required_configs.add_options()("points_to_delete_from_beginning",
                                       po::value<uint64_t>(&points_to_delete_from_beginning)->required(),
                                       "Number of points to lazily delete from the beginning");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("max_degree,R", po::value<uint32_t>(&R)->default_value(64),
                                       program_options_utils::MAX_BUILD_DEGREE);
        optional_configs.add_options()("Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
                                       program_options_utils::GRAPH_BUILD_COMPLEXITY);
        optional_configs.add_options()("alpha", po::value<float>(&alpha)->default_value(1.2f),
                                       program_options_utils::GRAPH_BUILD_ALPHA);
        optional_configs.add_options()("max_points_to_insert",
                                       po::value<uint64_t>(&max_points_to_insert)->default_value(0),
                                       "These number of points from the file are inserted after points_to_skip");
        optional_configs.add_options()("do_concurrent", po::value<bool>(&concurrent)->default_value(false),
                                       "Whether to overlap insert and delete");
        optional_configs.add_options()("start_deletes_after",
                                       po::value<uint64_t>(&start_deletes_after)->default_value(0),
                                       "Launch deletes after this many total inserted points");
        optional_configs.add_options()("start_point_norm", po::value<float>(&start_point_norm)->default_value(0),
                                       "Set the start point to a random point on a sphere of this radius");

        optional_configs.add_options()("label_file", po::value<std::string>(&label_file)->default_value(""),
                                       program_options_utils::LABEL_FILE);
        optional_configs.add_options()("universal_label", po::value<std::string>(&universal_label)->default_value(""),
                                       program_options_utils::UNIVERSAL_LABEL);
        optional_configs.add_options()("FilteredLbuild,Lf", po::value<uint32_t>(&Lf)->default_value(0),
                                       program_options_utils::FILTERED_LBUILD);
        optional_configs.add_options()("label_type", po::value<std::string>(&label_type)->default_value("uint"),
                                       program_options_utils::LABEL_TYPE_DESCRIPTION);
        optional_configs.add_options()("unique_labels_supported",
                                       po::value<uint32_t>(&unique_labels_supported)->default_value(0),
                                       "Number of unique labels supported by the dynamic index.");
        optional_configs.add_options()(
            "num_start_points",
            po::value<uint32_t>(&num_start_pts)->default_value(diskann::defaults::NUM_FROZEN_POINTS_DYNAMIC),
            "Set the number of random start (frozen) points to use when inserting");
        optional_configs.add_options()("filter_lazy_update_ratio",
                                       po::value<float>(&filter_lazy_update_ratio)->default_value(0.01f),
                                       "Lazy metadata refresh ratio");
        optional_configs.add_options()("filter_lazy_update_min_ops",
                                       po::value<uint32_t>(&filter_lazy_update_min_ops)->default_value(128),
                                       "Lazy metadata refresh minimum operations");
        optional_configs.add_options()("expand_labels_k", po::value<uint32_t>(&expand_labels_k)->default_value(0),
                                       "Kept only for config compatibility");
        optional_configs.add_options()("label_projection_dim", po::value<uint32_t>(&label_projection_dim)->default_value(32),
                                       "Low-dimensional projection size for label correlation centroids (0 disables)");
        optional_configs.add_options()("save_final_index",
                                       po::bool_switch(&save_final_index)->default_value(false),
                                       "Save final compacted index");

        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);

        if (beginning_index_size == 0 && start_point_norm == 0)
        {
            std::cout << "When beginning_index_size is 0, use a start point with appropriate norm" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (label_type != std::string("ushort") && label_type != std::string("uint"))
    {
        std::cerr << "Invalid label type. Supported types are uint and ushort" << std::endl;
        return -1;
    }

    if (data_type != std::string("int8") && data_type != std::string("uint8") && data_type != std::string("float"))
    {
        std::cerr << "Invalid data type. Supported types are int8, uint8 and float" << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("mips") && data_type == std::string("float"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cerr << "Invalid distance function. Supported functions are l2, cosine and mips(float only)" << std::endl;
        return -1;
    }

    (void)unique_labels_supported;

    try
    {
        auto params = diskann::IndexWriteParametersBuilder(L, R)
                          .with_max_occlusion_size(500)
                          .with_alpha(alpha)
                          .with_num_threads(num_threads)
                          .with_filter_list_size(Lf)
                          .with_num_correlated_labels_to_expand(expand_labels_k)
                          .with_label_projection_dim(label_projection_dim)
                          .with_filter_lazy_update_ratio(filter_lazy_update_ratio)
                          .with_filter_lazy_update_min_ops(filter_lazy_update_min_ops)
                          .build();

        if (data_type == std::string("int8"))
        {
            if (label_type == std::string("ushort"))
            {
                build_incremental_index<int8_t, uint16_t>(metric, data_path, params, points_to_skip, max_points_to_insert,
                                                          beginning_index_size, start_point_norm, num_start_pts,
                                                          points_per_checkpoint, checkpoints_per_snapshot,
                                                          index_path_prefix, points_to_delete_from_beginning,
                                                          start_deletes_after, concurrent, label_file, universal_label,
                                                          save_final_index);
            }
            else
            {
                build_incremental_index<int8_t, uint32_t>(metric, data_path, params, points_to_skip, max_points_to_insert,
                                                          beginning_index_size, start_point_norm, num_start_pts,
                                                          points_per_checkpoint, checkpoints_per_snapshot,
                                                          index_path_prefix, points_to_delete_from_beginning,
                                                          start_deletes_after, concurrent, label_file, universal_label,
                                                          save_final_index);
            }
        }
        else if (data_type == std::string("uint8"))
        {
            if (label_type == std::string("ushort"))
            {
                build_incremental_index<uint8_t, uint16_t>(
                    metric, data_path, params, points_to_skip, max_points_to_insert, beginning_index_size,
                    start_point_norm, num_start_pts, points_per_checkpoint, checkpoints_per_snapshot, index_path_prefix,
                    points_to_delete_from_beginning, start_deletes_after, concurrent, label_file, universal_label,
                    save_final_index);
            }
            else
            {
                build_incremental_index<uint8_t, uint32_t>(
                    metric, data_path, params, points_to_skip, max_points_to_insert, beginning_index_size,
                    start_point_norm, num_start_pts, points_per_checkpoint, checkpoints_per_snapshot, index_path_prefix,
                    points_to_delete_from_beginning, start_deletes_after, concurrent, label_file, universal_label,
                    save_final_index);
            }
        }
        else if (data_type == std::string("float"))
        {
            if (label_type == std::string("ushort"))
            {
                build_incremental_index<float, uint16_t>(metric, data_path, params, points_to_skip, max_points_to_insert,
                                                         beginning_index_size, start_point_norm, num_start_pts,
                                                         points_per_checkpoint, checkpoints_per_snapshot,
                                                         index_path_prefix, points_to_delete_from_beginning,
                                                         start_deletes_after, concurrent, label_file, universal_label,
                                                         save_final_index);
            }
            else
            {
                build_incremental_index<float, uint32_t>(metric, data_path, params, points_to_skip, max_points_to_insert,
                                                         beginning_index_size, start_point_norm, num_start_pts,
                                                         points_per_checkpoint, checkpoints_per_snapshot,
                                                         index_path_prefix, points_to_delete_from_beginning,
                                                         start_deletes_after, concurrent, label_file, universal_label,
                                                         save_final_index);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Caught unknown exception" << std::endl;
        return -1;
    }

    return 0;
}
