// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common_includes.h"

#ifdef EXEC_ENV_OLS
#include "aligned_file_reader.h"
#endif

#include "distance.h"
#include "locking.h"
#include "natural_number_map.h"
#include "natural_number_set.h"
#include "neighbor.h"
#include "parameters.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch.h"
#include "in_mem_data_store.h"
#include "in_mem_graph_store.h"
#include "abstract_index.h"

#include "quantized_distance.h"
#include "pq_data_store.h"
#include <atomic>

#define OVERHEAD_FACTOR 1.1
#define EXPAND_IF_FULL 0
#define DEFAULT_MAXC 750

namespace diskann
{

inline double estimate_ram_usage(size_t size, uint32_t dim, uint32_t datasize, uint32_t degree)
{
    double size_of_data = ((double)size) * ROUND_UP(dim, 8) * datasize;
    double size_of_graph = ((double)size) * degree * sizeof(uint32_t) * defaults::GRAPH_SLACK_FACTOR;
    double size_of_locks = ((double)size) * sizeof(non_recursive_mutex);
    double size_of_outer_vector = ((double)size) * sizeof(ptrdiff_t);

    return OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector);
}

template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t> class Index : public AbstractIndex
{
    /**************************************************************************
     *
     * Public functions acquire one or more of _update_lock, _consolidate_lock,
     * _tag_lock, _delete_lock before calling protected functions which DO NOT
     * acquire these locks. They might acquire locks on _locks[i]
     *
     **************************************************************************/

  public:
    // Constructor for Bulk operations and for creating the index object solely
    // for loading a prexisting index.
    DISKANN_DLLEXPORT Index(const IndexConfig &index_config, std::shared_ptr<AbstractDataStore<T>> data_store,
                            std::unique_ptr<AbstractGraphStore> graph_store,
                            std::shared_ptr<AbstractDataStore<T>> pq_data_store = nullptr);

    // Constructor for incremental index
    DISKANN_DLLEXPORT Index(Metric m, const size_t dim, const size_t max_points,
                            const std::shared_ptr<IndexWriteParameters> index_parameters,
                            const std::shared_ptr<IndexSearchParams> index_search_params,
                            const size_t num_frozen_pts = 0, const bool dynamic_index = false,
                            const bool enable_tags = false, const bool concurrent_consolidate = false,
                            const bool pq_dist_build = false, const size_t num_pq_chunks = 0,
                            const bool use_opq = false, const bool filtered_index = false);

    DISKANN_DLLEXPORT ~Index();

    // Saves graph, data, metadata and associated tags.
    DISKANN_DLLEXPORT void save(const char *filename, bool compact_before_save = false);

    // Load functions
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void load(AlignedFileReader &reader, uint32_t num_threads, uint32_t search_l);
#else
    // Reads the number of frozen points from graph's metadata file section.
    DISKANN_DLLEXPORT static size_t get_graph_num_frozen_points(const std::string &graph_file);

    DISKANN_DLLEXPORT void load(const char *index_file, uint32_t num_threads, uint32_t search_l);
#endif

    // get some private variables
    DISKANN_DLLEXPORT size_t get_num_points();
    DISKANN_DLLEXPORT size_t get_max_points();

    DISKANN_DLLEXPORT bool detect_common_filters(uint32_t point_id, bool search_invocation,
                                                 const std::vector<LabelT> &incoming_labels);

    // Batch build from a file. Optionally pass tags vector.
    DISKANN_DLLEXPORT void build(const char *filename, const size_t num_points_to_load,
                                 const std::vector<TagT> &tags = std::vector<TagT>());

    // Batch build from a file. Optionally pass tags file.
    DISKANN_DLLEXPORT void build(const char *filename, const size_t num_points_to_load, const char *tag_filename);

    // Batch build from a data array, which must pad vectors to aligned_dim
    DISKANN_DLLEXPORT void build(const T *data, const size_t num_points_to_load, const std::vector<TagT> &tags);

    // Based on filter params builds a filtered or unfiltered index
    DISKANN_DLLEXPORT void build(const std::string &data_file, const size_t num_points_to_load,
                                 IndexFilterParams &filter_params);

    // Filtered Support
    DISKANN_DLLEXPORT void build_filtered_index(const char *filename, const std::string &label_file,
                                                const size_t num_points_to_load,
                                                const std::vector<TagT> &tags = std::vector<TagT>());

    DISKANN_DLLEXPORT void set_universal_label(const LabelT &label);

    // 运行时覆盖查询时标签扩展K（可选）
    void set_expand_labels_k(uint32_t k) override
    {
        _num_correlated_labels_to_expand = k;
        if (_num_correlated_labels_to_expand > 0 && !_location_to_labels.empty())
        {
            // 运行时启用扩展时，确保已根据当前已加载的标签数据计算相关性与Top-K
            calculate_label_correlations();
            compute_top_k_label_correlations();
        }
    }

    // Get converted integer label from string to int map (_label_map)
    DISKANN_DLLEXPORT LabelT get_converted_label(const std::string &raw_label) const;

    DISKANN_DLLEXPORT bool matches_any_labels(const TagT &tag, const std::vector<std::string> &raw_labels);

    // 【新增接口 - 中文说明】供上层按频率门槛决定是否扩展搜索
    uint32_t get_filter_frequency_threshold() const override
    {
        return _label_frequency_otsu_threshold;
    }

    uint32_t get_filter_frequency(const std::string &raw_label) const override
    {
        LabelT label{};
        auto it = _label_map.find(raw_label);
        if (it != _label_map.end())
        {
            label = it->second;
        }
        else
        {
            try
            {
                label = (LabelT)std::stoull(raw_label);
            }
            catch (const std::exception &)
            {
                if (_use_universal_label)
                {
                    label = _universal_label;
                }
                else
                {
                    return 0;
                }
            }
        }

        if (label >= _label_frequency.size())
        {
            auto delta_it = _pending_label_frequency_delta.find(label);
            if (delta_it == _pending_label_frequency_delta.end())
            {
                return 0;
            }
            return delta_it->second > 0 ? static_cast<uint32_t>(delta_it->second) : 0;
        }

        int64_t freq = static_cast<int64_t>(_label_frequency[label]);
        auto delta_it = _pending_label_frequency_delta.find(label);
        if (delta_it != _pending_label_frequency_delta.end())
        {
            freq += delta_it->second;
        }
        return freq > 0 ? static_cast<uint32_t>(freq) : 0;
    }

    // 【新增接口实现 - 中文说明】打印频率数组的基本信息（避免输出过多）
    void check_label_frequency() const override
    {
        uint64_t nonzero = 0;
        uint64_t sum = 0;
        uint32_t max_freq = 0;
        for (size_t i = 0; i < _label_frequency.size(); i++)
        {
            const uint32_t f = _label_frequency[i];
            if (f > 0)
            {
                nonzero += 1;
                sum += f;
                max_freq = std::max(max_freq, f);
                std::cout<<i<<" "<<f<<std::endl;
            }
        }

        diskann::cout << "label_frequency_size=" << _label_frequency.size() << " nonzero=" << nonzero
                      << " sum=" << sum << " max_freq=" << max_freq
                      << " otsu_threshold=" << _label_frequency_otsu_threshold << std::endl;
    }

    // 中文说明：支持“多标签 OR 合并搜索”，可指定是否额外加入默认全局起点。
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
    search_with_filter_label_group(const T *query, const std::vector<std::string> &raw_filter_labels, const size_t K,
                                   const uint32_t L, uint32_t *indices, float *distances,
                                   const int expand_num = -1, const bool include_unfiltered_starts = false);

    // 中文说明：与上面相同，但输出外部 tag，便于动态索引评测。
    DISKANN_DLLEXPORT std::pair<size_t, uint32_t>
    search_with_filter_label_group_tags(const T *query, const uint64_t K, const uint32_t L, TagT *tags,
                                        float *distances, std::vector<T *> &res_vectors,
                                        const std::vector<std::string> &raw_filter_labels,
                                        const int expand_num = -1, const bool include_unfiltered_starts = false);

    // 中文说明：基于倒排索引直接枚举候选并爆搜单标签。
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t>
    brute_force_search_filter_label(const T *query, const std::string &raw_filter_label, const size_t K,
                                    uint32_t *indices, float *distances);

    // 中文说明：基于倒排索引直接枚举候选并爆搜单标签，输出外部 tag。
    DISKANN_DLLEXPORT std::pair<size_t, uint32_t>
    brute_force_search_filter_label_tags(const T *query, const std::string &raw_filter_label, const uint64_t K,
                                         TagT *tags, float *distances, std::vector<T *> &res_vectors);

    // Set starting point of an index before inserting any points incrementally.
    // The data count should be equal to _num_frozen_pts * _aligned_dim.
    DISKANN_DLLEXPORT void set_start_points(const T *data, size_t data_count);
    // Set starting points to random points on a sphere of certain radius.
    // A fixed random seed can be specified for scenarios where it's important
    // to have higher consistency between index builds.
    DISKANN_DLLEXPORT void set_start_points_at_random(T radius, uint32_t random_seed = 0);

    // For FastL2 search on a static index, we interleave the data with graph
    DISKANN_DLLEXPORT void optimize_index_layout();

    // For FastL2 search on optimized layout
    DISKANN_DLLEXPORT void search_with_optimized_layout(const T *query, size_t K, size_t L, uint32_t *indices);

    // Added search overload that takes L as parameter, so that we
    // can customize L on a per-query basis without tampering with "Parameters"
    template <typename IDType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search(const T *query, const size_t K, const uint32_t L,
                                                           IDType *indices, float *distances = nullptr);

    // Initialize space for res_vectors before calling.
    DISKANN_DLLEXPORT size_t search_with_tags(const T *query, const uint64_t K, const uint32_t L, TagT *tags,
                                              float *distances, std::vector<T *> &res_vectors, bool use_filters = false,
                                              const std::string filter_label = "", const int expand_num = -1);

    // Filter support search
    template <typename IndexType>
    DISKANN_DLLEXPORT std::pair<uint32_t, uint32_t> search_with_filters(const T *query, const LabelT &filter_label,
                                                                        const size_t K, const uint32_t L,
                                                                        IndexType *indices, float *distances,
                                                                        const int expand_num = -1);

    // Will fail if tag already in the index or if tag=0.
    DISKANN_DLLEXPORT int insert_point(const T *point, const TagT tag);

    // Will fail if tag already in the index or if tag=0.
    DISKANN_DLLEXPORT int insert_point(const T *point, const TagT tag, const std::vector<LabelT> &label);

    // call this before issuing deletions to sets relevant flags
    DISKANN_DLLEXPORT int enable_delete();

    // Record deleted point now and restructure graph later. Return -1 if tag
    // not found, 0 if OK.
    DISKANN_DLLEXPORT int lazy_delete(const TagT &tag);

    // Record deleted points now and restructure graph later. Add to failed_tags
    // if tag not found.
    DISKANN_DLLEXPORT void lazy_delete(const std::vector<TagT> &tags, std::vector<TagT> &failed_tags);

    // Call after a series of lazy deletions
    // Returns number of live points left after consolidation
    // If _conc_consolidates is set in the ctor, then this call can be invoked
    // alongside inserts and lazy deletes, else it acquires _update_lock
    DISKANN_DLLEXPORT consolidation_report consolidate_deletes(const IndexWriteParameters &parameters);

    DISKANN_DLLEXPORT void prune_all_neighbors(const uint32_t max_degree, const uint32_t max_occlusion,
                                               const float alpha);

    DISKANN_DLLEXPORT bool is_index_saved();

    // repositions frozen points to the end of _data - if they have been moved
    // during deletion
    DISKANN_DLLEXPORT void reposition_frozen_point_to_end();
    DISKANN_DLLEXPORT void reposition_points(uint32_t old_location_start, uint32_t new_location_start,
                                             uint32_t num_locations);

    // DISKANN_DLLEXPORT void save_index_as_one_file(bool flag);

    DISKANN_DLLEXPORT void get_active_tags(tsl::robin_set<TagT> &active_tags);

    // memory should be allocated for vec before calling this function
    DISKANN_DLLEXPORT int get_vector_by_tag(TagT &tag, T *vec);

    DISKANN_DLLEXPORT void print_status();

    DISKANN_DLLEXPORT void count_nodes_at_bfs_levels();

    // This variable MUST be updated if the number of entries in the metadata
    // change.
    DISKANN_DLLEXPORT static const int METADATA_ROWS = 5;

    // ********************************
    //
    // Internals of the library
    //
    // ********************************

  protected:
    // overload of abstract index virtual methods
    virtual void _build(const DataType &data, const size_t num_points_to_load, TagVector &tags) override;

    virtual std::pair<uint32_t, uint32_t> _search(const DataType &query, const size_t K, const uint32_t L,
                                                  std::any &indices, float *distances = nullptr) override;
    virtual std::pair<uint32_t, uint32_t> _search_with_filters(const DataType &query,
                                                               const std::string &filter_label_raw, const size_t K,
                                                               const uint32_t L, const int expand_num,
                                                               std::any &indices,
                                                               float *distances) override;

    virtual int _insert_point(const DataType &data_point, const TagType tag) override;
    virtual int _insert_point(const DataType &data_point, const TagType tag, Labelvector &labels) override;

    virtual int _lazy_delete(const TagType &tag) override;

    virtual void _lazy_delete(TagVector &tags, TagVector &failed_tags) override;

    virtual void _get_active_tags(TagRobinSet &active_tags) override;

    virtual void _set_start_points_at_random(DataType radius, uint32_t random_seed = 0) override;

    virtual bool _matches_any_labels(const TagType &tag, const std::vector<std::string> &raw_labels) override;

    virtual int _get_vector_by_tag(TagType &tag, DataType &vec) override;

    virtual void _search_with_optimized_layout(const DataType &query, size_t K, size_t L, uint32_t *indices) override;

    virtual size_t _search_with_tags(const DataType &query, const uint64_t K, const uint32_t L, const TagType &tags,
                                     float *distances, DataVector &res_vectors, bool use_filters = false,
                                     const std::string filter_label = "", const int expand_num = -1) override;

    virtual void _set_universal_label(const LabelType universal_label) override;

    // No copy/assign.
    Index(const Index<T, TagT, LabelT> &) = delete;
    Index<T, TagT, LabelT> &operator=(const Index<T, TagT, LabelT> &) = delete;

    // Use after _data and _nd have been populated
    // Acquire exclusive _update_lock before calling
    void build_with_data_populated(const std::vector<TagT> &tags);

    // generates 1 frozen point that will never be deleted from the graph
    // This is not visible to the user
    void generate_frozen_point();

    // determines navigating node of the graph by calculating medoid of datafopt
    uint32_t calculate_entry_point();

    void parse_label_file(const std::string &label_file, size_t &num_pts_labels);

    // 【新增声明 - 中文说明】基于已加载的标签，准备频率统计与标签起点
    void prepare_label_metadata(const size_t num_points_to_load);

    // 计算每个标签的Top-K相关标签（按相关性分数降序）
    void compute_top_k_label_correlations();

    std::unordered_map<std::string, LabelT> load_label_map(const std::string &map_file);

    // Returns the locations of start point and frozen points suitable for use
    // with iterate_to_fixed_point.
    std::vector<uint32_t> get_init_ids();

    // The query to use is placed in scratch->aligned_query
    std::pair<uint32_t, uint32_t> iterate_to_fixed_point(InMemQueryScratch<T> *scratch, const uint32_t Lindex,
                                                         const std::vector<uint32_t> &init_ids, bool use_filter,
                                                         const std::vector<LabelT> &filters, bool search_invocation);

    void search_for_point_and_prune(int location, uint32_t Lindex, std::vector<uint32_t> &pruned_list,
                                    InMemQueryScratch<T> *scratch, bool use_filter = false,
                                    uint32_t filteredLindex = 0);

    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, std::vector<uint32_t> &pruned_list,
                         InMemQueryScratch<T> *scratch);

    void prune_neighbors(const uint32_t location, std::vector<Neighbor> &pool, const uint32_t range,
                         const uint32_t max_candidate_size, const float alpha, std::vector<uint32_t> &pruned_list,
                         InMemQueryScratch<T> *scratch);

    // Prunes candidates in @pool to a shorter list @result
    // @pool must be sorted before calling
    void occlude_list(const uint32_t location, std::vector<Neighbor> &pool, const float alpha, const uint32_t degree,
                      const uint32_t maxc, std::vector<uint32_t> &result, InMemQueryScratch<T> *scratch,
                      const tsl::robin_set<uint32_t> *const delete_set_ptr = nullptr);

    // 【新增声明 - 中文说明】计算标签相关性的内部方法，在构建过滤索引前调用
    void calculate_label_correlations();
    void initialize_label_projection();
    bool use_projected_label_centroids() const;
    void rebuild_projected_label_centroids();
    void update_projected_label_centroid(const LabelT &label);
    float compute_label_correlation_distance(const std::vector<float> &lhs, const std::vector<float> &rhs) const;
    // 基于当前标签质心重建 KMeans 簇索引
    void rebuild_label_centroid_clusters();
    // 在增量刷新时，仅更新脏标签涉及到的簇划分与簇中心
    void refresh_dirty_label_centroid_clusters(tsl::robin_set<uint32_t> &touched_clusters);
    // 重新计算单个簇的中心向量
    void recompute_label_cluster_center(uint32_t cluster_id);
    // 为某个标签质心找到最近的若干个簇
    std::vector<uint32_t> get_nearest_label_clusters(const std::vector<float> &centroid) const;
    // 基于簇筛选候选集后，重算给定标签集合的 Top-K 相关标签
    size_t recompute_label_top_correlations(const std::vector<LabelT> &labels);

    // 【新增声明 - 中文说明】增量插入时，使用新点的标签更新相关性统计与矩阵
    void update_label_correlations_incremental(const std::vector<LabelT> &labels);

    struct PendingLabelEvent
    {
        bool is_insert = true;
        TagT tag{};
        uint32_t location = 0;
        uint64_t seq = 0;
    };

    // 判断是否达到标签元数据懒更新阈值（基于插入+删除累计次数）
    bool need_flush_pending_label_updates() const;
    // 应用累积的标签元数据增量（频率、门槛、centroid、medoid、相关性、Top-K）
    void flush_pending_label_updates(const bool force = false);
    // 记录插入事件及对应标签增量
    void record_insert_label_updates(uint32_t location, const TagT tag, const T *point,
                                     const std::vector<LabelT> &labels);
    // 记录删除事件及对应标签增量
    void record_delete_label_updates(uint32_t location, const TagT tag);
    // 批量更新时，仅刷新脏标签的medoid
    void refresh_dirty_label_medoids();
    // 批量更新时，基于簇筛选仅刷新受影响标签的 Top-K 相关性
    void refresh_dirty_label_correlations();

    // 中文说明：把原始标签字符串转换成内部标签ID；允许直接传数值字符串。
    bool try_convert_label(const std::string &raw_label, LabelT &label) const;
    // 中文说明：批量转换并排序去重，忽略无法识别的标签。
    std::vector<LabelT> convert_filter_labels(const std::vector<std::string> &raw_filter_labels) const;
    // 中文说明：把原始查询标签与相关标签扩展合并成真正参与图遍历的过滤集合。
    std::vector<LabelT> build_expanded_filter_labels(const std::vector<LabelT> &base_labels, int expand_num) const;
    // 中文说明：重建“标签 -> 活跃向量 location”的倒排索引。
    void rebuild_filter_inverted_index();
    // 中文说明：插入时把 location 加入对应标签的倒排链。
    void add_location_to_filter_inverted_index(uint32_t location, const std::vector<LabelT> &labels);
    // 中文说明：删除时把 location 从对应标签的倒排链移除。
    void remove_location_from_filter_inverted_index(uint32_t location, const std::vector<LabelT> &labels);
    // 中文说明：收集单标签爆搜候选；若启用了 universal label，会自动合并对应 posting。
    void collect_bruteforce_filter_candidates(const LabelT &filter_label, std::vector<uint32_t> &candidate_locations) const;

    template <typename IdType>
    std::pair<uint32_t, uint32_t> brute_force_search_pending_label(const T *query, const LabelT &filter_label,
                                                                   const size_t K, IdType *indices, float *distances,
                                                                   const bool return_tags = false,
                                                                   std::vector<T *> *res_vectors = nullptr);

    // add reverse links from all the visited nodes to node n.
    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, const uint32_t range,
                      InMemQueryScratch<T> *scratch);

    void inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, InMemQueryScratch<T> *scratch);

    // Acquire exclusive _update_lock before calling
    void link();

    // Acquire exclusive _tag_lock and _delete_lock before calling
    int reserve_location();

    // Acquire exclusive _tag_lock before calling
    size_t release_location(int location);
    size_t release_locations(const tsl::robin_set<uint32_t> &locations);

    // Resize the index when no slots are left for insertion.
    // Acquire exclusive _update_lock and _tag_lock before calling.
    void resize(size_t new_max_points);

    // Acquire unique lock on _update_lock, _consolidate_lock, _tag_lock
    // and _delete_lock before calling these functions.
    // Renumber nodes, update tag and location maps and compact the
    // graph, mode = _consolidated_order in case of lazy deletion and
    // _compacted_order in case of eager deletion
    DISKANN_DLLEXPORT void compact_data();
    DISKANN_DLLEXPORT void compact_frozen_point();

    // Remove deleted nodes from adjacency list of node loc
    // Replace removed neighbors with second order neighbors.
    // Also acquires _locks[i] for i = loc and out-neighbors of loc.
    void process_delete(const tsl::robin_set<uint32_t> &old_delete_set, size_t loc, const uint32_t range,
                        const uint32_t maxc, const float alpha, InMemQueryScratch<T> *scratch);

    void initialize_query_scratch(uint32_t num_threads, uint32_t search_l, uint32_t indexing_l, uint32_t r,
                                  uint32_t maxc, size_t dim);

    // Do not call without acquiring appropriate locks
    // call public member functions save and load to invoke these.
    DISKANN_DLLEXPORT size_t save_graph(std::string filename);
    DISKANN_DLLEXPORT size_t save_data(std::string filename);
    DISKANN_DLLEXPORT size_t save_tags(std::string filename);
    DISKANN_DLLEXPORT size_t save_delete_list(const std::string &filename);
#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT size_t load_graph(AlignedFileReader &reader, size_t expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_tags(AlignedFileReader &reader);
    DISKANN_DLLEXPORT size_t load_delete_set(AlignedFileReader &reader);
#else
    DISKANN_DLLEXPORT size_t load_graph(const std::string filename, size_t expected_num_points);
    DISKANN_DLLEXPORT size_t load_data(std::string filename0);
    DISKANN_DLLEXPORT size_t load_tags(const std::string tag_file_name);
    DISKANN_DLLEXPORT size_t load_delete_set(const std::string &filename);
#endif

  private:
    // Distance functions
    Metric _dist_metric = diskann::L2;

    // Data
    std::shared_ptr<AbstractDataStore<T>> _data_store;

    // Graph related data structures
    std::unique_ptr<AbstractGraphStore> _graph_store;

    char *_opt_graph = nullptr;

    // Dimensions
    size_t _dim = 0;
    size_t _nd = 0;         // number of active points i.e. existing in the graph
    size_t _max_points = 0; // total number of points in given data set

    // _num_frozen_pts is the number of points which are used as initial
    // candidates when iterating to closest point(s). These are not visible
    // externally and won't be returned by search. At least 1 frozen point is
    // needed for a dynamic index. The frozen points have consecutive locations.
    // See also _start below.
    size_t _num_frozen_pts = 0;
    size_t _frozen_pts_used = 0;
    size_t _node_size;
    size_t _data_len;
    size_t _neighbor_len;

    //  Start point of the search. When _num_frozen_pts is greater than zero,
    //  this is the location of the first frozen point. Otherwise, this is a
    //  location of one of the points in index.
    uint32_t _start = 0;

    bool _has_built = false;
    bool _saturate_graph = false;
    bool _save_as_one_file = false; // plan to support in next version
    bool _dynamic_index = false;
    bool _enable_tags = false;
    bool _normalize_vecs = false; // Using normalied L2 for cosine.
    bool _deletes_enabled = false;

    // Filter Support

    bool _filtered_index = false;
    // Location to label is only updated during insert_point(), all other reads are protected by
    // default as a location can only be released at end of consolidate deletes
    std::vector<std::vector<LabelT>> _location_to_labels;
    tsl::robin_set<LabelT> _labels;
    std::string _labels_file;
    std::unordered_map<LabelT, uint32_t> _label_to_start_id;
    std::unordered_map<uint32_t, uint32_t> _medoid_counts;

    bool _use_universal_label = false;
    LabelT _universal_label = 0;
    uint32_t _filterIndexingQueueSize;
    std::unordered_map<std::string, LabelT> _label_map;

    // 【新增成员 - 中文说明】标签相关性缓存：当前仅缓存已重算标签的候选相关分数
    // 采用嵌套map: labelA -> (labelB -> score)
    std::unordered_map<LabelT, std::unordered_map<LabelT, float>> _label_correlation_matrix;

    // 每个标签的Top-K相关标签列表
    std::unordered_map<LabelT, std::vector<std::pair<float, LabelT>>> _label_top_correlations;

    // 【新增成员 - 中文说明】每个标签的中心向量（用于基于中心距离的相关度）
    std::unordered_map<LabelT, std::vector<float>> _label_centroids;
    // 兼容旧参数保留的投影缓存；当前相关度计算已不再使用低维投影
    std::unordered_map<LabelT, std::vector<float>> _projected_label_centroids;
    std::vector<float> _label_projection_matrix;
    uint32_t _label_projection_dim{32};
    // 标签质心聚类索引：簇中心、簇成员与标签到簇的映射
    std::vector<std::vector<float>> _label_cluster_centers;
    std::vector<std::vector<LabelT>> _label_cluster_members;
    std::unordered_map<LabelT, uint32_t> _label_to_cluster_id;
    uint32_t _label_cluster_target_count{256};
    uint32_t _label_cluster_probe_count{4};
    uint32_t _label_cluster_kmeans_reps{8};

    // 【新增成员 - 中文说明】相关性统计：标签出现次数与标签对共现次数（用于增量更新）
    std::unordered_map<LabelT, uint64_t> _label_occurrence_count; // count(label)
    // cnt(labelA,labelB)：剪枝后图中“拓扑边”产生的连接次数（对称累计）
    std::unordered_map<LabelT, std::unordered_map<LabelT, uint64_t>> _label_pair_edge_count;

    // Indexing parameters
    uint32_t _indexingQueueSize;
    uint32_t _indexingRange;
    uint32_t _indexingMaxC;
    float _indexingAlpha;
    uint32_t _indexingThreads;

    // 查询时扩展的K
    uint32_t _num_correlated_labels_to_expand{0};

    // 【新增成员 - 中文说明】标签频率数组与Otsu门槛值（门槛值是“频率值”，用于比较 freq<=threshold）
    std::vector<uint32_t> _label_frequency;
    uint32_t _label_frequency_otsu_threshold = 0;

    // 标签元数据懒更新：累计插入+删除次数
    std::atomic<uint64_t> _pending_label_update_ops{0};
    float _filter_lazy_update_ratio{0.01f};
    uint32_t _filter_lazy_update_min_ops{128};
    uint64_t _pending_event_seq{0};
    tsl::robin_set<LabelT> _dirty_labels;
    std::unordered_map<LabelT, int64_t> _pending_label_frequency_delta;
    std::unordered_map<LabelT, std::vector<float>> _pending_label_centroid_sum_delta;
    std::unordered_map<LabelT, int64_t> _pending_label_centroid_count_delta;

    // 新标签在批量刷新前使用临时事件序列进行暴力扫描查询
    tsl::robin_set<LabelT> _pending_new_labels;
    std::unordered_map<LabelT, std::vector<PendingLabelEvent>> _pending_new_label_events;

    // label -> 活跃tag集合（用于局部重选medoid）
    std::unordered_map<LabelT, tsl::robin_set<TagT>> _label_to_active_tags;
    // 标签 -> 活跃向量 location 集合（用于低频标签直接爆搜）
    std::unordered_map<LabelT, tsl::robin_set<uint32_t>> _filter_inverted_index;

    // Query scratch data structures
    ConcurrentQueue<InMemQueryScratch<T> *> _query_scratch;

    // Flags for PQ based distance calculation
    bool _pq_dist = false;
    bool _use_opq = false;
    size_t _num_pq_chunks = 0;
    // REFACTOR
    // uint8_t *_pq_data = nullptr;
    std::shared_ptr<QuantizedDistance<T>> _pq_distance_fn = nullptr;
    std::shared_ptr<AbstractDataStore<T>> _pq_data_store = nullptr;
    bool _pq_generated = false;
    FixedChunkPQTable _pq_table;

    //
    // Data structures, locks and flags for dynamic indexing and tags
    //

    // lazy_delete removes entry from _location_to_tag and _tag_to_location. If
    // _location_to_tag does not resolve a location, infer that it was deleted.
    tsl::sparse_map<TagT, uint32_t> _tag_to_location;
    natural_number_map<uint32_t, TagT> _location_to_tag;

    // _empty_slots has unallocated slots and those freed by consolidate_delete.
    // _delete_set has locations marked deleted by lazy_delete. Will not be
    // immediately available for insert. consolidate_delete will release these
    // slots to _empty_slots.
    natural_number_set<uint32_t> _empty_slots;
    std::unique_ptr<tsl::robin_set<uint32_t>> _delete_set;

    bool _data_compacted = true;    // true if data has been compacted
    bool _is_saved = false;         // Checking if the index is already saved.
    bool _conc_consolidate = false; // use _lock while searching

    // Acquire locks in the order below when acquiring multiple locks
    std::shared_timed_mutex // RW mutex between save/load (exclusive lock) and
        _update_lock;       // search/inserts/deletes/consolidate (shared lock)
    std::shared_timed_mutex // Ensure only one consolidate or compact_data is
        _consolidate_lock;  // ever active
    std::shared_timed_mutex // RW lock for _tag_to_location,
        _tag_lock;          // _location_to_tag, _empty_slots, _nd, _max_points, _label_to_start_id
    std::shared_timed_mutex // RW Lock on _delete_set and _data_compacted
        _delete_lock;       // variable

    // Per node lock, cardinality=_max_points + _num_frozen_points
    std::vector<non_recursive_mutex> _locks;

    static const float INDEX_GROWTH_FACTOR;
};
} // namespace diskann
