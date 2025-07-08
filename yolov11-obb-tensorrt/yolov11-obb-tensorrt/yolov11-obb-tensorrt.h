#ifndef YOLOV11_OBB_TRT_H
#define YOLOV11_OBB_TRT_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <string>
#include <fstream>
#include <iostream>

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

#include <windows.h>

#define max_img_size 8192
#define max_object_count_per_tile_before_nms 16384
#define max_object_count_per_tile_after_nms 512

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;
    ~ThreadSafeQueue() = default;

    // 禁止複製
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // 加入資料
    void push(const T& value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(value);
        }
        cond_var_.notify_one();  // 喚醒等待中的執行緒
    }

    void push(T&& value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
        }
        cond_var_.notify_one();
    }

    // 取出資料（阻塞直到有資料）
    void wait_and_pop(T& result) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this] { return !queue_.empty(); });
        result = std::move(queue_.front());
        queue_.pop();
    }

    // 嘗試取出資料（非阻塞）
    bool try_pop(T& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        result = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // 取得 queue 大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // 檢查是否為空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_var_;
};

struct OBBDetection {
    float x, y, w, h, angle;
    float score;
    int class_id;
    bool keep;
};
struct EngineData {
    std::vector<int> image_idxs;
    std::vector<int> tile_idxs;
    std::vector<std::vector<OBBDetection>> resultss;

    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    int input_bchw_size;
    half* input_bchw_gpu_ptr; //bchw=batch-channel-height-width
    int output_bbe_size;
    half* output_bbe_gpu_ptr; //bbe=batch-boxes-element

    float* post_process_data_gpu_ptr; //batch-boxes-(x-y-w-h-a-conf-cls-keep)
    int post_process_data_size;

    int* post_process_final_count_cpu_ptr; //batch in CPU reuse for before NMS and after NMS

    int* post_process_count_gpu_ptr; //batch in GPU for before NMS
    int* post_process_count_cpu_ptr; //batch in CPU for before NMS
    int* post_process_final_count_gpu_ptr; //batch in GPU for after NMS
    int* post_process_count_zeros_gpt_ptr; //batch <fill zero>
    int post_process_count_size;

    float* post_process_final_gpu_ptr; //batch-boxes-(x-y-w-h-a-score-cls) in GPU
    float* post_process_final_cpu_ptr; //batch-boxes-(x-y-w-h-a-score-cls) in CPU
    int post_process_final_size;

    cudaStream_t* stream;
    //cv::cuda::Stream cvStream;
};

struct BatchData {
    int img_h;
    int img_w;
    int batch_h;
    int batch_w;
    int offset_h;
    int offset_w;
    int overlap_h;
    int overlap_w;
    int batch_size;
};

struct ImageData {
    int image_idx;
    int tile_count;
    //int tile_done;
	//std::atomic<int> tile_done_counter{ 0 };
    std::string image_path;
    float resize_ratio;
    float input_buffer_size;
    float batch_buffer_size;

    cv::Mat original_img_cpu;
    cv::Mat input_img_cpu;
    cv::Mat draw_img_cpu;

    cv::cuda::GpuMat input_img_gpu;
    uchar* input_img_gpu_ptr;

    BatchData batch_info;

    std::vector<std::vector<OBBDetection>> resultss;
    std::vector<OBBDetection> final_results;
};

struct TileData {
    int image_idx;
    int tile_idx;

    std::vector<OBBDetection> results;
};



// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Filter out info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};
static Logger gLogger;

class YOLOv11_OBB_TRT {
public:
    YOLOv11_OBB_TRT();
    ~YOLOv11_OBB_TRT();


    bool init(int engine_threads, int image_threads, const char* engine_path, 
        int num_classes, int batch_size, int tile_h, int tile_w, int min_overlap_h, int min_overlap_w, 
        float conf_thres, float iou_thres, bool draw_bboxes, bool output_draw, bool output_csv, std::string result_folder_prefix, 
        std::vector<std::string> CLASSES, std::vector<cv::Scalar> COLORS);
    void terminate();
    void enqueue(std::string image_path);
    ImageData dequeue();
    std::string _result_folder_prefix;

private:
    bool inference(half* input_bchw_gpu, half* outpu_bbe_gpu, cudaStream_t stream);
    BatchData calculate_batch_info(cv::Mat img);
    bool pre_process(ImageData* im, int img_start_index, int img_end_index, EngineData* eng, int bchw_start_index, int bchw_end_index);
    bool post_process(EngineData& eng);
    void view_post_process_result(EngineData& eng);
    void tile2tile_nms(std::vector<OBBDetection>& rs1, std::vector<OBBDetection>& rs2);
    void deleteImageData(ImageData& im);
    void deleteEngineData(EngineData* eng);
    void post_process_cpu(half* output_bbe_gpu_ptr, int output_bbe_size, int num_classes, int batch_size, const half* input_bchw_gpu_ptr, int B, int C, int H, int W, cudaStream_t stream);

    // -- engine parameter --
    int _engine_threads;
    int _image_threads;
    std::string _engine_path;
    int _num_classes;
    int _batch_size;
    int _tile_h;
    int _tile_w;
    int _min_overlap_h;
    int _min_overlap_w;
    std::atomic<int> _image_counter{0};
    int _group_size;
    float _conf_thres;
    float _iou_thres;
    bool _draw_bboxes;
    bool _output_draw;
    bool _output_csv;
    std::vector<std::string> _CLASSES;
    std::vector<cv::Scalar> _COLORS;

    ThreadSafeQueue<std::string> _input_path_queue;

    ThreadSafeQueue<ImageData> _unused_data_queue;
    ThreadSafeQueue<ImageData> _input_data_queue;
    std::unordered_map<int, ImageData> _result_data_collector;
    std::unordered_map<int, std::atomic<int>> _image_tile_counters;
	//std::vector<std::atomic<int>> _image_tile_counters;
    
    std::vector<EngineData> engs;
    ThreadSafeQueue<int> _unused_engine_queue;
    ThreadSafeQueue<int> _tensorrt_engine_queue;
    ThreadSafeQueue<int> _postprocess_engine_queue;
    
    ThreadSafeQueue<TileData> _tile_data_queue;
    ThreadSafeQueue<ImageData> _output_data_queue;
    ThreadSafeQueue<ImageData> _result_data_queue;

    bool _end_worker;

    void decode_image_worker(); // image paht (string) -> OpenCV Mat (image) -> OpenCV GpuMat (image) -> input data (image)
    void pre_process_worker(); // input data (image) -> inference data (engine) / prepare result collector (image)
    void tensorrt_worker(); // inference data (engine) -> TensorRT (engine) -> prediction data (engine)
    void post_process_worker(); // prediction data (engine) -> tile NMS (GPU) -> tile result data (tile)
    void result_collect_worker(); // tile result data (tile) -> result data collector (image)
    void output_worker(); // result data collector (image) -> doing whole image NMS (CPU) -> output data (image) 

    std::vector<std::thread> _decode_threads;
    std::vector<std::thread> _pre_process_threads;
    std::vector<std::thread> _tensorrt_threads;
    std::vector<std::thread> _post_process_threads;
    std::vector<std::thread> _result_collect_threads;
    std::vector<std::thread> _output_threads;

};



std::vector<OBBDetection> non_max_suppression_obb(const float* output, int num_boxes, int num_classes, float conf_thres, float iou_thres);
void draw_rotated_box(cv::Mat& img, const OBBDetection& det, const std::vector<std::string>& class_names, const std::vector<cv::Scalar>& colors);



#endif  // YOLOV11_OBB_TRT_H