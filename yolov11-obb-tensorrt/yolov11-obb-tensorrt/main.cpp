#include "yolov11-obb-tensorrt.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>

using namespace nvinfer1;

#include <vector>
#include <algorithm>

#include <windows.h>
#include <chrono>


int main()
{
    const std::vector<std::string> CLASSES = {
        "plane", "ship", "storage tank", "baseball diamond", "tennis court",
        "basketball court", "ground track field", "harbor", "bridge",
        "large vehicle", "small vehicle", "helicopter", "roundabout",
        "soccer ball field", "swimming pool"
    };

    const std::vector<cv::Scalar> COLORS = {
        {255, 0, 0},     {0, 255, 0},     {0, 0, 255},     {255, 255, 0},   {0, 255, 255},
        {255, 0, 255},   {128, 128, 0},   {128, 0, 128},   {0, 128, 128},   {192, 192, 192},
        {128, 128, 128}, {64, 64, 64},    {255, 165, 0},   {75, 0, 130},    {238, 130, 238}
    };

    YOLOv11_OBB_TRT inferencer;
    inferencer.init(8, 16, "yolo11n-obb-fp16.trt", 15, 16, 1024, 1024, 256, 256, 0.25, 0.45, true, true, true, "_obb_results", CLASSES, COLORS);

    //std::string input_folder = "D:/Workspace/YOLOv11_OBB_TensorRT_Projects/DOTA/test/images";
    std::string input_folder = "C:/test/images";

    std::string output_folder = input_folder + inferencer._result_folder_prefix;
    CreateDirectoryA(output_folder.c_str(), NULL);
    std::vector<std::string> image_paths;
    cv::glob(input_folder + "/*.png", image_paths, false);

    std::cout << "image_paths.size():" << image_paths.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    double tensorrt_buffer_count = 0.0f;
    double image_buffer_count = 0.0f;
    int tile_count = 0;
    for (int i = 0;i < image_paths.size();++i) {
        std::cout << "enqueue: " << image_paths[i] << std::endl;
        inferencer.enqueue(image_paths[i]);
    }

    for (int i = 0;i < image_paths.size();++i) {
        ImageData im = inferencer.dequeue();
        tensorrt_buffer_count += im.batch_buffer_size;
        image_buffer_count += im.input_buffer_size;
        tile_count += im.tile_count;

        cv::imshow("results", im.draw_img_cpu);
        cv::waitKey(1);

        /*
        std::string image_path = im.image_path;
        std::cout << "Image Path: " << image_path << std::endl;
        for (int j = 0;j < im.final_results.size();++j) {
            float x = im.final_results[j].x;
            float y = im.final_results[j].y;
            float w = im.final_results[j].w;
            float h = im.final_results[j].h;
            float a = im.final_results[j].angle;
            float s = im.final_results[j].score;
            int c = im.final_results[j].class_id;
            std::cout << "x: " << x << ", y: " << y << ", w: " << w << ", h: " << h
                << ", angle: " << a << ", score: " << s << ", class: " << CLASSES[c] << std::endl;
        }
        */
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    float tensorrt_buffer_rate = (tensorrt_buffer_count / (1000 * 1000)) / (duration.count() / 1000.0);
    float image_buffer_rate = (image_buffer_count / (1000 * 1000)) / (duration.count() / 1000.0);
    std::cout << "Execution Time: " << duration.count() << " ms ";
    std::cout << "TensorRT Buffer Rate: " << tensorrt_buffer_rate << " MB/s ";
    std::cout << "Image Buffer Rate: " << image_buffer_rate << " MB/s ";
    std::cout << "Image Rate: " << (image_paths.size()) / ((duration.count() / 1000.0)) << " FPS" << std::endl;
    std::cout << "Tile Rate: " << (tile_count) / ((duration.count() / 1000.0)) << " FPS" << std::endl;

    inferencer.terminate();

    /*
    //inferencer.enqueue("D:/Workspace/YOLOv11_OBB_TensorRT_Projects/DOTA/test/images/P0006.png");
    inferencer.enqueue("boats.jpg");
    inferencer.dequeue();
    */

    /*
    inferencer.init(1, 1, "yolo11n-obb-fp16.trt", 15, 1, 1024, 1024, 256, 256, 0.25, 0.45, true, false, false);
    inferencer.enqueue("boats.jpg");
    ImageData im = inferencer.dequeue();
    cv::imshow("results", im.draw_img_cpu);
    cv::waitKey(0);
    */


    /*
    ImageData im = inferencer.dequeue();

    std::cout << "img_h:" << im.batch_info.img_h << std::endl;
    std::cout << "img_w:" << im.batch_info.img_w << std::endl;
    std::cout << "batch_size:" << im.batch_info.batch_size << std::endl;
    std::cout << "batch_h:" << im.batch_info.batch_h << std::endl;
    std::cout << "batch_w:" << im.batch_info.batch_w << std::endl;
    std::cout << "offset_h:" << im.batch_info.offset_h << std::endl;
    std::cout << "offset_w:" << im.batch_info.offset_w << std::endl;
    std::cout << "overlap_h:" << im.batch_info.overlap_h << std::endl;
    std::cout << "overlap_w:" << im.batch_info.overlap_w << std::endl;

    cv::Mat input_gpu;
    im.input_img_gpu.download(input_gpu);
    cv::imshow("input", im.input_img_cpu);
    cv::imshow("input_gpu", input_gpu);
    cv::waitKey(0);
    */

    /*
    ImageData im = inferencer.dequeue();
    cv::Mat img = im.input_img_cpu;
    for (const auto& det : im.final_results) {
        draw_rotated_box(img, det, CLASSES, COLORS);
    }
    cv::imshow("results", img);
    cv::imwrite("boats_result.jpg", img);
    cv::waitKey(0);
    */



    //inferencer.terminate();
    system("pause");
}











/*
struct OBBDetection {
    float x, y, w, h, angle;
    float conf;
    int class_id;
};
*/





// Logger for TensorRT
/*
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Filter out info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

static Logger gLogger;
*/

inline void checkCuda(cudaError_t ret, const char* msg = "") {
    if (ret != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " " << cudaGetErrorString(ret) << std::endl;
        exit(1);
    }
}

// Load serialized engine from file
std::vector<char> loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to load engine file");
    return std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

#include <opencv2/opencv.hpp>


int run_test() {
    std::string enginePath = "yolo11n-obb-fp16.trt";
    std::string imagePath = "boats1024.jpg";

    const std::vector<std::string> CLASSES = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool"
    };

    const std::vector<cv::Scalar> COLORS = {
        {255, 0, 0},     {0, 255, 0},     {0, 0, 255},     {255, 255, 0},   {0, 255, 255},
        {255, 0, 255},   {128, 128, 0},   {128, 0, 128},   {0, 128, 128},   {192, 192, 192},
        {128, 128, 128}, {64, 64, 64},    {255, 165, 0},   {75, 0, 130},    {238, 130, 238}
    };

    // Load engine
    auto engineData = loadEngine(enginePath);

    // TensorRT runtime
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();

    // Input and output binding info
    //int inputIndex = engine->getBindingIndex("images");  // or use 0 if unsure
    //int outputIndex = engine->getBindingIndex(engine->getBindingName(1));
    int inputIndex = 0;
    int outputIndex = 1;

    //auto inputDims = engine->getBindingDimensions(inputIndex);
    //auto outputDims = engine->getBindingDimensions(outputIndex);

    auto inputDims = engine->getTensorShape("images");
    auto outputDims = engine->getTensorShape("output0");

    //int inputH = inputDims.d[2];
    //int inputW = inputDims.d[3];
    //int inputC = inputDims.d[1];
    //int batchSize = inputDims.d[0];

    int inputH = 1024;
    int inputW = 1024;
    int inputC = 3;
    int batchSize = 1;
    int NC = 15;



    inputDims.d[0] = batchSize;
    inputDims.d[1] = inputC;
    inputDims.d[2] = inputH;
    inputDims.d[3] = inputW;

    context->setInputShape("images", inputDims);

    std::cout << "Input:[" << inputDims.d[0] << "," << inputDims.d[1] << "," << inputDims.d[2] << "," << inputDims.d[3] << "]" << std::endl;


    outputDims = context->getTensorShape("output0");

    std::cout << "Output:[" << outputDims.d[0] << "," << outputDims.d[1] << "," << outputDims.d[2] <<"]" << std::endl;


    size_t inputSize = batchSize * inputC * inputH * inputW * sizeof(half);
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i)
        outputSize *= outputDims.d[i];
    outputSize *= sizeof(half);

    // Preprocess image
    cv::Mat img = cv::imread(imagePath);
    half* inputDataFP16 = (half*)malloc(inputH * inputW * 3 * sizeof(half));
    cv::Mat resized(inputH, inputW, CV_16FC3, inputDataFP16);
    cv::resize(img, resized, cv::Size(inputW, inputH));
    resized.convertTo(resized, CV_16F, 1.0 / 255);

    std::vector<half> inputTensor(batchSize * inputC * inputH * inputW);
    std::vector<half> outputTensor(outputSize / sizeof(half));

    half* resized_ptr = resized.ptr<half>();

    // Convert to CHW
    int index = 0;
    for (int c = 0; c < inputC; ++c) {
        for (int h = 0; h < inputH; ++h) {
            for (int w = 0; w < inputW; ++w) {
                //inputTensor[index++] = resized.at<cv::Vec3f>(h, w)[c];
                //inputTensor[index++] = *((half*)(resized.data[(h*inputW*inputC+w*inputC+c)*2]));
                int idx = h * inputW * inputC + w * inputC + c;
                inputTensor[index++] = resized_ptr[idx];
            }
        }
    }

    // Allocate GPU memory
    void* buffers[2];
    checkCuda(cudaMalloc(&buffers[inputIndex], inputSize), "input malloc");
    checkCuda(cudaMalloc(&buffers[outputIndex], outputSize), "output malloc");

    // Copy input to device
    checkCuda(cudaMemcpy(buffers[inputIndex], inputTensor.data(), inputSize, cudaMemcpyHostToDevice), "input memcpy");

    // Inference
    context->executeV2(buffers);

    // Copy output from device
    checkCuda(cudaMemcpy(outputTensor.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost), "output memcpy");

    /*
    std::cout << "Inference done. First 5 output values:\n";
    for (int i = 0; i < batchSize; ++i) { //batch
        for (int j = 0; j < outputDims.d[2]; ++j) { //boxes index
            std::cout << i << "," << j << ":" << std::endl;
            for (int k = 0; k < NC + 5; ++k) { //element index
                int index = i*(NC+5)*outputDims.d[2] + k * outputDims.d[2] + j;
                std::cout << outputTensor[index] << " ";
            }
            std::cout << std::endl;
        }
    }
    */

    // Cleanup
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    std::vector<float> outputTensor_fp32(outputSize / sizeof(half));
    cv::Size size(batchSize, outputTensor.size()/ batchSize);
    cv::Mat float16_out(size, CV_16F, outputTensor.data());
    cv::Mat float32_out(size, CV_32F, outputTensor_fp32.data());
    float16_out.convertTo(float32_out, CV_32F);

    auto detections = non_max_suppression_obb(outputTensor_fp32.data(), outputDims.d[2], NC, 0.25, 0.45);
    for (const auto& d : detections) {
        //std::cout << "Class " << d.class_id << ": (" << d.x << ", " << d.y << ", " << d.w << ", " << d.h
        //    << ", angle " << d.angle << "), confidence: " << d.conf << "\n";

        for (const auto& det : detections) {
            draw_rotated_box(img, det, CLASSES, COLORS);
        }

    }

    cv::imshow("Detection", img);
    cv::waitKey(0);

    return 0;
}


/*
int main() {
    run_test();
}
*/
