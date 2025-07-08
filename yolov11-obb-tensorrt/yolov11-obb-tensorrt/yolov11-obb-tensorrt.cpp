#include "yolov11-obb-tensorrt.h"


YOLOv11_OBB_TRT::YOLOv11_OBB_TRT() {}
YOLOv11_OBB_TRT::~YOLOv11_OBB_TRT() { terminate(); }

bool YOLOv11_OBB_TRT::init(int engine_threads, int image_threads,
    const char* engine_path,
    int num_classes,
    int batch_size,
    int tile_h,
    int tile_w,
    int min_overlap_h,
    int min_overlap_w,
    float conf_thres,
    float iou_thres,
    bool draw_bboxes,
    bool output_draw,
    bool output_csv,
    std::string result_folder_prefix,
    std::vector<std::string> CLASSES,
	std::vector<cv::Scalar> COLORS
    )
{
    _engine_threads = engine_threads;
    _image_threads = image_threads;
    _engine_path = engine_path;
    _num_classes = num_classes;
    _batch_size = batch_size;
    _tile_h = tile_h;
    _tile_w = tile_w;
    _min_overlap_h = min_overlap_h;
    _min_overlap_w = min_overlap_w;
    _conf_thres = conf_thres;
    _iou_thres = iou_thres;
    _draw_bboxes = draw_bboxes;
    _output_draw = output_draw;
    _output_csv = output_csv;
    _result_folder_prefix = result_folder_prefix;
    _CLASSES = CLASSES;
    _COLORS = COLORS;

    std::cout << "engine_path:" << engine_path << std::endl;

    // Load serialized engine from file
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) throw std::runtime_error("Failed to load engine file");
    std::vector<char> engineData = std::vector<char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::cout << "engine readed." << std::endl;
    for (int i = 0;i < engine_threads;++i) {
        std::cout << "Loading Engine for Thread #" << i << std::endl;

        EngineData engine_thread_data;

        engine_thread_data.runtime = nvinfer1::createInferRuntime(gLogger);
        if (!engine_thread_data.runtime) {
            std::cout << "[ERROR] Failed to create TensorRT runtime." << std::endl;
            return false;
        }

        engine_thread_data.engine = engine_thread_data.runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine_thread_data.engine) {
            std::cout << "[ERROR] Failed to deserialize engine." << std::endl;
            return false;
        }

        engine_thread_data.context = engine_thread_data.engine->createExecutionContext();
        if (!engine_thread_data.context) {
            std::cout << "[ERROR] Failed to create execution context." << std::endl;
            return false;
        }

        auto inputDims = engine_thread_data.engine->getTensorShape("images");
        auto outputDims = engine_thread_data.engine->getTensorShape("output0");

        inputDims.d[0] = this->_batch_size;
        inputDims.d[1] = 3;
        inputDims.d[2] = this->_tile_h;
        inputDims.d[3] = this->_tile_w;

        engine_thread_data.context->setInputShape("images", inputDims);

        std::cout << "Input:[" << inputDims.d[0] << "," << inputDims.d[1] << "," << inputDims.d[2] << "," << inputDims.d[3] << "]" << std::endl;
        outputDims = engine_thread_data.context->getTensorShape("output0");
        std::cout << "Output:[" << outputDims.d[0] << "," << outputDims.d[1] << "," << outputDims.d[2] << "]" << std::endl;
        
        cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
        engine_thread_data.stream = stream;
        cudaStreamCreate(engine_thread_data.stream);
        //(*engine_thread_data).cvStream = cv::cuda::StreamAccessor::wrapStream(*(*engine_thread_data).stream);

        engine_thread_data.input_bchw_size = inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * inputDims.d[3] * sizeof(half);
        engine_thread_data.output_bbe_size = outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(half);

        cudaMalloc(&(engine_thread_data.input_bchw_gpu_ptr), engine_thread_data.input_bchw_size);
        cudaMalloc(&(engine_thread_data.output_bbe_gpu_ptr), engine_thread_data.output_bbe_size);

        engine_thread_data.post_process_data_size = outputDims.d[0] * max_object_count_per_tile_before_nms * 8 * sizeof(float);
        cudaMalloc(&(engine_thread_data.post_process_data_gpu_ptr), engine_thread_data.post_process_data_size);

        engine_thread_data.post_process_count_size = outputDims.d[0] * sizeof(int);
        cudaMalloc(&(engine_thread_data.post_process_count_gpu_ptr), engine_thread_data.post_process_count_size);
        cudaMalloc(&(engine_thread_data.post_process_final_count_gpu_ptr), engine_thread_data.post_process_count_size);
        cudaMalloc(&(engine_thread_data.post_process_count_zeros_gpt_ptr), engine_thread_data.post_process_count_size);

        engine_thread_data.post_process_count_cpu_ptr = (int*)malloc(engine_thread_data.post_process_count_size);
        //engine_thread_data.post_process_count_cpu_ptr = new int[engine_thread_data.post_process_count_size / sizeof(int)];
        engine_thread_data.post_process_final_count_cpu_ptr = (int*)malloc(engine_thread_data.post_process_count_size);
        //engine_thread_data.post_process_final_count_cpu_ptr = new int[engine_thread_data.post_process_count_size / sizeof(int)];
        memset(engine_thread_data.post_process_final_count_cpu_ptr, 0, engine_thread_data.post_process_count_size);
        cudaMemcpy(engine_thread_data.post_process_count_zeros_gpt_ptr, engine_thread_data.post_process_final_count_cpu_ptr, engine_thread_data.post_process_count_size, cudaMemcpyHostToDevice);

        engine_thread_data.post_process_final_size = outputDims.d[0] * max_object_count_per_tile_after_nms * 7 * sizeof(float);
        cudaMalloc(&(engine_thread_data.post_process_final_gpu_ptr), engine_thread_data.post_process_final_size);
        engine_thread_data.post_process_final_cpu_ptr = (float*)malloc(engine_thread_data.post_process_final_size);
        //engine_thread_data.post_process_final_cpu_ptr = new float[engine_thread_data.post_process_final_size / sizeof(int)];

        _group_size = outputDims.d[2];

        engine_thread_data.context->setTensorAddress("images", engine_thread_data.input_bchw_gpu_ptr);
        engine_thread_data.context->setTensorAddress("output0", engine_thread_data.output_bbe_gpu_ptr);

        engine_thread_data.image_idxs.resize(batch_size);
        engine_thread_data.tile_idxs.resize(batch_size);
        engine_thread_data.resultss.resize(batch_size);

        engs.push_back(engine_thread_data);
        _unused_engine_queue.push(i);
    }

    for (int i = 0; i < image_threads; ++i) {
        ImageData im;
        cudaMalloc(&(im.input_img_gpu_ptr), max_img_size * max_img_size * 3 * sizeof(uchar));
        _unused_data_queue.push(im);
    }

    _end_worker = false;
    _image_counter.store(0);

    for (int i = 0;i < image_threads;++i) {
        std::thread decode_thread([this]() { decode_image_worker(); });
        _decode_threads.push_back(std::move(decode_thread));
    }

    for (int i = 0;i < engine_threads;++i) {
        std::thread pre_process_thread([this]() { pre_process_worker(); });
        _pre_process_threads.push_back(std::move(pre_process_thread));
    }

    for (int i = 0;i < engine_threads;++i) {
        std::thread tensorrt_thread([this]() { tensorrt_worker(); });
        _tensorrt_threads.push_back(std::move(tensorrt_thread));
    }

    for (int i = 0;i < engine_threads;++i) {
        std::thread post_process_thread([this]() { post_process_worker(); });
        _post_process_threads.push_back(std::move(post_process_thread));
    }

    for (int i = 0;i < 1;++i) {
        std::thread result_collect_thread([this]() { result_collect_worker(); });
        _result_collect_threads.push_back(std::move(result_collect_thread));
    }

    for (int i = 0;i < image_threads;++i) {
        std::thread output_thread([this]() { output_worker(); });
        _output_threads.push_back(std::move(output_thread));
    }
}

void YOLOv11_OBB_TRT::deleteImageData(ImageData& im) {
    cudaFree(im.input_img_gpu_ptr);
}

void YOLOv11_OBB_TRT::deleteEngineData(EngineData* eng) {
    cudaFree((*eng).input_bchw_gpu_ptr);
    cudaFree((*eng).output_bbe_gpu_ptr);
    cudaFree((*eng).post_process_data_gpu_ptr);
    //free(eng.post_process_final_count_cpu_ptr);
    //delete eng.post_process_final_count_cpu_ptr;
    cudaFree((*eng).post_process_count_gpu_ptr);
    //free(eng.post_process_count_cpu_ptr);
    //delete eng.post_process_count_cpu_ptr;
    cudaFree((*eng).post_process_final_count_gpu_ptr);
    cudaFree((*eng).post_process_count_zeros_gpt_ptr);
    cudaFree((*eng).post_process_final_gpu_ptr);
    //free(eng.post_process_final_cpu_ptr);
    //delete eng.post_process_final_cpu_ptr;
    if ((*eng).context) {
        (*eng).context = nullptr;
    }
    if ((*eng).engine) {
        (*eng).engine = nullptr;
    }
    if ((*eng).runtime) {
        (*eng).runtime = nullptr;
    }
}

void YOLOv11_OBB_TRT::terminate() {
    _end_worker = true;
    
    //Join all child threads
    for (int i = 0; i < _decode_threads.size(); ++i) {
        _decode_threads[i].join();
    }
    for (int i = 0; i < _pre_process_threads.size(); ++i) {
        _pre_process_threads[i].join();
    }
    for (int i = 0; i < _tensorrt_threads.size(); ++i) {
        _tensorrt_threads[i].join();
    }
    for (int i = 0; i < _post_process_threads.size(); ++i) {
        _post_process_threads[i].join();
    }
    for (int i = 0; i < _result_collect_threads.size(); ++i) {
        _result_collect_threads[i].join();
    }
    for (int i = 0; i < _output_threads.size(); ++i) {
        _output_threads[i].join();
    }

    //Delete ImageData
    while (_unused_data_queue.size() != 0) {
        ImageData im;
        _unused_data_queue.wait_and_pop(im);
        deleteImageData(im);
    }
    while (_input_data_queue.size() != 0) {
        ImageData im;
        _input_data_queue.wait_and_pop(im);
        deleteImageData(im);
    }
    while (_output_data_queue.size() != 0) {
        ImageData im;
        _output_data_queue.wait_and_pop(im);
        deleteImageData(im);
    }

    //Delete EngineData
    while (_unused_engine_queue.size() != 0) {
        //EngineData eng;
        int i;
        _unused_engine_queue.wait_and_pop(i);
        deleteEngineData(&engs[i]);
    }
    while (_tensorrt_engine_queue.size() != 0) {
        //EngineData eng;
        int i;
        _tensorrt_engine_queue.wait_and_pop(i);
        deleteEngineData(&engs[i]);
    }
    while (_postprocess_engine_queue.size() != 0) {
        //EngineData eng;
        int i;
        _postprocess_engine_queue.wait_and_pop(i);
        deleteEngineData(&engs[i]);
    }

    //Clear TileData
    if (_tile_data_queue.size() != 0) {
        _tile_data_queue.empty();
    }
}

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

BatchData YOLOv11_OBB_TRT::calculate_batch_info(cv::Mat img) {
    BatchData batch;

    int height = img.rows;
    int width = img.cols;
    batch.img_h = height;
    batch.img_w = width;

    int batch_h = ceil_div(height, _tile_h);
    int offset_h;
    if (batch_h == 1) {
        offset_h = 0;
    }
    else {
        offset_h = (height-_tile_h) / (batch_h - 1);
    }
    int overlap_h = _tile_h - offset_h;
    while (overlap_h < _min_overlap_h) {
        batch_h += 1;
        offset_h = (height - _tile_h) / (batch_h - 1);
        overlap_h = _tile_h - offset_h;
    }
    batch.batch_h = batch_h;
    batch.offset_h = offset_h;
    batch.overlap_h = overlap_h;

    int batch_w = ceil_div(width, _tile_w);
    int offset_w;
    if (batch_w == 1) {
        offset_w = 0;
    }
    else {
        offset_w = (width-_tile_w) / (batch_w - 1);
    }
    int overlap_w = _tile_w - offset_w;
    while (overlap_w < _min_overlap_w) {
        batch_w += 1;
        offset_w = (width - _tile_w) / (batch_w - 1);
        overlap_w = _tile_w - offset_w;
    }
    batch.batch_w = batch_w;
    batch.offset_w = offset_w;
    batch.overlap_w = overlap_w;

    batch.batch_size = batch_h * batch_w;

    /*
    std::cout << "batch_size:" << batch.batch_size << std::endl;
    std::cout << "batch_h:" << batch_h << std::endl;
    std::cout << "batch_w:" << batch_w << std::endl;
    std::cout << "offset_h:" << offset_h << std::endl;
    std::cout << "offset_w:" << offset_w << std::endl;
    std::cout << "overlap_h:" << overlap_h << std::endl;
    std::cout << "overlap_w:" << overlap_w << std::endl;
    */

    return batch;
}

void YOLOv11_OBB_TRT::enqueue(std::string image_path) {
    _input_path_queue.push(image_path);
}

ImageData YOLOv11_OBB_TRT::dequeue() {
    ImageData im;
    _result_data_queue.wait_and_pop(im);
    return im;
}

void YOLOv11_OBB_TRT::decode_image_worker() {
    bool success;
    std::string image_path;
    while (1) {
        if (_end_worker) break;
        success = _input_path_queue.try_pop(image_path);
        if (!success) {
            Sleep(100);
            continue;
        }
        ImageData im;
        _unused_data_queue.wait_and_pop(im);

        //std::cout << "Start processing: " << im.image_path << std::endl;

        cv::Mat new_img = cv::imread(image_path, cv::IMREAD_COLOR);

        if (new_img.empty()) {
            std::cout << "Failed to load image: " << image_path << std::endl;
            return;
        }

        
        im.original_img_cpu = new_img.clone();

        int original_w = new_img.cols;
        int original_h = new_img.rows;
        float scale = 1.0f;
        if (original_w < _tile_w || original_h < _tile_h) {
            float scale_w = static_cast<float>(_tile_w) / original_w;
            float scale_h = static_cast<float>(_tile_h) / original_h;
            scale = scale_w > scale_h ? scale_w : scale_h;
        }
        if (original_w > max_img_size || original_h > max_img_size) {
            float scale_w = static_cast<float>(max_img_size) / original_w;
            float scale_h = static_cast<float>(max_img_size) / original_h;
            scale = scale_w < scale_h ? scale_w : scale_h;
        }
        float final_scale = scale;
        //std::cout << "final_scale:" << final_scale << std::endl;
        if (final_scale != 1.0f) {
            int new_w = static_cast<int>(original_w * final_scale);
            int new_h = static_cast<int>(original_h * final_scale);
            std::cout << "resize due to size not meet requirement: " << "old_h:" << original_h << " old_w:" << original_w << "new_h:" << new_h << " new_w:" << new_w << std::endl;
            cv::resize(new_img, new_img, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        }
        im.resize_ratio = final_scale;

        //cv::imshow("new_img", new_img);
        //cv::waitKey(100);
        

        im.input_img_cpu = new_img;
        im.image_path = image_path;
        int height = im.input_img_cpu.rows;
        int width = im.input_img_cpu.cols;
        int channel = im.input_img_cpu.channels();
        int img_size = height * width * channel;
        
        if (img_size > max_img_size * max_img_size * 3) {
            std::cout << "Error: img_size larger than max_img_size: "<< image_path << std::endl;
            _unused_data_queue.push(im);
            continue;
        }
        

        im.input_img_gpu = cv::cuda::GpuMat(im.input_img_cpu.rows, im.input_img_cpu.cols, CV_8UC3, im.input_img_gpu_ptr);
        im.input_img_gpu.upload(im.input_img_cpu);
        im.batch_info = calculate_batch_info(im.input_img_cpu);
        im.tile_count = im.batch_info.batch_size;
        //im.tile_done = 0;
        //im.tile_done_counter.store(0);

        im.input_buffer_size = height * width * 3 * sizeof(uchar);
        im.batch_buffer_size = im.batch_info.batch_size * 3 * _tile_h * _tile_w * sizeof(uchar);

        //im.resultss.clear();
        im.resultss.resize(im.tile_count);
        for (int i = 0;i < im.tile_count;++i) {
			im.resultss[i].clear();
        }
        im.image_idx = _image_counter.fetch_add(1);
        //_image_tile_counters.push_back(std::atomic<int>(0));
        //std::cout << "set image_idx:" << im.image_idx << " tile_count=" << im.tile_count << std::endl;

        std::cout << "image_idx:" << im.image_idx << " tile_count=" << im.tile_count << " path=" << im.image_path << std::endl;

        _input_data_queue.push(im);
    }
    return;
}

void YOLOv11_OBB_TRT::pre_process_worker() {
    bool success;
    ImageData im;
    int eng_idx;
    EngineData* eng;
    int remaining_img_batch = 0;
    int remaining_engine_batch;
    while (1) {
        if (_end_worker) break;
        _unused_engine_queue.wait_and_pop(eng_idx);
		eng = &engs[eng_idx];
        remaining_engine_batch = _batch_size;

        //std::cout << "remaining_engine_batch:" << remaining_engine_batch << std::endl;

        for (int i = 0; i < (*eng).image_idxs.size(); ++i) {
            (*eng).image_idxs[i] = -1;
            (*eng).tile_idxs[i] = -1;
        }

        while (1) {
            if (_end_worker) break;
            
            if (remaining_img_batch == 0) { // next img
                success = _input_data_queue.try_pop(im);
                //std::cout << "_input_data_queue.try_pop(im):" << success << std::endl;
                if (!success) { // if no next image
                    if (remaining_engine_batch != _batch_size) { // doing now if no next image to concat in batch
                        break;
                    }
                    Sleep(100);
                    continue;
                }
                remaining_img_batch = im.batch_info.batch_size;
            }

            int img_start_index = im.batch_info.batch_size - remaining_img_batch;
            int img_end_index = img_start_index + min(remaining_img_batch, remaining_engine_batch);
            int pre_process_batch_count = img_end_index - img_start_index;
            int bchw_start_index = _batch_size - remaining_engine_batch;
            int bchw_end_index = bchw_start_index + pre_process_batch_count;

            pre_process(&im, img_start_index, img_end_index, eng, bchw_start_index, bchw_end_index);

            for (int i = bchw_start_index, j = img_start_index; i < bchw_end_index; ++i, ++j) {
                (*eng).image_idxs[i] = im.image_idx;
                (*eng).tile_idxs[i] = j;
            }
            //std::cout << "img #" << im.image_idx << ":[" << img_start_index << ":" << img_end_index << "]" << "eng:[" << bchw_start_index << ":" << bchw_end_index << "]" << std::endl;

            remaining_img_batch -= pre_process_batch_count;
            remaining_engine_batch -= pre_process_batch_count;

            
            if (remaining_img_batch == 0) {
                _result_data_collector[im.image_idx] = im;
                _image_tile_counters[im.image_idx].store(0);
            }
            

            if (remaining_engine_batch == 0) {
				//cudaStreamSynchronize(*eng.stream);
                //_tensorrt_engine_queue.push(eng_idx);
                break;
            }
        }
        if(_end_worker) _unused_engine_queue.push(eng_idx);
        else _tensorrt_engine_queue.push(eng_idx);
    }
    return;
}

void visualizeBCHWFromGPU(const half* input_bchw_gpu_ptr, int B, int C, int H, int W, cudaStream_t stream) {
    if (C != 3) {
        std::cout << "Only C == 3 supported for RGB visualization!" << std::endl;
        return;
    }

    size_t num_elements = static_cast<size_t>(B) * C * H * W;

    // 1. Allocate CPU buffers
    std::vector<half> bchw_fp16(num_elements);
    std::vector<float> bchw_fp32(num_elements);

    // 2. Copy from GPU to CPU
    cudaMemcpyAsync(bchw_fp16.data(), input_bchw_gpu_ptr, num_elements * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. Convert to float32
    for (size_t i = 0; i < num_elements; ++i) {
        bchw_fp32[i] = __half2float(bchw_fp16[i]);
    }

    // 4. For each batch, merge C=3 channels into one RGB image
    for (int b = 0; b < B; ++b) {
        std::vector<cv::Mat> channels(3);

        for (int c = 0; c < 3; ++c) {
            cv::Mat channel(H, W, CV_32F);

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t idx = ((size_t)b * C * H * W) + c * H * W + h * W + w;
                    channel.at<float>(h, w) = bchw_fp32[idx];
                }
            }
            channels[c] = channel;
        }

        cv::Mat merged;
        cv::merge(channels, merged);              // 3-channel float image
        cv::normalize(merged, merged, 0, 1, cv::NORM_MINMAX);
        merged.convertTo(merged, CV_8UC3, 255);   // Convert to 8-bit RGB
        cv::cvtColor(merged, merged, cv::COLOR_BGR2RGB);
        std::string winname = "Batch " + std::to_string(b);
        cv::imshow(winname, merged);
    }
    cv::waitKey(0);
}


std::vector<cv::Mat> outputMatFromGpuBCHW(const half* input_bchw_gpu_ptr, int B, int C, int H, int W, cudaStream_t stream) {
    std::vector<cv::Mat> results;
    if (C != 3) {
        std::cout << "Only C == 3 supported for RGB visualization!" << std::endl;
        return results;
    }

    size_t num_elements = static_cast<size_t>(B) * C * H * W;

    // 1. Allocate CPU buffers
    std::vector<half> bchw_fp16(num_elements);
    std::vector<float> bchw_fp32(num_elements);

    // 2. Copy from GPU to CPU
    cudaMemcpyAsync(bchw_fp16.data(), input_bchw_gpu_ptr, num_elements * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 3. Convert to float32
    for (size_t i = 0; i < num_elements; ++i) {
        bchw_fp32[i] = __half2float(bchw_fp16[i]);
    }

    // 4. For each batch, merge C=3 channels into one RGB image
    for (int b = 0; b < B; ++b) {
        std::vector<cv::Mat> channels(3);

        for (int c = 0; c < 3; ++c) {
            cv::Mat channel(H, W, CV_32F);

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    size_t idx = ((size_t)b * C * H * W) + c * H * W + h * W + w;
                    channel.at<float>(h, w) = bchw_fp32[idx];
                }
            }
            channels[c] = channel;
        }

        cv::Mat merged;
        cv::merge(channels, merged);              // 3-channel float image
        cv::normalize(merged, merged, 0, 1, cv::NORM_MINMAX);
        merged.convertTo(merged, CV_8UC3, 255);   // Convert to 8-bit RGB
        cv::cvtColor(merged, merged, cv::COLOR_BGR2RGB);
        //std::string winname = "Batch " + std::to_string(b);
        //cv::imshow(winname, merged);
        results.push_back(merged);
    }
    //cv::waitKey(0);
    return results;
}

void YOLOv11_OBB_TRT::tensorrt_worker() {
    bool success;
    int eng_idx;
    EngineData* eng;
    while (1) {
        if (_end_worker) break;
        success = _tensorrt_engine_queue.try_pop(eng_idx);
        if (!success) {
            Sleep(100);
            continue;
        }
        eng = &engs[eng_idx];
        //visualizeBCHWFromGPU(eng.input_bchw_gpu_ptr, _batch_size, 3, _tile_h, _tile_w, *eng.stream);
        /*
        eng.context->setTensorAddress("images", eng.input_bchw_gpu_ptr);
        cudaMemsetAsync(eng.output_bbe_gpu_ptr, 0, eng.output_bbe_size, *eng.stream);
        eng.context->setTensorAddress("output0", eng.output_bbe_gpu_ptr);
        */
        (*eng).context->enqueueV3(*(*eng).stream);
        //cudaStreamSynchronize(*eng.stream);
        
        /*
        cudaStreamSynchronize(*(*eng).stream);
        void* buffers[2];
        buffers[0] = (void*)(*eng).input_bchw_gpu_ptr;
        buffers[1] = (void*)(*eng).output_bbe_gpu_ptr;
        (*eng).context->executeV2(buffers);
        */
        
        

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        _postprocess_engine_queue.push(eng_idx);
    }
}


// 計算 rotated IoU using OpenCV
float rotated_iou(const OBBDetection& a, const OBBDetection& b) {
    cv::RotatedRect ra(cv::Point2f(a.x, a.y), cv::Size2f(a.w, a.h), a.angle * 180.0f / CV_PI);
    cv::RotatedRect rb(cv::Point2f(b.x, b.y), cv::Size2f(b.w, b.h), b.angle * 180.0f / CV_PI);

    std::vector<cv::Point2f> intersectPts;
    float interArea = static_cast<float>(cv::rotatedRectangleIntersection(ra, rb, intersectPts) == cv::INTERSECT_NONE
        ? 0.0
        : std::fabs(cv::contourArea(intersectPts)));

    float unionArea = a.w * a.h + b.w * b.h - interArea;
    return interArea / (unionArea + 1e-6f);
}

std::vector<OBBDetection> non_max_suppression_obb(
    const float* output, int num_boxes, int num_classes,
    float conf_thres, float iou_thres) {

    std::vector<OBBDetection> dets;

    for (int i = 0; i < num_boxes; ++i) {
        float x = output[i + 0 * (num_boxes)];
        float y = output[i + 1 * (num_boxes)];
        float w = output[i + 2 * (num_boxes)];
        float h = output[i + 3 * (num_boxes)];
        float angle = output[i + (num_classes + 4) * (num_boxes)];  // assume in radians

        // find best class
        float best_conf = 0.0f;
        int best_cls = -1;
        for (int c = 0; c < num_classes; ++c) {
            float conf = output[i + (4 + c) * (num_boxes)];
            //std::cout << "conf:" << c << "=" << conf << " " ;
            if (conf > best_conf) {
                best_conf = conf;
                best_cls = c;
            }
        }
        //std::cout << std::endl;

        if (best_conf > conf_thres) {
            //std::cout << "xywhacc:" << x << " " << y << " " << w << " " << h << " " << angle << " " << best_conf << " " << best_cls << std::endl;
            dets.push_back({ x, y, w, h, angle, best_conf, best_cls });
        }
    }

    // sort by confidence
    std::sort(dets.begin(), dets.end(), [](const OBBDetection& a, const OBBDetection& b) {
        return a.score > b.score;
        });

    // rotated NMS
    std::vector<OBBDetection> results;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        results.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id &&
                rotated_iou(dets[i], dets[j]) > iou_thres) {
                removed[j] = true;
            }
        }
    }

    return results;
}


void draw_label_with_adaptive_text(cv::Mat& img, const std::string& text, int x, int y, int cls_id, const cv::Scalar& bg_color) {
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.25;
    int thickness = 1;
    int baseline = 0;
    int pad = 1;

    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
    int text_width = text_size.width;
    int text_height = text_size.height;

    // Compute brightness to choose black or white text
    double brightness = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0];
    cv::Scalar text_color = (brightness > 128) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

    // Background rectangle
    cv::Point top_left(x, y - text_height - baseline - pad);
    cv::Point bottom_right(x + text_width + pad * 2, y);
    cv::rectangle(img, top_left, bottom_right, bg_color, cv::FILLED);

    // Put text
    cv::putText(img, text,
        cv::Point(top_left.x + pad, bottom_right.y - baseline),
        font_face, font_scale, text_color, thickness, cv::LINE_AA);
}


void draw_rotated_box(cv::Mat& img, const OBBDetection& det, const std::vector<std::string>& class_names, const std::vector<cv::Scalar>& colors) {
    float x = det.x, y = det.y, w = det.w, h = det.h, angle_rad = det.angle;
    float angle_deg = angle_rad * 180.0f / CV_PI;

    cv::RotatedRect rect(cv::Point2f(x, y), cv::Size2f(w, h), angle_deg);
    cv::Point2f vertices[4];
    rect.points(vertices);

    std::vector<cv::Point> contour;
    for (int i = 0; i < 4; ++i)
        contour.push_back(vertices[i]);

    int cls_id = det.class_id;
    float score = det.score;
    std::string label = class_names[cls_id] + ", " + cv::format("%.2f", score);

    // Draw rotated box
    cv::polylines(img, contour, true, colors[cls_id], 1, cv::LINE_AA);

    // Draw label
    draw_label_with_adaptive_text(img, label, static_cast<int>(x), static_cast<int>(y), cls_id, colors[cls_id]);
}

void YOLOv11_OBB_TRT::post_process_cpu(half* output_bbe_gpu_ptr, int output_bbe_size, int num_classes, int batch_size, const half* input_bchw_gpu_ptr, int B, int C, int H, int W, cudaStream_t stream) {
    std::vector<cv::Mat> imgs = outputMatFromGpuBCHW(input_bchw_gpu_ptr, B, C, H, W, stream);

    std::cout << "imgs.size():" << imgs.size() << std::endl;
    
    int outputSize = output_bbe_size / sizeof(half);
    int group_size = outputSize / ((num_classes + 5) * batch_size);

    std::vector<half> outputTensor(outputSize);
    cudaMemcpy(outputTensor.data(), output_bbe_gpu_ptr, outputSize * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<float> outputTensor_fp32(outputSize);
    cv::Size size(B, outputTensor.size() / B);
    cv::Mat float16_out(size, CV_16F, outputTensor.data());
    cv::Mat float32_out(size, CV_32F, outputTensor_fp32.data());
    float16_out.convertTo(float32_out, CV_32F);

    for (int b = 0; b < batch_size; ++b) {
        cv::Mat img = imgs[b];
        //cv::imshow("Orig", img);
        //cv::waitKey(0);

        auto detections = non_max_suppression_obb(&(((float*)(outputTensor_fp32.data()))[b * group_size * (5 + num_classes)]), group_size, num_classes, 0.25, 0.45);
        for (const auto& d : detections) {
            //std::cout << "Class " << d.class_id << ": (" << d.x << ", " << d.y << ", " << d.w << ", " << d.h
            //    << ", angle " << d.angle << "), confidence: " << d.conf << "\n";
            draw_rotated_box(img, d, _CLASSES, _COLORS);

        }

        cv::imshow("Detection", img);
        cv::waitKey(0);
    }
}

void YOLOv11_OBB_TRT::view_post_process_result(EngineData& eng) {
    std::vector<cv::Mat> imgs = outputMatFromGpuBCHW(eng.input_bchw_gpu_ptr, _batch_size, 3, _tile_h, _tile_w, *eng.stream);
    for (int b = 0; b < _batch_size; ++b) {
        cv::Mat img = imgs[b];
        for (const auto& det : eng.resultss[b]) {
            draw_rotated_box(img, det, _CLASSES, _COLORS);
        }
        std::string imshow_window_name = "Batch #" + cv::format("%d", b);
        cv::imshow(imshow_window_name, img);
    }
    cv::waitKey(100);
}


void YOLOv11_OBB_TRT::post_process_worker() {
    bool success;
    int eng_idx;
    EngineData* eng;
    while (1) {
        if (_end_worker) break;
        success = _postprocess_engine_queue.try_pop(eng_idx);
        if (!success) {
            Sleep(100);
            continue;
        }
        eng = &engs[eng_idx];

        /*
        cudaStreamSynchronize(*eng.stream);
        post_process_cpu(eng.output_bbe_gpu_ptr, eng.output_bbe_size, _num_classes, _batch_size,
            eng.input_bchw_gpu_ptr, _batch_size, 3, _tile_h, _tile_w, *eng.stream);
        */

        if (post_process(*eng)) {
            
            std::cout << "post process error!!" << std::endl;
        }
        /*
        else {
            std::cout << "post process success!!" << std::endl;
        }
        */

        //view post process result
        //view_post_process_result(eng);

        //Check mem usage and Push result to Tile Queue
        for (int b = 0; b < _batch_size; ++b) {
            if ((*eng).image_idxs[b] != -1) {
                
                //Check if array out of range (may not crash but loss some objects)
                if ((*eng).post_process_count_cpu_ptr[b]>= max_object_count_per_tile_before_nms) {
                    std::cout << "Batch #" << b << " post processing meta result not enough space: "
                        << (*eng).post_process_count_cpu_ptr[b] << " >= "
                        << max_object_count_per_tile_before_nms << "\t consider increase max_object_count_per_tile_before_nms." << std::endl;
                }
                if ((*eng).post_process_final_count_cpu_ptr[b] >= max_object_count_per_tile_after_nms) {
                    std::cout << "Batch #" << b << " post processing final result not enough space: "
                        << (*eng).post_process_final_count_cpu_ptr[b] << " >= "
                        << max_object_count_per_tile_after_nms << "\t consider increase max_object_count_per_tile_after_nms." << std::endl;
                }

                //Push to Tile Data Queue
                TileData td;
                td.image_idx = (*eng).image_idxs[b];
                td.tile_idx = (*eng).tile_idxs[b];
                td.results = (*eng).resultss[b];
				//std::cout << "td.image_idx:" << td.image_idx << " td.tile_idx:" << td.tile_idx << " td.results.size():" << td.results.size() << std::endl;
                _tile_data_queue.push(td);
            }
        }

        //recycle the resource
        _unused_engine_queue.push(eng_idx);
    }
}


void YOLOv11_OBB_TRT::result_collect_worker() {
    bool success;
    TileData td;
    while (1) {
        if (_end_worker) break;
        success = _tile_data_queue.try_pop(td);
        if (!success) {
            Sleep(100);
            continue;
        }

        while (!_result_data_collector.count(td.image_idx)) {
            Sleep(100);
        }

        while (!_image_tile_counters.count(td.image_idx)) {
            Sleep(100);
        }


        _result_data_collector[td.image_idx].resultss[td.tile_idx] = td.results;
        //_result_data_collector[td.image_idx].tile_done += 1;
        //int tile_done = _result_data_collector[td.image_idx].tile_done_counter.fetch_add(1) + 1;
		int tile_done = _image_tile_counters[td.image_idx].fetch_add(1) + 1;
        //if (_result_data_collector[td.image_idx].tile_done == _result_data_collector[td.image_idx].tile_count) {
        if (tile_done == _result_data_collector[td.image_idx].tile_count) {
            _output_data_queue.push(_result_data_collector[td.image_idx]);
            _result_data_collector.erase(td.image_idx);
        }

        //std::cout << "td.image_idx:" << td.image_idx << std::endl;
        //std::cout << "td.results.size():" << td.results.size() << std::endl;
        //std::cout << "image_idx: " << td.image_idx << " tile_done: " << tile_done << "/" << _result_data_collector[td.image_idx].tile_count << std::endl;
        //std::cout << "_result_data_collector[td.image_idx].tile_count:"  << std::endl;
    }
}

void YOLOv11_OBB_TRT::tile2tile_nms(std::vector<OBBDetection>& rs1, std::vector<OBBDetection>& rs2) {
    for (int i = 0; i < rs1.size(); ++i) {
        for (int j = 0; j < rs2.size(); ++j) {
            float iou = rotated_iou(rs1[i], rs2[j]);
            if (iou >= _iou_thres) {
                if (rs1[i].score <= rs2[j].score) {
                    rs1[i].keep = false;
                }
                else {
                    rs2[j].keep = false;
                }
            }
        }
    }
}

std::string add_folder_prefix(std::string old_path, std::string prefix) {
    //std::cout << old_path << std::endl;
    size_t pos = old_path.find_last_of("\\/");
    std::string folder = old_path.substr(0, pos);  // "D:\\images"
    std::string filename = old_path.substr(pos + 1);  // "1.jpg"
    std::string new_path = folder + prefix + "\\" + filename;
    return new_path;
}

void YOLOv11_OBB_TRT::output_worker() {
    bool success;
    ImageData im;
    while (1) {
        if (_end_worker) break;
        success = _output_data_queue.try_pop(im);
        if (!success) {
            Sleep(100);
            continue;
        }
        ImageData output_im;

        output_im.batch_info = im.batch_info;
        output_im.image_idx = im.image_idx;
        output_im.image_path = im.image_path;
        output_im.input_img_cpu = im.input_img_cpu;
        output_im.original_img_cpu = im.original_img_cpu;
        output_im.tile_count = im.tile_count;
        output_im.input_buffer_size = im.input_buffer_size;
        output_im.batch_buffer_size = im.batch_buffer_size;
        //output_im.tile_done = im.tile_done;

        //Offset to Global
        for (int tile_idx = 0; tile_idx < im.resultss.size(); ++tile_idx) {
            int tile_offset_x = (tile_idx % im.batch_info.batch_w) * im.batch_info.offset_w;
            int tile_offset_y = (tile_idx / im.batch_info.batch_w) * im.batch_info.offset_h;
            for (int i = 0; i < im.resultss[tile_idx].size(); ++i) {
                im.resultss[tile_idx][i].x += tile_offset_x;
                im.resultss[tile_idx][i].y += tile_offset_y;
                if (im.resize_ratio != 1.0f) {
                    im.resultss[tile_idx][i].x /= im.resize_ratio;
                    im.resultss[tile_idx][i].y /= im.resize_ratio;
                    im.resultss[tile_idx][i].w /= im.resize_ratio;
                    im.resultss[tile_idx][i].h /= im.resize_ratio;
                }
                im.resultss[tile_idx][i].keep = true;
            }
        }

        //Global NMS
        for (int bx1 = 0; bx1 < im.batch_info.batch_w; ++bx1) {
            int bx2s = bx1 - 1 > 0 ? bx1 - 1 : 0;
            for (int by1 = 0; by1 < im.batch_info.batch_h; ++by1) {
                int by2s = by1 - 1 > 0 ? by1 - 1 : 0;
                int tile_idx1 = (by1 * im.batch_info.batch_w) + bx1;
                for (int bx2 = bx2s; bx2 < im.batch_info.batch_w; ++bx2) {
                    for (int by2 = by2s; by2 < im.batch_info.batch_h; ++by2) {
                        int tile_idx2 = (by2 * im.batch_info.batch_w) + bx2;
                        if (tile_idx2 < tile_idx1) {
                            tile2tile_nms(im.resultss[tile_idx1], im.resultss[tile_idx2]);
                        }
                    }
                }
            }
        }

        //Put it together
        for (int tile_idx = 0; tile_idx < im.resultss.size(); ++tile_idx) {
            for (int i = 0; i < im.resultss[tile_idx].size(); ++i) {
                if (im.resultss[tile_idx][i].keep) {
                    output_im.final_results.push_back(im.resultss[tile_idx][i]);
                }
            }
        }

        //std::cout << output_im.final_results.size() << " objects detected in " << output_im.image_path << std::endl;

        if (_draw_bboxes) {
            output_im.draw_img_cpu = output_im.original_img_cpu.clone();
            for (const auto& det : output_im.final_results) {
                draw_rotated_box(output_im.draw_img_cpu, det, _CLASSES, _COLORS);
            }
        }

        if (_draw_bboxes && _output_draw) {
            std::string new_path = add_folder_prefix(output_im.image_path, _result_folder_prefix);
            std::vector<int> params;
            //params.push_back(cv::IMWRITE_JPEG_QUALITY);
            //params.push_back(100);
            //cv::imwrite(new_path + "_obb.jpg", output_im.draw_img_cpu, params);
            cv::imwrite(new_path + "_obb.jpg", output_im.draw_img_cpu);
            //std::cout << new_path << std::endl;
        }

        if (_output_csv) {
            std::string new_path = add_folder_prefix(output_im.image_path, _result_folder_prefix);
			std::ofstream csv_file(new_path + "_obb.csv");
			if (!csv_file.is_open()) {
				std::cout << "Failed to open CSV file: " << new_path + "_obb.csv" << std::endl;
				continue;
			}
			csv_file << "x,y,w,h,angle,score,class\n";
			for (const auto& det : output_im.final_results) {
				csv_file << det.x << "," << det.y << "," << det.w << "," << det.h << ","
					<< det.angle << "," << det.score << "," << _CLASSES[det.class_id] << "\n";
			}
			csv_file.close();
        }

        std::cout << "image_idx:" << output_im.image_idx << " objects:" << output_im.final_results.size() << " path:" << output_im.image_path << std::endl;

        //std::cout << "Done processing: " << output_im.image_path << std::endl;

        //output data
        _result_data_queue.push(output_im);

        //reuse resource
        _unused_data_queue.push(im);
    }
}