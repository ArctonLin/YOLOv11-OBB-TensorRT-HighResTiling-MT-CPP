#include "yolov11-obb-tensorrt.h"

__global__ void preprocess_yolov11_kernel(
    uchar* img_ptr, //hwc
    int img_h, int img_w,
    int tile_h, int tile_w,
    int batch_h, int batch_w,
    int offset_h, int offset_w,
    int img_start_index, int img_end_index,
    int bchw_start_index, int bchw_end_index,
    int this_batch_size,
    half* bchw_ptr, //bchw
    bool inverse_channel //true=RGB, false=BGR
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // bchw width idx
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // bchw height idx
    int b = blockIdx.z * blockDim.z + threadIdx.z;  // bchw batch idx

    if (x >= tile_w || y >= tile_h || b < bchw_start_index || b >= bchw_end_index) {
        return;
    }

    int img_batch_idx = b - bchw_start_index + img_start_index;

    // 計算該 batch 的 tile 原始位置
    int batch_x = img_batch_idx % batch_w;
    int batch_y = img_batch_idx / batch_w;
    int src_x = batch_x * offset_w + x;
    int src_y = batch_y * offset_h + y;
    int src_idx = (src_y * img_w + src_x) * 3;
    int dst_idx = b * 3 * tile_h * tile_w + y * tile_w + x;
    if (inverse_channel) {
        bchw_ptr[dst_idx + 0 * tile_h * tile_w] = __float2half(img_ptr[src_idx + 2] / 255.0f);  // R
        bchw_ptr[dst_idx + 1 * tile_h * tile_w] = __float2half(img_ptr[src_idx + 1] / 255.0f);  // G
        bchw_ptr[dst_idx + 2 * tile_h * tile_w] = __float2half(img_ptr[src_idx + 0] / 255.0f);  // B
    }
    else {
        bchw_ptr[dst_idx + 0 * tile_h * tile_w] = __float2half(img_ptr[src_idx + 0] / 255.0f);  // B
        bchw_ptr[dst_idx + 1 * tile_h * tile_w] = __float2half(img_ptr[src_idx + 1] / 255.0f);  // G
        bchw_ptr[dst_idx + 2 * tile_h * tile_w] = __float2half(img_ptr[src_idx + 2] / 255.0f);  // R
    }
}

bool YOLOv11_OBB_TRT::pre_process(ImageData* im, int img_start_index, int img_end_index, EngineData* eng, int bchw_start_index, int bchw_end_index) {
    int this_batch_size = bchw_end_index - bchw_start_index;

    dim3 block(24, 24, 1);
    dim3 grid((_tile_w + block.x - 1) / block.x,
        (_tile_h + block.y - 1) / block.y,
        (_batch_size + block.z - 1) / block.z);

    preprocess_yolov11_kernel <<<grid, block, 0, *(*eng).stream >>> (
        (*im).input_img_gpu_ptr,
        (*im).batch_info.img_h, (*im).batch_info.img_w,
        _tile_h, _tile_w,
        (*im).batch_info.batch_h, (*im).batch_info.batch_w,
        (*im).batch_info.offset_h, (*im).batch_info.offset_w,
        img_start_index, img_end_index,
        bchw_start_index, bchw_end_index,
        this_batch_size,
        (*eng).input_bchw_gpu_ptr,
        true
        );

	//cudaStreamSynchronize(*(*eng).stream);
    //cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    //std::cout << "End Pre Process without Error!!" << std::endl;
    return 0;
}

__global__ void postprocess_yolov11_obb_kernel(
    half* predictions, //[4,20,21504] bs,NC+5,group_size
    int batch_size,
    int group_size, //21504 with 1024x1024 input
    int num_classes,
    float conf_thres,
    float* outputs, //[4,16384,8] bs,max_object_count_per_tile_before_nms,object_data
    int* counts,
    int max_count
    ) {
    int g = blockIdx.x * blockDim.x + threadIdx.x; //group idx this is 21504 in [4,20,21504]
    int b = blockIdx.y * blockDim.y + threadIdx.y; //batch idx this is 4 in [4,20,21504]

    if (b >= batch_size || g >= group_size) return;

    //printf("g:%d/%d\n", g, group_size);

    int batch_offset = b * (num_classes + 5) * group_size;
    float x = __half2float(predictions[batch_offset + g + 0 * (group_size)]);
    float y = __half2float(predictions[batch_offset + g + 1 * (group_size)]);
    float w = __half2float(predictions[batch_offset + g + 2 * (group_size)]);
    float h = __half2float(predictions[batch_offset + g + 3 * (group_size)]);
    float angle = __half2float(predictions[batch_offset + g + (num_classes + 4) * (group_size)]);  // assume in radians

    // find best class
    float best_conf = 0.0f;
    int best_cls = -1;
    for (int c = 0; c < num_classes; ++c) {
        float conf = __half2float(predictions[batch_offset + g + (4 + c) * (group_size)]);
        if (conf > best_conf) {
            best_conf = conf;
            best_cls = c;
        }
    }

    if (best_conf >= conf_thres) {
        //printf("conf_thres=%f\n", conf_thres);
        //std::cout << "xywhacc:" << x << " " << y << " " << w << " " << h << " " << angle << " " << best_conf << " " << best_cls << std::endl; + keep
        //dets.push_back({ x, y, w, h, angle, best_conf, best_cls });
        int idx = atomicAdd(&counts[b], 1);
        if (idx >= max_count) return;
        int batch_offset = b * max_count * 8;
        outputs[batch_offset + idx * 8 + 0] = x;
        outputs[batch_offset + idx * 8 + 1] = y;
        outputs[batch_offset + idx * 8 + 2] = w;
        outputs[batch_offset + idx * 8 + 3] = h;
        outputs[batch_offset + idx * 8 + 4] = angle;
        outputs[batch_offset + idx * 8 + 5] = best_conf;
        outputs[batch_offset + idx * 8 + 6] = best_cls;
        outputs[batch_offset + idx * 8 + 7] = 1.0f; //keep
        /*
        printf("before nms: bxywha: %d %f %f %f %f %f\n",
            b,
            outputs[batch_offset + idx * 8 + 0],
            outputs[batch_offset + idx * 8 + 1],
            outputs[batch_offset + idx * 8 + 2],
            outputs[batch_offset + idx * 8 + 3],
            outputs[batch_offset + idx * 8 + 4]);
        if (((int)(best_cls + 0.001f)) != 1) {
            printf("found not ship!!\n");
        }
        */
    }
}


struct float2pt {
    float x, y;
    __device__ float2pt() {}
    __device__ float2pt(float x_, float y_) : x(x_), y(y_) {}
};

// Rotate Coordinate
__device__ float2pt rotate(float2pt pt, float angle_rad) {
    float ca = cosf(angle_rad);
    float sa = sinf(angle_rad);
    return float2pt(pt.x * ca - pt.y * sa, pt.x * sa + pt.y * ca);
}

// Calculate the 4 Corner of OBB
__device__ void get_corners(float cx, float cy, float w, float h, float angle_rad, float2pt corners[4]) {
    float2pt half_extent[] = {
        float2pt(-w / 2, -h / 2),
        float2pt(w / 2, -h / 2),
        float2pt(w / 2,  h / 2),
        float2pt(-w / 2,  h / 2)
    };
    for (int i = 0; i < 4; ++i) {
        float2pt rot = rotate(half_extent[i], angle_rad);
        corners[i] = float2pt(cx + rot.x, cy + rot.y);
    }
}

// Calculate two OBB intersection
__device__ float polygon_area(const float2pt* pts, int count) {
    float area = 0.0f;
    for (int i = 0; i < count; ++i) {
        const float2pt& p1 = pts[i];
        const float2pt& p2 = pts[(i + 1) % count];
        area += (p1.x * p2.y - p2.x * p1.y);
    }
    return fabsf(area * 0.5f);
}

// Sutherland–Hodgman Polygon Cutting，Get intersection of poly1 and poly2
__device__ int polygon_clip(const float2pt* subject, int n_subject, const float2pt* clip, int n_clip, float2pt* output) {
    float2pt input[16], temp[16];
    int input_size = n_subject;
    for (int i = 0; i < n_subject; ++i) input[i] = subject[i];

    for (int i = 0; i < n_clip; ++i) {
        float2pt cp1 = clip[i];
        float2pt cp2 = clip[(i + 1) % n_clip];
        float2pt edge = float2pt(cp2.x - cp1.x, cp2.y - cp1.y);
        float2pt normal = float2pt(-edge.y, edge.x);

        int new_size = 0;
        for (int j = 0; j < input_size; ++j) {
            float2pt cur = input[j];
            float2pt prev = input[(j + input_size - 1) % input_size];

            float d1 = (cur.x - cp1.x) * normal.x + (cur.y - cp1.y) * normal.y;
            float d2 = (prev.x - cp1.x) * normal.x + (prev.y - cp1.y) * normal.y;

            if (d2 >= 0 && d1 >= 0) {
                temp[new_size++] = cur;
            }
            else if (d2 >= 0 && d1 < 0) {
                float t = d2 / (d2 - d1);
                temp[new_size++] = float2pt(prev.x + t * (cur.x - prev.x), prev.y + t * (cur.y - prev.y));
            }
            else if (d2 < 0 && d1 >= 0) {
                float t = d2 / (d2 - d1);
                temp[new_size++] = float2pt(prev.x + t * (cur.x - prev.x), prev.y + t * (cur.y - prev.y));
                temp[new_size++] = cur;
            }
        }
        input_size = new_size;
        for (int j = 0; j < new_size; ++j) input[j] = temp[j];
    }

    for (int i = 0; i < input_size; ++i) output[i] = input[i];
    return input_size;
}

// Calculate IOU of two OBB
__device__ float obb_iou(float x1, float y1, float w1, float h1, float angle1,
    float x2, float y2, float w2, float h2, float angle2) {
    float2pt corners1[4], corners2[4];
    get_corners(x1, y1, w1, h1, angle1, corners1);
    get_corners(x2, y2, w2, h2, angle2, corners2);

    float2pt inter_pts[16];
    int n = polygon_clip(corners1, 4, corners2, 4, inter_pts);
    if (n < 3) return 0.0f;

    float inter_area = polygon_area(inter_pts, n);
    float area1 = w1 * h1;
    float area2 = w2 * h2;

    return inter_area / (area1 + area2 - inter_area + 1e-6f);
}

__global__ void postprocess_yolov11_obb_nms_kernel(
    float* outputs, //[4,4096,8]
    int* counts,
    int max_count,
    int batch_size,
    float iou_thres
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //group idx this is 4096 in [4,4096,8]
    int j = blockIdx.y * blockDim.y + threadIdx.y; //group idx this is 4096 in [4,4096,8], j always smaller than i
    int b = blockIdx.z * blockDim.z + threadIdx.z; //batch idx this is 4 in [4,4096,8]

    if (b >= batch_size || i >= counts[b] || i >= max_count || j >= i) return; // consider j smaller than i, i smaller than (candidate object count of current batch) or (max count)
    int batch_offset = b * max_count * 8;

    if (outputs[batch_offset + i * 8 + 6] != outputs[batch_offset + j * 8 + 6]) return; //different class

    float iou = obb_iou(
        outputs[batch_offset + i * 8 + 0],
        outputs[batch_offset + i * 8 + 1],
        outputs[batch_offset + i * 8 + 2],
        outputs[batch_offset + i * 8 + 3],
        outputs[batch_offset + i * 8 + 4],
        outputs[batch_offset + j * 8 + 0],
        outputs[batch_offset + j * 8 + 1],
        outputs[batch_offset + j * 8 + 2],
        outputs[batch_offset + j * 8 + 3],
        outputs[batch_offset + j * 8 + 4]);

    if (iou >= iou_thres) { // delete one of it
        if (outputs[batch_offset + j * 8 + 5] >= outputs[batch_offset + i * 8 + 5]) { //keep j if conf_j > conf_i
            outputs[batch_offset + i * 8 + 7] = 0.0f; // delete i
            /*
            printf("in nms: delete: bxywha: %d %f %f %f %f %f\n",
                b,
                outputs[batch_offset + i * 8 + 0],
                outputs[batch_offset + i * 8 + 1],
                outputs[batch_offset + i * 8 + 2],
                outputs[batch_offset + i * 8 + 3],
                outputs[batch_offset + i * 8 + 4]);
            */
        }
        else { //keep i
            outputs[batch_offset + j * 8 + 7] = 0.0f; // delete j
            /*
            printf("in nms: delete: bxywha: %d %f %f %f %f %f\n",
                b,
                outputs[batch_offset + j * 8 + 0],
                outputs[batch_offset + j * 8 + 1],
                outputs[batch_offset + j * 8 + 2],
                outputs[batch_offset + j * 8 + 3],
                outputs[batch_offset + j * 8 + 4]);
            */
        }
    }
}

__global__ void postprocess_yolov11_obb_get_final_kernel(
    float* outputs, //[4,4096,8]
    int* counts,
    int max_count,
    float* final_outputs, //[4,256,7]
    int* final_counts,
    int max_final_count,
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; //group idx this is 4096 in [4,4096,8]
    int b = blockIdx.y * blockDim.y + threadIdx.y; //batch idx this is 4 in [4,4096,8]
    if (b >= batch_size || i >= counts[b] || i >= max_count) return;

    if (outputs[b * max_count * 8 + i * 8 + 7] == 1.0f) { //keep
        int idx = atomicAdd(&final_counts[b], 1);
        if (idx >= max_final_count) return;
        int batch_offset = b * max_count * 8;
        int final_batch_offset = b * max_final_count * 7;
        final_outputs[final_batch_offset + idx * 7 + 0] = outputs[batch_offset + i * 8 + 0]; //x
        final_outputs[final_batch_offset + idx * 7 + 1] = outputs[batch_offset + i * 8 + 1]; //y
        final_outputs[final_batch_offset + idx * 7 + 2] = outputs[batch_offset + i * 8 + 2]; //w
        final_outputs[final_batch_offset + idx * 7 + 3] = outputs[batch_offset + i * 8 + 3]; //h
        final_outputs[final_batch_offset + idx * 7 + 4] = outputs[batch_offset + i * 8 + 4]; //angle
        final_outputs[final_batch_offset + idx * 7 + 5] = outputs[batch_offset + i * 8 + 5]; //conf
        final_outputs[final_batch_offset + idx * 7 + 6] = outputs[batch_offset + i * 8 + 6]; //cls
        /*
        printf("outputs bxywha: %d %f %f %f %f %f\n",
            b,
            final_outputs[final_batch_offset + idx * 7 + 0],
            final_outputs[final_batch_offset + idx * 7 + 1],
            final_outputs[final_batch_offset + idx * 7 + 2],
            final_outputs[final_batch_offset + idx * 7 + 3],
            final_outputs[final_batch_offset + idx * 7 + 4]);
        */
    }
}

bool YOLOv11_OBB_TRT::post_process(EngineData& eng) {
    
    //std::cout << "post process entry!!" << std::endl;

    cudaMemcpyAsync(eng.post_process_count_gpu_ptr, eng.post_process_count_zeros_gpt_ptr, eng.post_process_count_size, cudaMemcpyDeviceToDevice, *eng.stream);
    
    dim3 block(24, 24, 1);
    dim3 grid(
        (_group_size + block.x - 1) / block.x,
        (_batch_size + block.y - 1) / block.y,
        1);

    postprocess_yolov11_obb_kernel << <grid, block, 0, *eng.stream >> > (
        eng.output_bbe_gpu_ptr,
        _batch_size,
        _group_size,
        _num_classes,
        _conf_thres,
        eng.post_process_data_gpu_ptr,
        eng.post_process_count_gpu_ptr,
        max_object_count_per_tile_before_nms
        );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpyAsync(eng.post_process_count_cpu_ptr, eng.post_process_count_gpu_ptr, eng.post_process_count_size, cudaMemcpyDeviceToHost, *eng.stream);
    
    block=dim3(24, 24, 1);
    grid=dim3(
        (_group_size + block.x - 1) / block.x,
        (_group_size + block.y - 1) / block.y,
        (_batch_size + block.z - 1) / block.z);
    
    postprocess_yolov11_obb_nms_kernel << <grid, block, 0, *eng.stream >> > (
        eng.post_process_data_gpu_ptr,
        eng.post_process_count_gpu_ptr,
        max_object_count_per_tile_before_nms,
        _batch_size,
        _iou_thres
        );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpyAsync(eng.post_process_final_count_gpu_ptr, eng.post_process_count_zeros_gpt_ptr, eng.post_process_count_size, cudaMemcpyDeviceToDevice, *eng.stream);
    
    block=dim3(24, 24, 1);
    grid=dim3(
        (max_object_count_per_tile_before_nms + block.x - 1) / block.x,
        (_batch_size + block.y - 1) / block.y,
        1);

    postprocess_yolov11_obb_get_final_kernel << <grid, block, 0, *eng.stream >> > (
        eng.post_process_data_gpu_ptr,
        eng.post_process_count_gpu_ptr,
        max_object_count_per_tile_before_nms,
        eng.post_process_final_gpu_ptr,
        eng.post_process_final_count_gpu_ptr,
        max_object_count_per_tile_after_nms,
        _batch_size
        );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    //std::cout << "Post Process Result Downloading..." << std::endl;
    
    cudaMemcpyAsync(eng.post_process_final_cpu_ptr, eng.post_process_final_gpu_ptr, eng.post_process_final_size, cudaMemcpyDeviceToHost, *eng.stream);
    cudaMemcpyAsync(eng.post_process_final_count_cpu_ptr, eng.post_process_final_count_gpu_ptr, eng.post_process_count_size, cudaMemcpyDeviceToHost, *eng.stream);
    cudaStreamSynchronize(*eng.stream);

    //std::cout << "Post Process Result Downloaded!!" << std::endl;

    for (int b = 0; b < _batch_size; ++b) {
        eng.resultss[b].clear();
        if (eng.image_idxs[b] == -1) continue; //pre process early ending due to queue is empty
        int objects_count = eng.post_process_final_count_cpu_ptr[b] < max_object_count_per_tile_after_nms ? eng.post_process_final_count_cpu_ptr[b] : max_object_count_per_tile_after_nms;
        //std::cout << "batch #" << b << " detect " << objects_count << " objects." << std::endl;
        for (int i = 0; i < objects_count; ++i) {
            OBBDetection r;
            int final_batch_i_offset = b * max_object_count_per_tile_after_nms * 7 + i * 7;
            r.x = eng.post_process_final_cpu_ptr[final_batch_i_offset + 0];
            r.y = eng.post_process_final_cpu_ptr[final_batch_i_offset + 1];
            r.w = eng.post_process_final_cpu_ptr[final_batch_i_offset + 2];
            r.h = eng.post_process_final_cpu_ptr[final_batch_i_offset + 3];
            r.angle = eng.post_process_final_cpu_ptr[final_batch_i_offset + 4];
            r.score = eng.post_process_final_cpu_ptr[final_batch_i_offset + 5];
            r.class_id = eng.post_process_final_cpu_ptr[final_batch_i_offset + 6] + 0.0001f;
            //std::cout << "xywhasc:" << r.x << " " << r.y << " " << r.w << " " << r.h << " " << r.angle << " " << r.score << " " << r.class_id << " " << std::endl;
            eng.resultss[b].push_back(r);
        }
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    return 0;
}