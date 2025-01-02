#include <iostream>
#include "opencv2/opencv.hpp"
#include<iostream>
#include <vector>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInferPlugin.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <onnxruntime_cxx_api.h>
using namespace nvinfer1;

using namespace std;
using namespace cv;
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // 根据需要自定义日志输出逻辑
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: " << msg << std::endl;
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "INFO: " << msg << std::endl;
            break;
        default:
            break;
        }
    }
};
static Logger gLogger;

class Engine
{
public:
    Engine(string name, string enginpath, int batchSize, int inputC, int inputH, int inputW, int outputC, int outputH, int outputW);
    ~Engine();

    cv::Mat img_oringin;
    cv::Mat img;
    cv::Mat img_result;
    std::vector<float> input_data;
    std::vector<std::vector<float>> result;
    std::vector<cv::Mat> imageresult;

    float* inputData;
    double scale;
    double scale2;

    int top;
    int left;

    string name;
    string enginpath;
    int batchSize, inputC, inputH, inputW, outputC, outputH, outputW;

    size_t numInputNodes;
    size_t numOutputNodes;
    std::vector<std::string> input_node_names, output_node_names;

    IExecutionContext* context;
    void* buffers[3];
    /*shared_ptr<Ort::Session>session_;*/
    
    /// <summary>
    /// 打印MAT
    /// </summary>
    /// <param name="img_res"></param>
    void CheckMat(cv::Mat img_res ,int row,int cal)
    {
        for (int i = 0; i < row; ++i) {
            for (int j = 0; j < cal; ++j) {
                // 根据Mat的数据类型选择合适的指针类型
                if (img_res.type() == CV_8U) {
                    cout << "mat(" << i << ", " << j << ") = " << static_cast<int>(img_res.at<uchar>(i, j)) << endl;
                }
                else if (img_res.type() == CV_32F) {
                    cout << "mat(" << i << ", " << j << ") = " << img_res.at<float>(i, j) << endl;
                }
                else if (img_res.type() == CV_64F) {
                    cout << "mat(" << i << ", " << j << ") = " << img_res.at<double>(i, j) << endl;
                }
                // 根据需要添加更多类型
            }
        }
    }
    cv::Mat YOLO8ClsPredict(cv::Mat frame)
    {
        if (name == "trt")
        {
            YOLO8ClsAfter(TrtPredict(frame, 1));
            return img_result;
        }
        else if (name == "onnx")
        {
            YOLO8ClsAfter(OnnxPredict(frame, 1));
            return img_result;
        }
    }
    cv::Mat AnomalibPredict(cv::Mat frame)
    {
        if (name == "trt")
        {
            AnomalibAfter(TrtPredict(frame, 0));
            return img_result;
        }

        else if (name == "onnx")
        {
            AnomalibAfter(OnnxPredict(frame, 0));
            return img_result;
        }
    }
    cv::Mat YOLO8PosePredict(cv::Mat frame)
    {
        if (name == "trt")
        {
            YOLO8poseAfter(TrtPredict(frame, 1));
            return img_result;
        }
        else if (name == "onnx")
        {
            YOLO8poseAfter(OnnxPredict(frame, 1));
            return img_result;
        }
    }
    struct TensorInfo {
        vector<int64_t> shape;
        vector<float> data;
    };
  
    /// <summary>
    /// onnx推理
    /// </summary>
    /// <param name="frame"></param>
    /// <returns></returns>
    cv::Mat OnnxPredict(cv::Mat frame,bool letterBoxImage)
    {
        
        // 创建InferSession, 查询支持硬件设备
        // GPU Mode, 0 - gpu device id
        std::wstring modelPath = std::wstring(enginpath.begin(), enginpath.end());
        Ort::SessionOptions session_options;
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "onnx");

        session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
        //std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
        //OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        Ort::Session session_(env, modelPath.c_str(), session_options);
        std::vector<std::string> input_node_names;
        std::vector<std::string> output_node_names;

        size_t numInputNodes = session_.GetInputCount();
        size_t numOutputNodes = session_.GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;
        input_node_names.reserve(numInputNodes);

        // 从onnx获取input size 1*3*640*640
        int input_w = 0;
        int input_h = 0;
        for (int i = 0; i < numInputNodes; i++) {
            auto input_name = session_.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_w = input_dims[3];
            input_h = input_dims[2];
            //std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
        }
        // 从onnx获取outputsize 84*8400
        int output_h = 0;
        int output_w = 0;
        Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_h = output_dims[1]; // 84
        output_w = output_dims[2]; // 8400
        //std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
        //从onnx获取out and in name
        for (int i = 0; i < numOutputNodes; i++) {
            auto out_name = session_.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(out_name.get());
        }
        //std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;
        //图形预处理
        // format frame
        ResizeImage(frame, letterBoxImage);

        cv::Mat image = img;

        cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(inputW, inputH), cv::Scalar(0, 0, 0), true, false);
        /* ptrPM->ResizeImage(frame, 1);
         cv::Mat blob = ptrPM->img;*/
        size_t tpixels = inputH * inputW * 3;
        std::array<int64_t, 4> input_shape_info{ 1, 3, inputH, inputW };

        // set input data and inference
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
        const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
        const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
        std::vector<Ort::Value> ort_outputs;
        //预测
        try {
            ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());

        }
        catch (std::exception e) {
            std::cout << e.what() << std::endl;
        }

#pragma region anomalib
        //// 获取输出Tensor
        //TensorInfo output_info;
        //output_info.shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        //output_info.data.resize(output_info.shape[0] * output_info.shape[1] * output_info.shape[2] * output_info.shape[3]);
        //memcpy(output_info.data.data(), ort_outputs[0].GetTensorMutableData<float>(), output_info.data.size() * sizeof(float));
        //Mat img_res2(output_info.shape[2], output_info.shape[3], CV_32F, output_info.data.data());
        //AnomalibAfter(img_res2);
#pragma endregion

#pragma region yoloclass


#pragma endregion

        float* pdata = ort_outputs[0].GetTensorMutableData<float>();
        cv::Mat img_float(outputH, outputW, CV_32F, pdata);
       
        //delete[] pdata;
        session_options.release();
        session_.release();
        return img_float;
        //// output data
        //// 处理输出
        //float* pdata = ort_outputs[0].GetTensorMutableData<float>();

        ///*assert(ort_outputs.size() == 1 && ort_outputs.front().IsTensor());
        //float* pdata = ort_outputs[0].GetTensorMutableData<float>();*/

        //ptrPM->AfterProcessing222(pdata);
        ////delete[] pdata;
        ////delete[] ptrPM->inputData;
        ////delete[] outputData;
        //return ptrPM->img_result;
    }

    /// <summary>
    /// trt推理
    /// </summary>
    /// <param name="frame"></param>
    /// <returns></returns>
    cv::Mat TrtPredict(cv::Mat frame, bool letterBoxImage)
    {
        ResizeImage(frame, letterBoxImage);
        //// 创建CUDA流
        cudaStream_t stream;

        cudaStreamCreate(&stream);
        // 将inputBlob的数据复制到inputData中
        cudaMemcpyAsync(buffers[0], inputData, batchSize * inputC * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, stream);
        // 设置绑定
        void* bindings[3] = { buffers[0], buffers[1],buffers[2] };
        void* outputData = new float[batchSize * outputC * outputH * outputW];

        // 运行推理
        context->executeV2(bindings);

        //执行推理
        //context->enqueueV3(stream);

        // 等待推理完成（如果需要异步操作，可以不在这里同步）
        cudaStreamSynchronize(stream);
        
        cudaMemcpyAsync(outputData, buffers[1], batchSize * outputC * outputH * outputW * sizeof(float), cudaMemcpyDeviceToHost, stream);
        //数据后处理
        cv::Mat img_float(outputH, outputW, CV_32F, outputData);
       
        return img_float;
        
    }
    /// <summary>
    /// trt推理pro
    /// </summary>
    /// <param name="frame"></param>
    /// <returns></returns>
    cv::Mat TrtPredictPRO(cv::Mat frame, bool letterBoxImage)
    {
        ResizeImage(frame, letterBoxImage);
        //// 创建CUDA流
        cudaStream_t stream;
        cudaStreamCreate(&stream);
     
        // 将inputBlob的数据复制到inputData中
        cudaMemcpyAsync(buffers[0], inputData, batchSize * inputC * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, stream);
        // 设置绑定
        void* bindings[3] = { buffers[0], buffers[1],buffers[2] };
        // 运行推理
        context->executeV2(bindings);
        
        // 等待推理完成（如果需要异步操作，可以不在这里同步）
        cudaStreamSynchronize(stream);

        void* outputData = new float[batchSize * outputC * outputH * outputW];
        cudaMemcpyAsync(outputData, buffers[1], batchSize * outputC * outputH * outputW * sizeof(float), cudaMemcpyDeviceToHost, stream);
        //数据后处理
        cv::Mat img_float(outputH, outputW, CV_32F, outputData);
        // 清理资源
        cudaStreamDestroy(stream);
        return img_float;

    }
	std::vector<std::string> labels;
	std::vector<std::string> readClassNames()
	{
		std::vector<std::string> classNames;
		std::ifstream fp("class.txt");
		if (!fp.is_open())
		{
			printf("could not open file...\n");
			exit(-1);
		}
		std::string name;
		while (!fp.eof())
		{
			std::getline(fp, name);
			if (name.length())
				classNames.push_back(name);
		}
		fp.close();
		return classNames;
	}
    void YOLO8ClsAfter(cv::Mat data_float)
    {
        cv::Mat frame = this->img_oringin;

        //cv::Mat dout(outputH, outputW, CV_32F, (float*)outputData);
        cv::Mat det_output = data_float.t(); // 8400x84

        // post-process
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        // fix bug, boxes consistence!
        for (int i = 0; i < det_output.rows; i++) {
            cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            // 置信度 0～1之间
            if (score > 0.7)
            {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                int x = static_cast<int>((cx - 0.5 * ow - left) / scale);
                int y = static_cast<int>((cy - 0.5 * oh - top) / scale2);

                int width = static_cast<int>(ow / scale);
                int height = static_cast<int>(oh / scale2);

                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }
        // NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0, 0, indexes);
        if (indexes.size() < boxes.size())
        {
            for (size_t i = 0; i < indexes.size(); i++) {
                int index = indexes[i];
                int idx = classIds[index];
                cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
                cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                    cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
                putText(frame, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

            }
        }
        

        this->img_result = frame;
    }
    void AnomalibAfter(cv::Mat data_float)
    {
        //数据后处理
        cv::Mat frame = img_oringin;
        // 转换输出Tensor为Mat
        
        // 创建一个用于存储转换后数据的矩阵
        cv::Mat img_res;

        // 将浮点数矩阵转换为8位无符号整数矩阵
        data_float.convertTo(img_res, CV_8U, 1.0, 0.0);
        
        //data_float.convertTo(img_res, CV_8U, 255.0);
        // 阈值处理
        Mat thresh;
        threshold(img_res, thresh, 40, 255, THRESH_BINARY);
        //cv::imwrite("thresh.jpg", thresh);
       

        // 查找轮廓
        vector<vector<Point>> contours;
        findContours(thresh, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

        // 绘制轮廓
        for (const auto& cnt : contours) {
            Rect bounding_rect = boundingRect(cnt);
            float ratio = (float)256 / 224;
            rectangle(frame, Point(bounding_rect.x * ratio / scale, bounding_rect.y * ratio / scale2),
                Point((bounding_rect.x + bounding_rect.width) * ratio / scale, (bounding_rect.y + bounding_rect.height) * ratio / scale2),
                Scalar(0, 0, 255), 2);
        }
        this->img_result = frame;

        //// 保存图像
        //imwrite("img_name", image);
    }
    void YOLO8poseAfter(cv::Mat data_float)
    {
        cv::Mat frame = this->img_oringin;

        //cv::Mat dout(outputH, outputW, CV_32F, (float*)outputData);
        cv::Mat det_output = data_float.t(); // 8400x84
        CheckMat(det_output,8400,56);
        // post-process
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        // fix bug, boxes consistence!
        for (int i = 0; i < det_output.rows; i++) {
            cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            // 置信度 0～1之间
            if (score > 0.7)
            {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                int x = static_cast<int>((cx - 0.5 * ow - left) / scale);
                int y = static_cast<int>((cy - 0.5 * oh - top) / scale2);

                int width = static_cast<int>(ow / scale);
                int height = static_cast<int>(oh / scale2);

                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }
        // NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0, 0, indexes);
        if (indexes.size() < boxes.size())
        {
            for (size_t i = 0; i < indexes.size(); i++) {
                int index = indexes[i];
                int idx = classIds[index];
                cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
                cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                    cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
                putText(frame, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

            }
        }


        this->img_result = frame;

    }
    void ResizeImage(Mat imgorigin, bool letterBoxImage)
    {
        this->img_oringin = imgorigin;
        int ih = imgorigin.rows;
        int iw = imgorigin.cols;
        int h = inputH;
        int w = inputW;
        cv::Mat img;
        if (letterBoxImage)
        {
            this->scale = min((double)w / iw, (double)h / ih);//缩放比例
            this->scale2 = min((double)w / iw, (double)h / ih);//缩放比例
            int nw = static_cast<int>(iw * this->scale);
            int nh = static_cast<int>(ih * this->scale2);

            cv::resize(imgorigin, img, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
            //生成背景
            cv::Mat imgBack = cv::Mat::ones(h, w, CV_8UC3);
            cv::Scalar color(128, 128, 128);
            imgBack = color;

            top = (h - nh) / 2;
            int bottom = top + nh;
            left = (w - nw) / 2;
            int right = left + nw;

            cv::Mat imgrange = imgBack(cv::Range(top, bottom), cv::Range(left, right));
            img.copyTo(imgrange);
            this->img = imgBack;
        }
        else
        {
            top = 0;
            left = 0;
            this->scale = (double)w / iw;
            this->scale2 = (double)h / ih;
            cv::resize(imgorigin, img, cv::Size(inputW, inputH), 0, 0, cv::INTER_LINEAR);
            this->img = img;
        }
        inputData = new float[batchSize * inputC * inputH * inputW];
        // 将inputBlob的数据复制到inputData中
        for (int i = 0; i < inputH * inputW; i++) {
            inputData[i] = this->img.at<cv::Vec3b>(i / inputW, i % inputW)[0] / 255.0;  // R通道
            inputData[i + inputH * inputW] = this->img.at<cv::Vec3b>(i / inputW, i % inputW)[1] / 255.0;  // G通道
            inputData[i + 2 * inputH * inputW] = this->img.at<cv::Vec3b>(i / inputW, i % inputW)[2] / 255.0;  // B通道
        }
    }
private:

};

Engine::Engine(string name, string enginpath, int batchSize, int inputC, int inputH, int inputW, int outputC, int outputH, int outputW)
{
    this->batchSize = batchSize;
    this->inputC = inputC;
    this->inputH = inputH;
    this->inputW = inputW;
    this->outputC = outputC;
    this->outputH = outputH;
    this->outputW = outputW;
    this->labels = readClassNames();

    this->name = name;
    this->enginpath = enginpath;
    if (name == "trt")
    {
        //初始化trt
        initLibNvInferPlugins(&gLogger, "");
        IRuntime* runtime = createInferRuntime(gLogger);
        // 从文件中反序列化Engine对象
        std::ifstream engineFile(enginpath, std::ios::binary);
        if (!engineFile)
            std::cerr << "无法打开Engine文件进行读取。" << std::endl;
        engineFile.seekg(0, std::ios::end);
        const size_t fileSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(fileSize);
        engineFile.read(engineData.data(), fileSize);
        engineFile.close();
        ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine)
            std::cerr << "无法反序列化Engine对象。" << std::endl;
        // 创建TensorRT的执行上下文对象
        context = engine->createExecutionContext();
        //// 创建CUDA流
        cudaStream_t stream;

        cudaStreamCreate(&stream);

        // 分配GPU内存
        cudaMalloc(&buffers[0], batchSize * inputC * inputH * inputW * sizeof(float));  // 分配输入内存
        //cudaMalloc(&buffers[1], batchSize * outputC * outputH * outputW * sizeof(float));  // 分配输出内存
        cudaMalloc(&buffers[1], batchSize * outputC * outputH * outputW * sizeof(float));  // 分配输出内存
        cudaMalloc(&buffers[2], batchSize * outputC * outputH * outputW * sizeof(float));  // 分配输出内存
    }
    else if (name == "onnx")
    {

    }
}

Engine::~Engine()
{
    delete inputData;
    
}