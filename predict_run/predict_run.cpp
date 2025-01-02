#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include  "opencv2/opencv.hpp"
#include "Engine.h"
#include <time.h> 
using namespace std;

/// <summary>
/// 打开摄像头或者图片
/// </summary>
void OpenImg(string imgpath, string enginemode ,string enginpath ,int batchSize, int inputC, int inputH, int inputW, int outputC, int outputH, int outputW)
{
    
    shared_ptr<Engine>engine = make_shared<Engine>(enginemode, enginpath, batchSize, inputC, inputH, inputW, outputC, outputH, outputW);
    if (imgpath == "camera")
    {
        // 创建一个VideoCapture对象
        cv::VideoCapture cap(0); // 0代表打开默认摄像头

        // 检查摄像头是否成功打开
        if (!cap.isOpened()) 
            std::cerr << "Error: Could not open camera" << std::endl;
        cv::Mat frame;
        while (true) {
            float start = cv::getTickCount();
            // 读取当前帧
            cap >> frame;
            // 检查帧是否为空
            if (frame.empty()) {
                std::cerr << "Error: Could not grab a frame" << std::endl;
                break;
            }
            cv::Mat frame2 = engine->YOLO8ClsPredict(frame);
            //cv::Mat frame2 = engine->YOLO8PosePredict(frame);
            
            // 计算FPS render it
            float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
            float EEE = 1.0 / t;
            putText(frame2, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
             //显示当前帧
            cv::imshow("camera", frame2);
            // 按下ESC键退出
            if (cv::waitKey(10) == 27) 
                break;
        }
        // 释放摄像头
        cap.release();
        cv::destroyAllWindows();
    }
    else
    {
        cv::Mat frame = cv::imread(imgpath);
        // 获取开始时间点
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat frame2 = engine->AnomalibPredict(frame);
        // 获取结束时间点
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "predict time: " << std::fixed << std::setprecision(3)
            << double(duration.count()) << " ms" << std::endl;
        
        cv::imwrite("output.jpg", frame2);
    }
}
int main(int argc, char** argv)
{
 

    //anomalib
    //OpenImg("0004.jpg", "trt", "D:/source/nnnet/anomalib4080.engine", 1, 3, 256, 256, 1, 224, 224);
    //OpenImg("0001.jpg", "onnx", "model.onnx", 1, 3, 256, 256, 1, 224, 224);

    //yoloclass
    //OpenImg("camera", "trt", "D:/source/nnnet/yolov8ni8.engine", 1, 3, 640, 640, 1, 84, 8400);
    //OpenImg("camera", "trt", "D:/source/nnnet/yolov8n1060.engine", 1, 3, 640, 640, 1, 84, 8400);
    //OpenImg("camera", "onnx", "D:/source/nnnet/yolov8n.onnx", 1, 3, 640, 640, 1, 84, 8400);

    //yoloseg
    //OpenImg("two_runners1.jpg", "onnx", "yolov8n-seg.onnx", 1, 3, 640, 640, 1, 84, 8400);

    //yolopose
    OpenImg("camera", "onnx", "D:/source/nnnet/yolov8n-pose.onnx", 1, 3, 640, 640, 1, 56, 8400);

    return 0;
}
//#include <iostream>
//#include <filesystem>
//#include <string>
//#include <iomanip>
//namespace fs = std::filesystem;
//int main() {
//    // 指定要遍历的文件夹路径
//    fs::path directory = "your_directory_path";
//
//    // 检查路径是否存在且是一个目录
//    if (!fs::exists(directory) || !fs::is_directory(directory)) {
//        std::cerr << "The specified path is not a valid directory." << std::endl;
//        return 1;
//    }
//
//    int counter = 1;
//    std::string extension;
//
//    // 遍历目录中的每个文件
//    for (const auto& entry : fs::directory_iterator(directory)) {
//        // 检查文件是否是图片（根据扩展名）
//        extension = entry.path().extension().string();
//        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp" || extension == ".gif") {
//            // 构造新的文件名
//            std::string new_filename = "0000" + std::to_string(counter);
//            new_filename = new_filename.substr(new_filename.size() - 4) + extension;
//
//            // 构造新的文件路径
//            fs::path new_path = entry.path().parent_path() / new_filename;
//
//            // 重命名文件
//            fs::rename(entry.path(), new_path);
//
//            // 增加计数器
//            counter++;
//        }
//    }
//
//    std::cout << "Files have been renamed successfully." << std::endl;
//    return 0;
//}