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
/// ������ͷ����ͼƬ
/// </summary>
void OpenImg(string imgpath, string enginemode ,string enginpath ,int batchSize, int inputC, int inputH, int inputW, int outputC, int outputH, int outputW)
{
    string baseImgPath = "d:/source/img/";
    string baseEnginePath = "d:/source/nnnet/";
    shared_ptr<Engine>engine = make_shared<Engine>(enginemode, enginpath, batchSize, inputC, inputH, inputW, outputC, outputH, outputW);
    if (imgpath == "camera")
    {
        // ����һ��VideoCapture����
        cv::VideoCapture cap(0); // 0�����Ĭ������ͷ

        // �������ͷ�Ƿ�ɹ���
        if (!cap.isOpened()) 
            std::cerr << "Error: Could not open camera" << std::endl;
        cv::Mat frame;
        while (true) {
            // ��ȡ��ǰ֡
            cap >> frame;
            // ���֡�Ƿ�Ϊ��
            if (frame.empty()) {
                std::cerr << "Error: Could not grab a frame" << std::endl;
                break;
            }
            //cv::Mat frame2 = engine->Predict(frame);
            // ��ʾ��ǰ֡
            //cv::imshow("camera", frame2);
            // ����ESC���˳�
            if (cv::waitKey(10) == 27) 
                break;
        }
        // �ͷ�����ͷ
        cap.release();
        cv::destroyAllWindows();
    }
    else
    {
        imgpath += baseImgPath;
        cv::Mat frame = cv::imread(imgpath);
        // ��ȡ��ʼʱ���
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat frame2 = engine->AnomalibPredict(frame);
        // ��ȡ����ʱ���
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "����ʱ��: " << std::fixed << std::setprecision(3)
            << double(duration.count()) << " ����" << std::endl;
        
        cv::imwrite("output.jpg", frame2);
    }
}
int main(int argc, char** argv)
{
 

    //anomalib
    //OpenImg("0001.jpg", "trt", "anomalib4080.engine", 1, 3, 256, 256, 1, 224, 224);
    OpenImg("0001.jpg", "onnx", "model.onnx", 1, 3, 256, 256, 1, 224, 224);

    //yoloclass
    //OpenImg("camera", "trt", "yolov8ni8.engine", 1, 3, 640, 640, 1, 84, 8400);
     //OpenImg("camera", "trt", "yolov8n1060.engine", 1, 3, 640, 640, 1, 84, 8400);
    //OpenImg("camera", "onnx", "yolov8n.onnx", 1, 3, 640, 640, 1, 84, 8400);

    //yoloseg
    //OpenImg("two_runners1.jpg", "onnx", "yolov8n-seg.onnx", 1, 3, 640, 640, 1, 84, 8400);


    return 0;
}
