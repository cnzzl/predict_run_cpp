#include <iostream>
#include <fstream>
#include "Onnx2Trt.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

// 实例化记录器界面，捕获所有警告性信息，但忽略信息性消息
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			std::cout << msg << std::endl;
		}
	}
}logger;

void ONNX2TensorRT(const char* ONNX_file, std::string& Engine_file, bool& FP16, bool& INT8, std::string& image_dir, const char*& calib_table) {
	std::cout << "Load ONNX file form: " << ONNX_file << "\nStart export..." << std::endl;
	// 1.创建构建器的实例
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

	// 2.创建网络定义
	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

	// 3.创建一个 ONNX 解析器来填充网络
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

	// 4.读取模型文件并处理任何错误
	parser->parseFromFile(ONNX_file, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		std::cout << parser->getError(i)->desc() << std::endl;

	// 5.创建构建配置，指定TensorRT如何优化模型
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	// 如果是动态模型，则需要设置大小
	/*
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();
	// 配置最小允许batch
	input_dims.d[0] = 1;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
	// 配置最大允许batch
	// if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);
	*/

	// 6.设置属性来控制 TensorRT 如何优化网络
	// 设置内存池的空间
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
	if (FP16) {
		// 判断硬件是否支持FP16
		if (!builder->platformHasFastFp16()) {
			std::cout << "不支持FP16量化！" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	else if (INT8) {
		if (!builder->platformHasFastInt8()) {
			std::cout << "不支持INT8量化！" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		nvinfer1::IInt8EntropyCalibrator2* calibrator = new Calibrator(1, 640, 640, image_dir, calib_table);
		config->setInt8Calibrator(calibrator);
	}

	// 7.指定配置后，构建引擎
	nvinfer1::IHostMemory* serializeModel = builder->buildSerializedNetwork(*network, *config);

	// 8.保存TensorRT模型
	std::ofstream engine(Engine_file, std::ios::binary);
	engine.write(reinterpret_cast<const char*>(serializeModel->data()), serializeModel->size());

	// 9.序列化引擎包含权重的必要副本，因此不再需要解析器、网络定义、构建器配置和构建器，可以安全地删除
	delete parser;
	delete network;
	delete config;
	delete builder;

	// 10.将引擎保存到磁盘后 ，并且可以删除它被序列化到的缓冲区
	delete serializeModel;
	std::cout << "Export success, Save as: " << Engine_file << std::endl;
}

int main(int argc, char** argv) {
	// ONNX 文件路径
	const char* ONNX_file = "yolov8n.onnx";
	// ENGINE 文件保存路径
	std::string Engine_file = "yolov8ni8.engine";

	// 当量化为INT8时，图片路径
	std::string image_dir = "calibimg/";
	// 当量化为INT8时，校准表路径（存在读取，不存在创建）
	const char* calib_table = "weights/calibrator.table";

	// 选择量化方式，若两个都为false，使用FP32生成 ENGINE文件
	bool FP16 = 0;
	bool INT8 = 1;

	std::ifstream file(ONNX_file, std::ios::binary);
	if (!file.good()) {
		std::cout << "Load ONNX file failed！" << std::endl;
	}

	ONNX2TensorRT(ONNX_file, Engine_file, FP16, INT8, image_dir, calib_table);

	return 0;
}
