#include <iostream>
#include <fstream>
#include "Onnx2Trt.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"

// ʵ������¼�����棬�������о�������Ϣ����������Ϣ����Ϣ
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			std::cout << msg << std::endl;
		}
	}
}logger;

void ONNX2TensorRT(const char* ONNX_file, std::string& Engine_file, bool& FP16, bool& INT8, std::string& image_dir, const char*& calib_table) {
	std::cout << "Load ONNX file form: " << ONNX_file << "\nStart export..." << std::endl;
	// 1.������������ʵ��
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

	// 2.�������綨��
	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

	// 3.����һ�� ONNX ���������������
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

	// 4.��ȡģ���ļ��������κδ���
	parser->parseFromFile(ONNX_file, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		std::cout << parser->getError(i)->desc() << std::endl;

	// 5.�����������ã�ָ��TensorRT����Ż�ģ��
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	// ����Ƕ�̬ģ�ͣ�����Ҫ���ô�С
	/*
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	auto input_dims = input_tensor->getDimensions();
	// ������С����batch
	input_dims.d[0] = 1;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
	// �����������batch
	// if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
	input_dims.d[0] = maxBatchSize;
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
	config->addOptimizationProfile(profile);
	*/

	// 6.�������������� TensorRT ����Ż�����
	// �����ڴ�صĿռ�
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
	if (FP16) {
		// �ж�Ӳ���Ƿ�֧��FP16
		if (!builder->platformHasFastFp16()) {
			std::cout << "��֧��FP16������" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	else if (INT8) {
		if (!builder->platformHasFastInt8()) {
			std::cout << "��֧��INT8������" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		nvinfer1::IInt8EntropyCalibrator2* calibrator = new Calibrator(1, 640, 640, image_dir, calib_table);
		config->setInt8Calibrator(calibrator);
	}

	// 7.ָ�����ú󣬹�������
	nvinfer1::IHostMemory* serializeModel = builder->buildSerializedNetwork(*network, *config);

	// 8.����TensorRTģ��
	std::ofstream engine(Engine_file, std::ios::binary);
	engine.write(reinterpret_cast<const char*>(serializeModel->data()), serializeModel->size());

	// 9.���л��������Ȩ�صı�Ҫ��������˲�����Ҫ�����������綨�塢���������ú͹����������԰�ȫ��ɾ��
	delete parser;
	delete network;
	delete config;
	delete builder;

	// 10.�����汣�浽���̺� �����ҿ���ɾ���������л����Ļ�����
	delete serializeModel;
	std::cout << "Export success, Save as: " << Engine_file << std::endl;
}

int main(int argc, char** argv) {
	// ONNX �ļ�·��
	const char* ONNX_file = "yolov8n.onnx";
	// ENGINE �ļ�����·��
	std::string Engine_file = "yolov8ni8.engine";

	// ������ΪINT8ʱ��ͼƬ·��
	std::string image_dir = "calibimg/";
	// ������ΪINT8ʱ��У׼��·�������ڶ�ȡ�������ڴ�����
	const char* calib_table = "weights/calibrator.table";

	// ѡ��������ʽ����������Ϊfalse��ʹ��FP32���� ENGINE�ļ�
	bool FP16 = 0;
	bool INT8 = 1;

	std::ifstream file(ONNX_file, std::ios::binary);
	if (!file.good()) {
		std::cout << "Load ONNX file failed��" << std::endl;
	}

	ONNX2TensorRT(ONNX_file, Engine_file, FP16, INT8, image_dir, calib_table);

	return 0;
}
