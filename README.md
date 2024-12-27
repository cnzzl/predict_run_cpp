# predict_run
本代码是用C++写的推理引擎，用于处理onnx，tensorrt等模型的推理部署
```cpp
//实例化类
//enginemode:选择onnx，trt
//enginpath:引擎文件的地址
shared_ptr<Engine>engine = make_shared<Engine>(enginemode, enginpath, batchSize, inputC, inputH, inputW, outputC, outputH, outputW);
```
主要收集了以下的网络模型的前后处理：
## Anomalib
Anomalib是一个异常检测（Abnormal）的库（library），里面的内容的确十分丰富，集成了十余种近年来准确率较高的缺陷（异常）检测算法，基本都是无监督学习的方法，诸如padim算法、fastflow算法等
其输入为1x3x256x256的float
其输出为1x1x224x224的float

```cpp
//Anomalib推理，输出bbox框
engine->AnomalibPredict(frame);
```
## YOLO cls

