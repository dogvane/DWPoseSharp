# DWPoseSharp

用c#实现的使用 onnx 版的DWPose

项目的代码，从 https://github.com/IDEA-Research/DWPose 里的 [python代码](https://github.com/IDEA-Research/DWPose/tree/onnx/ControlNet-v1-1-nightly/annotator/dwpose) 移植过来的。

## 依赖

Microsoft.ML.OnnxRuntime (1.15.1) 【原项目用1.15.1转的模型，不清楚降低版本后还能否使用】

OpencvSharp4 (4.8) 【版本实际上无限制】

## 模型文件

从 https://huggingface.co/yzd-v/DWPose 下载：


dw-ll_ucoco_384.onnx

yolox_l.onnx



## 使用样例


```csharp
Wholebody.SetOnnxFolder("../../../../../ckpts");

var img = Cv2.ImRead("../../../demo.jpg");

var wholebody = new Wholebody();
var poseDatas = wholebody.Det(img);
DrawUtils.DrawPoseData(img, poseDatas);
img.SaveImage("pose.jpg");

```
