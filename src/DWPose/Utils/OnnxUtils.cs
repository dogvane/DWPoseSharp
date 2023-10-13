using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace DWPose
{
    /// <summary>
    /// 一些辅助类
    /// </summary>
    public static class OnnxUtils
    {
        /// <summary>
        /// 图片数据转 onnx 用的tensort数据
        /// 批处理用
        /// </summary>
        /// <param name="outImg"></param>
        /// <param name="useStd"></param>
        /// <returns></returns>
        public static unsafe DenseTensor<float> ImgToTensort(this List<Mat> outImgs, bool useStd = false)
        {
            var outImg = outImgs.First();

            float[] chwArray = new float[3 * outImg.Height * outImg.Width * outImgs.Count];

            byte[] paddedArray = new byte[outImg.Total() * outImg.Channels()];

            for (var imgIndex = 0; imgIndex < outImgs.Count; imgIndex++)
            {
                outImg = outImgs[imgIndex];

                Marshal.Copy(outImg.Data, paddedArray, 0, paddedArray.Length);
                var m = imgIndex * 3 * outImg.Height * outImg.Width;

                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < outImg.Height; h++)
                    {
                        for (int w = 0; w < outImg.Width; w++)
                        {
                            if (useStd)
                            {
                                switch (c)
                                {
                                    case 0:
                                        chwArray[m + c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 123.676f) / 58.395f;
                                        break;
                                    case 1:
                                        chwArray[m + c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 116.28f) / 57.12f;
                                        break;
                                    case 2:
                                        chwArray[m + c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 103.53f) / 57.375f;
                                        break;
                                }

                            }
                            else
                                chwArray[m + c * outImg.Height * outImg.Width + h * outImg.Width + w] = paddedArray[h * outImg.Width * 3 + w * 3 + c];
                        }
                    }
                }
            }

            var imageShpae = new int[] { outImgs.Count, 3, outImg.Height, outImg.Width };
            return new DenseTensor<float>(chwArray, imageShpae);
        }
        
        /// <summary>
        /// 图片数据转 onnx 用的tensort数据
        /// </summary>
        /// <param name="outImg"></param>
        /// <param name="useStd"></param>
        /// <returns></returns>
        public static unsafe DenseTensor<float> ImgToTensort(this Mat outImg, bool useStd = false)
        {
            byte[] paddedArray = new byte[outImg.Total() * outImg.Channels()];

            Marshal.Copy(outImg.Data, paddedArray, 0, paddedArray.Length);

            // todo 这里需要做一下性能测试

            float[] chwArray = new float[3 * outImg.Height * outImg.Width];
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < outImg.Height; h++)
                {
                    for (int w = 0; w < outImg.Width; w++)
                    {
                        if (useStd)
                        {
                            switch(c)
                                {
                                case 0:
                                    chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] -123.676f) / 58.395f;
                                    break;
                                case 1:
                                    chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 116.28f) / 57.12f;
                                    break;
                                case 2:
                                    chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = (paddedArray[h * outImg.Width * 3 + w * 3 + c] - 103.53f) / 57.375f;
                                    break;
                            }
                                
                        }
                        else
                            chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = paddedArray[h * outImg.Width * 3 + w * 3 + c];
                    }
                }
            }

            var imageShpae = new int[] { 1, 3, outImg.Height, outImg.Width };
            return new DenseTensor<float>(chwArray, imageShpae);
        }

        public static string GetPrintString(this float[,] dst)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < dst.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < dst.GetLength(1); j++)
                {
                    sb.Append(dst[i, j]);
                    if (j != dst.GetLength(1) - 1)
                    {
                        sb.Append(", ");
                    }
                }
                sb.Append("]");
                if (i != dst.GetLength(0) - 1)
                {
                    sb.AppendLine(",");
                }
                else
                {
                    sb.AppendLine();
                }
            }
            return sb.ToString();
        }

        public static void PrintArray(this float[,] dst)
        {
            Console.WriteLine(GetPrintString(dst));
        }

        static string _onnx_folder = "";
        static OnnxRunModel _runModel = OnnxRunModel.CPU;
        static int _deviceId = 0;

        /// <summary>
        /// config onnx base info.
        /// </summary>
        /// <param name="modelsBasefolder">onnx model save folder</param>
        /// <param name="runModel"></param>
        /// <param name="deviceId">If you are using CUDA or DirectML and have multiple devices, you need to set deviceId (0,1,2...)</param>
        public static void ConfigOnnx(string modelsBasefolder = "", OnnxRunModel runModel = OnnxRunModel.CPU, int deviceId = 0)
        {
            _onnx_folder = modelsBasefolder;
            _runModel = runModel;
            _deviceId = deviceId;
        }

        public static InferenceSession GetSession(string det_onnx_name)
        {
            var name = Path.Combine(_onnx_folder, det_onnx_name);

            if (!File.Exists(name))
            {
                throw new Exception($"not find onnx file:{name}");
            }

            var config = new SessionOptions();

            switch(_runModel)
            {
                case OnnxRunModel.Cuda:
                    config.AppendExecutionProvider_CUDA(0);
                    break;
                case OnnxRunModel.DirectML:
                    config.AppendExecutionProvider_DML();
                    break;
            }
            config.EnableMemoryPattern = false;

            return new InferenceSession(name, config);
        }
    }

    public enum OnnxRunModel
    {
        CPU,

        Cuda,

        DirectML,
    }
}
