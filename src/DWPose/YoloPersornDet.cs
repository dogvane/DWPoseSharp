using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace DWPose
{
    /// <summary>
    /// yolo 的人物检测工具
    /// </summary>
    public class YoloPersornDet
    {
        internal static string onnx_folder = "";

        public static string det_onnx_name = "yolox_l.onnx";

        static InferenceSession s_session;

        private static InferenceSession GetSession()
        {
            var name = Path.Combine(onnx_folder, det_onnx_name);

            if (!File.Exists(name))
            {
                throw new Exception($"onnx 模型文件不存在: file:{name}");
            }


            if (s_session != null)
                return s_session;

            s_session = new InferenceSession(name);

            return s_session;
        }

        public List<YoloPrediction> FindPerson(string imgfilename)
        {
            var img = Cv2.ImRead(imgfilename);

            return FindPerson(img);
        }


        public List<YoloPrediction> FindPerson(Mat orgImg)
        {
            InferenceSession yolox = GetSession();

            // onnx input image size
            var yolo_width = 640;
            var yolo_height = 640;

            var inputName = "images";
            var inputShape = new int[] { 1, 3, yolo_width, yolo_height };
            (var inputTensor, var scRate) = ConvertToTorchTensrot(orgImg, new[] { yolo_width, yolo_height });

            var ret = yolox.Run(new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) });
            DenseTensor<float> yoloret = (DenseTensor<float>)ret.First().Value;
            var shape = yoloret.Dimensions.ToArray();
            var arr = yoloret.ToArray<float>();

            var predictClasses = 80;

            var detections = ParseResults(arr, scRate, new[] { yolo_width, yolo_height }, predictClasses: predictClasses).NMSFilter().ConfidenceFilter();
            
            detections = detections.Where(o => o.LabelIndex == 0).ToList(); // 只保留人的预测框
            return detections;
        }


        #region 图片进入yolo onnx 预处理相关函数

        /// <summary>
        /// 将图片转换为 pyTorch 常用格式的 Tensor （给onnx用）
        /// [1, channle(3), Widht, Heigit]
        /// </summary>
        /// <param name="orgImg"></param>
        /// <param name="size">len(size) == 2 </param>
        /// <param name="keep_aspect_ratio">是否保持原图长宽比例</param>
        /// <returns></returns>
        unsafe public static (DenseTensor<float> tensors, float rate) ConvertToTorchTensrot(Mat orgImg, int[] size, bool keep_aspect_ratio = true)
        {
            var newImgSize = new Size(size[0], size[1]);

            using var outImg = new Mat(size: newImgSize, orgImg.Type());
            // 给 outImg 填充 114
            outImg.SetTo(new Scalar(114, 114, 114));
            var 比例 = Math.Min((float)size[0] / orgImg.Width, (float)size[1] / orgImg.Height);

            if (keep_aspect_ratio)
            {
                // 保持长宽比例，则需要先计算出缩放比例，以及缩放后的绘制区域
                //Console.WriteLine($"长宽比例{比例}");

                // 在输出图片里，绘制的坐标
                //var x轴偏移 = (size[0] - orgImg.Width * 比例) / 2;
                //var y轴偏移 = (size[1] - orgImg.Height * 比例) / 2;
                // 不是按比例剧中，而是直接从左上角开始填充
                var x轴偏移 = 0;
                var y轴偏移 = 0;

                var 绘制区域 = new Rect((int)x轴偏移, (int)y轴偏移, (int)(orgImg.Width * 比例), (int)(orgImg.Height * 比例));
                //Console.WriteLine($"绘制区域:{绘制区域}");

                Cv2.Resize(orgImg, outImg[绘制区域], new Size(绘制区域.Width, 绘制区域.Height));

                return (imgToTensort(outImg), 比例);
            }
            else
            {
                Cv2.Resize(orgImg, outImg, newImgSize);

                return (imgToTensort(outImg), orgImg.Width / orgImg.Height);
            }
        }

        private static unsafe DenseTensor<float> imgToTensort(Mat outImg)
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
                        // 这里不需要 除以 255f
                        chwArray[c * outImg.Height * outImg.Width + h * outImg.Width + w] = paddedArray[h * outImg.Width * 3 + w * 3 + c];
                    }
                }
            }
            
            var imageShpae = new int[] { 1, 3, outImg.Width, outImg.Height };
            return new DenseTensor<float>(chwArray, imageShpae);
        }

        #endregion

        #region yolo onnx 后处理相关函数

        public static (float[,] grids, float[,] expandedStrides) BuildPostProcessData(int[] imgSize, bool p6 = false)
        {
            int[] strides = p6 ? new int[] { 8, 16, 32, 64 } : new int[] { 8, 16, 32 };
            int[] hsizes = strides.Select(s => imgSize[0] / s).ToArray();
            int[] wsizes = strides.Select(s => imgSize[1] / s).ToArray();
            int index = 0;
            var size_grid = 0;

            for (int i = 0; i < strides.Length; i++)
            {
                int hsize = hsizes[i];
                int wsize = wsizes[i];
                size_grid += hsize * wsize;

            }
            float[,] grid = new float[size_grid, 2];
            float[,] expandedStride = new float[size_grid, 1];

            for (int i = 0; i < strides.Length; i++)
            {
                int hsize = hsizes[i];
                int wsize = wsizes[i];
                int stride = strides[i];
                float[,] xv = new float[hsize, wsize];
                float[,] yv = new float[hsize, wsize];

                for (int j = 0; j < hsize; j++)
                {
                    for (int k = 0; k < wsize; k++)
                    {
                        xv[j, k] = k;
                        yv[j, k] = j;
                    }
                }


                for (int j = 0; j < hsize; j++)
                {
                    for (int k = 0; k < wsize; k++)
                    {
                        grid[index, 0] = xv[j, k];
                        grid[index, 1] = yv[j, k];
                        expandedStride[index, 0] = stride;
                        index++;
                    }
                }
            }

            return (grid, expandedStride);
        }

        public List<YoloPrediction> ParseResults(float[] results, float scRate, int[] size, float confidence = 0.45f, int predictClasses = 1000)
        {
            //output  1 25200 {dimensions}
            //box format  0,1,2,3 ->box,4->confidence，5-{dimensions} -> classes confidence
            int dimensions = predictClasses + 5;
            int rows = results.Length / dimensions;
            int confidenceIndex = 4;
            int labelStartIndex = 5;

            var (grides, expanded_strides) = BuildPostProcessData(size);

            List<YoloPrediction> detections = new List<YoloPrediction>();

            for (int i = 0; i < rows; ++i)
            {
                var index = i * dimensions;

                var _confidence = results[index + confidenceIndex];

                if (_confidence <= confidence)
                    continue;

                for (int j = labelStartIndex; j < dimensions; ++j)
                {
                    if (results[index + j] * _confidence > 0.5f)
                    {
                        var value_0 = results[index];
                        var value_1 = results[index + 1];
                        var value_2 = results[index + 2];
                        var value_3 = results[index + 3];
                        //Console.WriteLine($"({value_0},{value_1},{value_2},{value_3})  {scRate}");

                        value_0 = (grides[i, 0] + value_0) * expanded_strides[i, 0];
                        value_1 = (grides[i, 1] + value_1) * expanded_strides[i, 0];
                        value_2 = (float)(expanded_strides[i, 0] * Math.Exp(value_2));
                        value_3 = (float)(expanded_strides[i, 0] * Math.Exp(value_3));

                        //var bbox = new BBox(value_0, value_1, value_2, value_3);

                        var x0 = (value_0 - (value_2 / 2f));
                        var x1 = (value_1 - (value_3 / 2f));
                        var x2 = (value_0 + (value_2 / 2f));
                        var x3 = (value_1 + (value_3 / 2f));
                        x0 /= scRate;
                        x1 /= scRate;
                        x2 /= scRate;
                        x3 /= scRate;

                        var bbox = new BBox(x0, x1, x2, x3);

                        //Console.WriteLine($"row:{i} confidence:{_confidence} {results[index + j]}    ({value_0},{value_1},{value_2},{value_3}) bbox:{bbox}");
                        var l_index = j - labelStartIndex;

                        detections.Add(new YoloPrediction()
                        {
                            BBox = bbox,
                            Confidence = results[index + j],
                            LabelIndex = l_index,
                            LabelName = YoloLabes.ContainsKey(l_index) ? YoloLabes[l_index] : null
                        });
                    }
                }
            }
            return detections;
        }

        Dictionary<int, string> YoloLabes = YoloDetExtend.GetLabelsMap();

        #endregion
    }

    static class YoloDetExtend
    {
        static Dictionary<int, string> s_yoloLabesMap = null;

        /// <summary>
        /// 标签ID和最终label名称的对应表
        /// </summary>
        /// <returns></returns>
        public static Dictionary<int, string> GetLabelsMap()
        {
            if (s_yoloLabesMap != null)
                return s_yoloLabesMap;

            s_yoloLabesMap = new Dictionary<int, string>();
            for (var i = 0; i < YoloLabes.Length; i++)
            {
                s_yoloLabesMap.Add(i, YoloLabes[i]);
            }

            return s_yoloLabesMap;
        }

        public static readonly string[] YoloLabes = new string[] {
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush" };


        /// <summary>
        /// Run NMS on current Prediction list.
        /// </summary>
        /// <param name="predictions"></param>
        /// <param name="IOU_threshold"></param>
        /// <returns></returns>
        public static List<YoloPrediction> NMSFilter(this List<YoloPrediction> predictions, float IOU_threshold = 0.45f)
        {
            List<YoloPrediction> final_predications = new List<YoloPrediction>();

            for (int i = 0; i < predictions.Count; i++)
            {
                int j = 0;
                for (j = 0; j < final_predications.Count; j++)
                {
                    if (final_predications[j] % predictions[i] > IOU_threshold)
                    {
                        break;
                    }
                }
                if (j == final_predications.Count)
                {
                    final_predications.Add(predictions[i]);
                }
            }
            return final_predications;
        }

        /// <summary>
        /// Filter out detections that have confidence below threshold.
        /// </summary>
        /// <param name="predictions"></param>
        /// <param name="confidence_threshold"></param>
        /// <returns></returns>
        public static List<YoloPrediction> ConfidenceFilter(this List<YoloPrediction> predictions, float confidence_threshold = 0.3f)
        {
            List<YoloPrediction> final_predications = new List<YoloPrediction>();

            foreach (var p in predictions)
            {
                if (p.Confidence >= confidence_threshold)
                    final_predications.Add(p);
            }
            return final_predications;
        }

    }

}
