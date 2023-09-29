using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace DWPose
{
    public class Wholebody
    {
        internal static string onnx_folder = "";

        public static string pose_onnx_name = "dw-ll_ucoco_384.onnx";

        static InferenceSession s_session;

        YoloPersornDet yoloPersornDet = new YoloPersornDet();

        private static InferenceSession GetSession()
        {
            var name = Path.Combine(onnx_folder, pose_onnx_name);

            if (!File.Exists(name))
            {
                throw new Exception($"onnx 模型文件不存在: file:{name}");
            }

            if (s_session != null)
                return s_session;

            s_session = new InferenceSession(name);

            return s_session;
        }

        public DWPoseData[] Det(string imgfilename)
        {
            if (!File.Exists(imgfilename))
                throw new Exception($"not find image file:{imgfilename}");

            var oriImg = Cv2.ImRead(imgfilename);

            return Det(oriImg);
        }

        public DWPoseData[] Det(Mat oriImg)
        {

            List<YoloPrediction> detections = yoloPersornDet.FindPerson(oriImg);

            var dwpose = GetSession();
            List<DWPoseData> retPoseData = new List<DWPoseData>();

            var model_input_size = new[] { 288, 384 };  // onnx 模型的输入尺寸
            var (out_img, out_center, out_scale) = PreProcess(oriImg, detections, model_input_size);
            var simcc_split_ratio = 2.0f;

            var inputTensor = out_img.ImgToTensort(true);
            var ret = dwpose.Run((new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) }));
            var simcc_x = ret.First(o => o.Name == "simcc_x").AsTensor<float>();
            var simcc_y = ret.First(o => o.Name == "simcc_y").AsTensor<float>();

            var simcc_x_size = new[] { simcc_x.Dimensions[1], simcc_x.Dimensions[2], };
            var simcc_y_size = new[] { simcc_y.Dimensions[1], simcc_y.Dimensions[2], };

            for (var index = 0; index < out_img.Count; index++)
            {
                var scale = out_scale[index];
                var center = out_center[index];

                var sx = simcc_x.Skip(index * simcc_x_size[0] * simcc_x_size[1]).Take(simcc_x_size[0] * simcc_x_size[1]).ToArray();
                var sy = simcc_y.Skip(index * simcc_y_size[0] * simcc_y_size[1]).Take(simcc_y_size[0] * simcc_y_size[1]).ToArray();

                var (keypoints, scores) = Decode(sx, simcc_x_size, sy, simcc_y_size, simcc_split_ratio);

                var keypoints_info = PostProcess(model_input_size, center, scale, keypoints, scores);

                DWPoseData poseData = BuildPoseData(keypoints_info);

                retPoseData.Add(poseData);
            }

            return retPoseData.ToArray();
        }

        #region 预处理


        /// <summary>
        /// 数据的前处理
        /// </summary>
        /// <param name="orgImg"></param>
        /// <param name="detections"></param>
        /// <param name="size"></param>
        /// <returns></returns>
        (List<Mat> out_img, List<float[]> out_center, List<float[]> out_scale) PreProcess(Mat orgImg, List<YoloPrediction> detections, int[] size)
        {            
            // 预处理主要做以下工作
            // 1. 获得人物的中心点和缩放比例
            // 2. 裁剪出人物的图像
            // 3. 缩放到指定的大小
            // 4. 转换为Tensor
            List<Mat> out_img = new List<Mat>();
            List<float[]> out_center = new List<float[]>();
            List<float[]> out_scale = new List<float[]>();

            foreach (var item in detections)
            {
                var (center, scale) = item.BBox.BboxXyxy2Cs(1.25f);

                var (resized_img, imgScale) = TopDownAffine(size, scale, center, orgImg);

                using Mat normalizedImage = new Mat();

                Cv2.Normalize(resized_img, normalizedImage, 0, 255, NormTypes.MinMax, MatType.CV_8UC3);

                out_img.Add(resized_img);

                out_center.Add(center);
                out_scale.Add(imgScale);
            }

            return (out_img, out_center, out_scale);
        }
        static (Mat, float[]) TopDownAffine(int[] input_size, float[] bbox_scale, float[] bbox_center, Mat img)
        {
            var w = input_size[0];
            var h = input_size[1];
            var warp_size = new Size(w, h);

            // reshape bbox to fixed aspect ratio
            bbox_scale = FixAspectRatio(bbox_scale, (float)w / h);

            // get the affine matrix
            var center = bbox_center;
            var scale = bbox_scale;
            var warp_mat = GetWarpMatrix(center, scale, 0, new[] { w, h });

            // do affine transform
            var dst = new Mat();
            Cv2.WarpAffine(img, dst, warp_mat, warp_size, flags: InterpolationFlags.Linear);

            return (dst, bbox_scale);
        }

        static float[] FixAspectRatio(float[] bbox_scale, float aspect_ratio)
        {
            var w = bbox_scale[0];
            var h = bbox_scale[1];
            if (w > h * aspect_ratio)
            {
                h = w / aspect_ratio;
            }
            else
            {
                w = h * aspect_ratio;
            }

            return new float[] { w, h };
        }

        public static Mat GetWarpMatrix(float[] center, float[] scale, float rot, int[] outputSize, float[] shift = null, bool inv = false)
        {
            if (shift == null)
                shift = new float[] { 0, 0 };

            float srcW = scale[0];
            float dstW = outputSize[0];
            float dstH = outputSize[1];
            float rotRad = DegreeToRadian(rot);
            float[] srcDir = RotatePoint(new float[] { 0, srcW * -0.5f }, rotRad);
            float[] dstDir = new float[] { 0, dstW * -0.5f };

            float[,] src = new float[3, 2]
            {
                { center[0] + scale[0] * shift[0],  center[1] + scale[1] * shift[1] },
                { center[0] + srcDir[0] + scale[0] * shift[0], center[1] + srcDir[1] + scale[1] * shift[1] },
                {0, 0 }
            };

            var t = Get3rdPoint(new[] { src[0, 0], src[0, 1] }, new[] { src[1, 0], src[1, 1] });
            src[2, 0] = t[0]; src[2, 1] = t[1];

            float[,] dst = new float[3, 2]
            {
                {dstW * 0.5f, dstH * 0.5f },
                {dstW * 0.5f + dstDir[0], dstH * 0.5f + dstDir[1] },
                {0,0 },
            };

            var t2 = Get3rdPoint(new[] { dst[0, 0], dst[0, 1] }, new[] { dst[1, 0], dst[1, 1] });
            dst[2, 0] = t2[0]; dst[2, 1] = t2[1];

            var srcMat = new Mat(3, 2, MatType.CV_32FC1, src);
            var dstMat = new Mat(3, 2, MatType.CV_32FC1, dst);
            Mat warpMat;

            if (inv)
            {
                warpMat = Cv2.GetAffineTransform(dstMat, srcMat);
            }
            else
            {
                warpMat = Cv2.GetAffineTransform(srcMat, dstMat);
            }

            return warpMat;
        }

        private static float DegreeToRadian(float degree)
        {
            return degree * (float)Math.PI / 180;
        }
        private static float[] RotatePoint(float[] point, float angle)
        {
            var sn = (float)Math.Sin(angle);
            var cs = (float)Math.Cos(angle);
            var rot_mat = new float[,] {
                {cs, -sn },
                { sn, cs }
            };

            var rotatedPoint = new float[]
            {
                point[0] * rot_mat[0, 0] + point[1] * rot_mat[0, 1],
                point[0] * rot_mat[1, 0] + point[1] * rot_mat[1, 1]
            };

            return rotatedPoint;
        }
        public static float[] Get3rdPoint(float[] a, float[] b)
        {
            var direction = new float[2];
            direction[0] = a[0] - b[0];
            direction[1] = a[1] - b[1];

            var c = new float[] { b[0] - direction[1], b[1] + direction[0] };
            return c;
        }


        #endregion

        #region 后处理

        private static DWPoseData BuildPoseData(float[,] keypoints_info)
        {
            for (var i = 0; i < keypoints_info.GetLength(0); i++)
            {
                // 基于阈值过滤
                if (keypoints_info[i, 2] < 0.3f)
                {
                    keypoints_info[i, 0] = -1;
                    keypoints_info[i, 1] = -1;
                    keypoints_info[i, 2] = -1;
                }
            }


            // 数据拆分
            var bodys = new float[18, 3];

            for (var i = 0; i < 18; i++)
            {
                bodys[i, 0] = keypoints_info[i, 0];
                bodys[i, 1] = keypoints_info[i, 1];
                bodys[i, 2] = keypoints_info[i, 2];
            }

            var foots = new float[24 - 18, 2];
            for (var i = 18; i < 24; i++)
            {
                foots[i - 18, 0] = keypoints_info[i, 0];
                foots[i - 18, 1] = keypoints_info[i, 1];
            }

            var faces = new float[92 - 24, 2];
            for (var i = 24; i < 92; i++)
            {
                faces[i - 24, 0] = keypoints_info[i, 0];
                faces[i - 24, 1] = keypoints_info[i, 1];
            }

            var headCount = 113 - 92;
            var hands = new float[113 - 92, 2];
            for (var i = 92; i < 113; i++)
            {
                hands[i - 92, 0] = keypoints_info[i, 0];
                hands[i - 92, 1] = keypoints_info[i, 1];
            }

            var hands2 = new float[headCount, 2];
            for (var i = 113; i < 113 + headCount; i++)
            {
                hands2[i - 113, 0] = keypoints_info[i, 0];
                hands2[i - 113, 1] = keypoints_info[i, 1];
            }

            var poseData = new DWPoseData
            {
                Bodys = bodys,
                Foots = foots,
                Faces = faces,
                Hands1 = hands,
                Hands2 = hands2,
            };
            return poseData;
        }

        static float[] NP_Mean(float[,] source, int start, int end)
        {
            var l1 = Math.Min(source.GetLength(0), end - start + 1);
            var l2 = source.GetLength(1);

            var ret = new float[l2];

            for (var x = start; x < start + l1; x++)
            {
                for (var y = 0; y < l2; y++)
                {
                    ret[y] = ret[y] + source[x, y];
                }
            }

            for (var y = 0; y < l2; y++)
            {
                ret[y] = ret[y] / l1;
            }

            return ret;
        }

        /// <summary>
        /// 数据的后处理
        /// </summary>
        /// <param name="model_input_size"></param>
        /// <param name="center"></param>
        /// <param name="scale"></param>
        /// <param name="keypoints"></param>
        /// <param name="scores"></param>
        /// <returns></returns>
        private static float[,] PostProcess(int[] model_input_size, float[] center, float[] scale, float[,] keypoints, float[] scores)
        {
            var kpLen = keypoints.GetLength(0);

            // 这里 +1 是为了方便未来某处的数据插入用
            var keypoints_info = new float[kpLen + 1, 3];

            for (var x = 0; x < kpLen - 1; x++)
            {
                var y = x;
                if (x > 17)     // 返回数组里的 17 位置，先空着，后面再填充
                    y = y + 1;

                keypoints_info[y, 0] = keypoints[x, 0] / model_input_size[0] * scale[0] + center[0] - scale[0] / 2f;
                keypoints_info[y, 1] = keypoints[x, 1] / model_input_size[1] * scale[1] + center[1] - scale[1] / 2f;
                keypoints_info[x, 2] = scores[x];
            }

            var neck = NP_Mean(keypoints_info, 5, 6);
            if (keypoints_info[5, 2] > 0.3)
                neck[2] = keypoints_info[6, 2] > 0.3 ? 1 : 0;

            keypoints_info[17, 0] = neck[0];
            keypoints_info[17, 1] = neck[1];
            keypoints_info[17, 2] = neck[2];

            var mmpose_idx = new int[] { 17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3 };
            var openpose_idx = new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17 };
            var keypoints_info_s = new float[kpLen + 1, 3];
            Array.Copy(keypoints_info, 0, keypoints_info_s, 0, keypoints_info.Length);

            for (var i = 0; i < mmpose_idx.Length; i++)
            {
                var _mmpose = mmpose_idx[i];
                var _openPose = openpose_idx[i];

                keypoints_info[_openPose, 0] = keypoints_info_s[_mmpose, 0];
                keypoints_info[_openPose, 1] = keypoints_info_s[_mmpose, 1];
                keypoints_info[_openPose, 2] = keypoints_info_s[_mmpose, 2];
            }

            return keypoints_info;
        }

        public static int[] Argmax(float[] arr, int[] size)
        {
            int[] argmax = new int[size[0]];
            int stride = size[1];
            for (int i = 0; i < size[0]; i++)
            {
                float max = float.MinValue;
                int maxIndex = 0;
                for (int j = 0; j < stride; j++)
                {
                    if (arr[i * stride + j] > max)
                    {
                        max = arr[i * stride + j];
                        maxIndex = j;
                    }
                }
                argmax[i] = maxIndex;
            }
            return argmax;
        }

        static float[,] ArrayStack(params int[][] arrs)
        {
            var len = arrs.Length;
            var ret = new float[arrs[0].Length, len];
            for(var i = 0;i < arrs[0].Length; i++)
            {
                for (var l = 0; l < len; l++)
                    ret[i, l] = arrs[l][i];
            }

            return ret;
        }

        public static float[] Amax(float[] arr, int[] size)
        {
            int dim1 = size[0];
            int dim2 = size[1];
            float[] result = new float[dim1];

            for (int i = 0; i < dim1; i++)
            {
                float max = float.MinValue;
                for (int j = 0; j < dim2; j++)
                {
                    if (arr[i* dim2 + j] > max)
                    {
                        max = arr[i * dim2 + j];
                    }
                }
                result[i] = max;
            }

            return result;
        }

        public static (float[,], float[]) GetSimccMaximum(float[] simcc_x, int[] simcc_x_size, float[]simcc_y, int[] simcc_y_size)
        {
            var x_locs = Argmax(simcc_x, simcc_x_size);
            var y_locs = Argmax(simcc_y, simcc_y_size);


            // 合并 x_locs 和 y_locs 为一个数组，并转为 float[,2]
            var locs = ArrayStack(x_locs, y_locs);
            var max_val_x = Amax(simcc_x, simcc_x_size);
            var max_val_y = Amax(simcc_y, simcc_y_size);

            // get maximum value across x and y axis
            float[] vals = new float[max_val_x.Length];

            for(var i = 0;i < vals.Length; i++)
            {
                vals[i] = max_val_x[i] > max_val_y[i] ? max_val_x[i] : max_val_y[i];
                if (vals[i] <= 0.0f)
                {
                    locs[i, 0] = -1;
                    locs[i, 1] = -1;
                }
            }

            return (locs, vals);
        }

        public static (float[,], float[]) Decode(float[] simcc_x, int[] simcc_x_size, float[] simcc_y, int[] simcc_y_size, float simcc_split_ratio)
        {
            var (keypoints, scores) = GetSimccMaximum(simcc_x, simcc_x_size, simcc_y, simcc_y_size);

            if(simcc_split_ratio != 1)
            {
                var w = keypoints.GetLength(0);
                var h = keypoints.GetLength(1);
                for(var x= 0; x < w; x ++)
                {
                    for(var y = 0; y < h; y++)
                    {
                        keypoints[x, y] /= simcc_split_ratio;
                    }
                }
            }

            return (keypoints, scores);
        }

        public static void SetOnnxFolder(string folder)
        {
            onnx_folder = folder;
            YoloPersornDet.onnx_folder = folder;
        }

        #endregion

    }
}
