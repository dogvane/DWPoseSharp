using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DWPose
{
    public class DrawUtils
    {
        public static Mat DrawPersonBox(string orgImgFileName, List<YoloPrediction> detections)
        {
            var orgImg = Cv2.ImRead(orgImgFileName);


            // 基于预测出来的 detections 给 orgImg 画预测框
            foreach (var det in detections)
            {
                var x1 = det.BBox.CenterX - det.BBox.Width / 2;
                var x2 = det.BBox.CenterX + det.BBox.Width / 2;
                var y1 = det.BBox.CenterY - det.BBox.Height / 2;
                var y2 = det.BBox.CenterY + det.BBox.Height / 2;
                var color = Scalar.RandomColor();

                Cv2.Rectangle(orgImg, new Point(x1, y1), new Point(x2, y2), color, 2);
            }

            return orgImg;
        }

        public static byte[] HSVToRGB(float[] hsv, int count)
        {
            if (hsv == null || count <= 0)
            {
                throw new ArgumentException("Invalid input");
            }
            byte[] rgb = new byte[3];
            for (int i = 0; i < 3; i++)
            {
                float h = hsv[i] / (float)count;
                float s = 1.0f;
                float v = 1.0f;
                // Check if the HSV value is in the range of 0 to 1  
                if (h >= 0 && h < 1)
                {
                    float p = 2.0f * 3.14159265359f * h;
                    float q = h * (1.0f - h);
                    float t = (float)Math.Pow((float)(1.0 - h * 6.0f), 6.0f);
                    v = t;
                    s = 1.0f - (float)Math.Pow((float)(1.0 - h * 6.0f), 6.0f);
                }
                else
                {
                    h = h < 0 ? (float)(360.0f + h) : h;
                    s = 1.0f;
                    v = 1.0f;
                }
                int iR = (int)(v * 255);
                int iG = (int)(s * 255);
                int iB = (int)(h * 255);
                rgb[i] = (byte)(iR << 16 | iG << 8 | iB);
            }
            return rgb;
        }


        private static void DrawHands(Mat orgImage, float[,] hands)
        {
            int[,] edges = new int[,] { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 4 }, { 0, 5 }, { 5, 6 }, { 6, 7 }, { 7, 8 }, { 0, 9 }, { 9, 10 }, { 10, 11 }, { 11, 12 }, { 0, 13 }, { 13, 14 }, { 14, 15 }, { 15, 16 }, { 0, 17 }, { 17, 18 }, { 18, 19 }, { 19, 20 } };

            for (var i = 0; i < edges.GetLength(0); i++)
            {
                var ie = edges[i, 0]; var e = edges[i, 1];
                var x1 = hands[ie, 0]; var y1 = hands[ie, 1];
                var x2 = hands[e, 0]; var y2 = hands[e, 1];
                if (x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0)
                {
                    var _1 = (float)ie / edges.GetLength(0);
                    var rgb = HSVToRGB(new[] { _1, 1.0f, 1.0f }, 255);
                    Cv2.Line(orgImage, (int)x1, (int)y1, (int)x2, (int)y2, new Scalar(rgb[0], rgb[1], rgb[2]), 2);
                }
            }

            for (var i = 0; i < hands.GetLength(0); i++)
            {
                var x = hands[i, 0];
                var y = hands[i, 1];
                if (x > 0 && y > 0)
                {
                    Cv2.Circle(orgImage, (int)x, (int)y, 4, new Scalar(0, 0, 255), -1);
                }
            }
        }

        public static void DrawPoseData(Mat orgImage, DWPoseData[] poseDatas)
        {
            foreach(var item in poseDatas)
            {
                DrawPoseData(orgImage, item);
            }
        }

        /// <summary>
        /// 最后的画图工作
        /// </summary>
        /// <param name="orgImage"></param>
        /// <param name="poseData"></param>
        public static void DrawPoseData(Mat orgImage, DWPoseData poseData)
        {
            // 画脸
            for (var i = 0; i < poseData.Faces.GetLength(0); i++)
            {
                var x = poseData.Faces[i, 0];
                var y = poseData.Faces[i, 1];
                if (x > 0 && y > 0)
                {
                    Cv2.Circle(orgImage, (int)x, (int)y, 3, new Scalar(255, 255, 255), -1);
                }
            }

            // 画手

            DrawHands(orgImage, poseData.Hands1);
            DrawHands(orgImage, poseData.Hands2);

            // 画身体
            var stickwidth = 4;

            int[,] limbSeq = new int[,]
                {
                    {2, 3}, {2, 6}, {3, 4}, {4, 5}, {6, 7}, {7, 8}, {2, 9}, {9, 10},
                    {10, 11}, {2, 12}, {12, 13}, {13, 14}, {2, 1}, {1, 15}, {15, 17},
                    {1, 16}, {16, 18}, {3, 17}, {6, 18}
                };

            int[,] colors = new int[,]
                {
                    {255, 0, 0}, {255, 85, 0}, {255, 170, 0}, {255, 255, 0}, {170, 255, 0},
                    {85, 255, 0}, {0, 255, 0}, {0, 255, 85}, {0, 255, 170}, {0, 255, 255},
                    {0, 170, 255}, {0, 85, 255}, {0, 0, 255}, {85, 0, 255}, {170, 0, 255},
                    {255, 0, 255}, {255, 0, 170}, {255, 0, 85}
                };

            var bodys = poseData.Bodys;

            for (int i = 0; i < 17; i++)
            {
                var index1 = limbSeq[i, 0] - 1;
                var index2 = limbSeq[i, 1] - 1;

                if (bodys[index1, 0] == -1 || bodys[index2, 0] == -1)
                {
                    // 检查以下两个点是否存在[n,2] 是得分，-1表示不存在
                    continue;
                }

                var X = new[] { bodys[index1, 1], bodys[index2, 1] };
                var Y = new[] { bodys[index1, 0], bodys[index2, 0] };
                var mX = X.Average();
                var mY = Y.Average();

                float length = (float)Math.Sqrt(Math.Pow(X[0] - X[1], 2) + Math.Pow(Y[0] - Y[1], 2));
                float angle = (float)(Math.Atan2(X[0] - X[1], Y[0] - Y[1]) * 180 / Math.PI);
                Point[] polygon = Cv2.Ellipse2Poly(new Point((int)mY, (int)mX), new Size((int)(length / 2), stickwidth), (int)angle, 0, 360, 1);
                Cv2.FillConvexPoly(orgImage, polygon, new Scalar(colors[i, 0], colors[i, 1], colors[i, 2]));

            }

            for (int i = 0; i < 18; i++)
            {
                if (bodys[i, 2] == -1)
                    continue;


                float x = bodys[i, 0];
                float y = bodys[i, 1];
                Cv2.Circle(orgImage, new Point((int)x, (int)y), 4, new Scalar(colors[i, 0], colors[i, 1], colors[i, 2]), -1);
            }
        }


    }
}
