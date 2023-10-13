using DWPose;
using OpenCvSharp;
using System.Diagnostics;

namespace DWPoseTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // ImageTest();
            VideoTest();
        }

        private static void VideoTest()
        {
            OnnxUtils.ConfigOnnx("../../../../../ckpts", OnnxRunModel.Cuda);

            // 使用 cv2 打开一个视频，并进行播放
            var cap = new VideoCapture("../../../demo.mp4");
            var fps = cap.Fps;
            var frameCount = cap.FrameCount;
            var frameWidth = cap.FrameWidth;
            var frameHeight = cap.FrameHeight;
            var delay = (int)(1000 / fps);

            var frame = new Mat();
            var wholebody = new Wholebody();
            var saveVideo = new VideoWriter("pose.mp4", FourCC.MP4V, fps, new Size(frameWidth, frameHeight));
            int runCount = 0;
            Stopwatch stopwatch = new Stopwatch();
            var times = new List<long>();

            while (true)
            {
                stopwatch.Restart();

                cap.Read(frame);
                if (frame.Empty())
                {
                    break;
                }

                var poseDatas = wholebody.Det(frame);
                DrawUtils.DrawPoseData(frame, poseDatas);
                saveVideo.Write(frame);
                stopwatch.Stop();
                times.Add(stopwatch.ElapsedMilliseconds);

                runCount++;
                if (runCount % fps == 0)
                {
                    Console.WriteLine($"{(int)(runCount * 100 / frameCount)}% timems: {times.Average()}");
                    times.Clear();
                }

                //Cv2.ImShow("frame", frame);
                //Cv2.WaitKey(delay);
            }
            saveVideo.Release();
        }

        private static void ImageTest()
        {
            OnnxUtils.ConfigOnnx("../../../../../ckpts", OnnxRunModel.CPU);

            var img = Cv2.ImRead("../../../demo.jpg");

            var wholebody = new Wholebody();
            var poseDatas = wholebody.Det(img);
            DrawUtils.DrawPoseData(img, poseDatas);
            img.SaveImage("pose.jpg");
        }
    }
}