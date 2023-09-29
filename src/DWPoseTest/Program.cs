using DWPose;
using OpenCvSharp;

namespace DWPoseTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Wholebody.SetOnnxFolder("../../../../../ckpts");
            
            var img = Cv2.ImRead("../../../demo.jpg");

            var wholebody = new Wholebody();
            var poseDatas = wholebody.Det(img);
            DrawUtils.DrawPoseData(img, poseDatas);
            img.SaveImage("pose.jpg");
        }
    }
}