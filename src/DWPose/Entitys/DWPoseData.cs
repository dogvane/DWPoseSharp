namespace DWPose
{
    public class DWPoseData
    {
        /// <summary>
        /// 18,3
        /// </summary>
        public float[,] Bodys { get; set; }

        /// <summary>
        /// 6,2
        /// </summary>
        public float[,] Foots { get; set; }

        /// <summary>
        /// 68,2
        /// </summary>
        public float[,] Faces { get; set; }

        /// <summary>
        /// 21,2
        /// </summary>
        public float[,] Hands1 { get; set; }

        /// <summary>
        /// 21,2
        /// 人是有2只手的
        /// 暂时不清楚左右，只分1，2
        /// </summary>
        public float[,] Hands2 { get; set; }

    }

}
