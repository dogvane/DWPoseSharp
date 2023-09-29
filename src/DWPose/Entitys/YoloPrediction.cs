namespace DWPose
{
    /// <summary>
    /// Prediction result representing one detected object.
    /// </summary>
    public class YoloPrediction
    {
        /// <summary>
        /// Location and size of detected object.
        /// </summary>
        public BBox BBox { get; set; } //[required]

        /// <summary>
        /// Lable index of the object, indicating object type.
        /// </summary>
        public int LabelIndex { get; set; }

        /// <summary>
        /// Name of the detected object. May be null if not in label list.
        /// </summary>
        public string? LabelName { get; set; }

        /// <summary>
        /// How certain is the predictor think this prediction is correct.
        /// </summary>
        public float Confidence { get; set; }

        /// <summary>
        /// Get the IOU of two prediction boxes.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>IOU value between 0 and 1</returns>
        public static float operator %(YoloPrediction a, YoloPrediction b)
        {
            return a.BBox % b.BBox;
        }
    }
}
