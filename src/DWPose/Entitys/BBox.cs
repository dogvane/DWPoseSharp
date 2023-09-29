﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DWPose
{
    /// <summary>
    /// Area detected as an object in detections.
    /// </summary>
    public class BBox
    {
        public float MinX, MinY, MaxX, MaxY;
        public BBox(float minX, float minY, float maxX, float maxY)
        {
            MinX = minX;
            MinY = minY;
            MaxX = maxX;
            MaxY = maxY;
        }

        /// <summary>
        /// X cord of the object center.
        /// </summary>
        public float CenterX => (MaxX + MinX) / 2;

        /// <summary>
        /// Y cord of the object center.
        /// </summary>
        public float CenterY => (MaxY + MinY) / 2;

        /// <summary>
        /// Width of the object.
        /// </summary>
        public float Width => MaxX - MinX;

        /// <summary>
        /// Height of the object.
        /// </summary>
        public float Height => MaxY - MinY;

        /// <summary>
        /// Area of the object.
        /// </summary>
        public float Area => Width * Height;

        /// <summary>
        /// Compute IOU of two boxes.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns>0~1 IOU value</returns>
        public static float operator %(BBox a, BBox b)
        {
            return ComputeIOU(a, b);
        }

        private static float ComputeIOU(BBox DRa, BBox DRb)
        {
            float ay1 = DRa.MinY;
            float ax1 = DRa.MinX;
            float ay2 = DRa.MaxY;
            float ax2 = DRa.MaxX;
            float by1 = DRb.MinY;
            float bx1 = DRb.MinX;
            float by2 = DRb.MaxY;
            float bx2 = DRb.MaxX;


            float x_left = Math.Max(ax1, bx1);
            float y_top = Math.Max(ay1, by1);
            float x_right = Math.Min(ax2, bx2);
            float y_bottom = Math.Min(ay2, by2);

            if (x_right < x_left || y_bottom < y_top)
                return 0;
            float intersection_area = (x_right - x_left) * (y_bottom - y_top);
            float bb1_area = (ax2 - ax1) * (ay2 - ay1);
            float bb2_area = (bx2 - bx1) * (by2 - by1);
            float iou = intersection_area / (bb1_area + bb2_area - intersection_area);

            //Debug.Assert(iou >= 0 && iou <= 1);
            return iou;
        }

        public override string ToString()
        {
            return $" ({CenterX},{CenterY})  {Width}x{Height}   [{MinX},{MinY}][{MaxX},{MaxY}]";
        }

        public (float[], float[]) BboxXyxy2Cs(float padding = 1.0f)
        {
            return (new[] { this.CenterX, this.CenterY }, new[] { this.Width * padding, this.Height * padding });
        }
    }
}
