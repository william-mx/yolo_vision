# ─── Imports ─────────────────────────────────────────────
import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSDurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, LabelInfo
from ament_index_python.packages import get_package_prefix

from yolo_vision.utils import parse_predictions
from ros2_pydata import image_to_np, np_to_compressedimage, to_detection2d_array, to_label_info

from ultralytics import YOLO

# ─── Node Definition ─────────────────────────────────────
class YoloVision(Node):
    def __init__(self, model_dir, model_name):
        super().__init__('yolo_vision')

        # ── Publish-control parameters (default: only plot) ──
        self.declare_parameter('publish_plot',       True)
        self.declare_parameter('publish_mask',       False)
        self.declare_parameter('publish_detections', False)

        self.publish_plot       = self.get_parameter('publish_plot').value
        self.publish_mask       = self.get_parameter('publish_mask').value
        self.publish_detections = self.get_parameter('publish_detections').value

        self.get_logger().info(
            f"Publish flags — plot: {self.publish_plot}, "
            f"mask: {self.publish_mask}, "
            f"detections: {self.publish_detections}"
        )

        # Ensure model directory exists
        if not os.path.exists(model_dir):
            self.get_logger().warn(f"Directory '{model_dir}' not found. Creating it.")
            os.makedirs(model_dir, exist_ok=True)

        # Resolve model path
        self.model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(self.model_path):
            model_name = 'yolo11n-seg.pt'
            self.get_logger().warn(f"Model not found. Using fallback '{model_name}'.")
            self.model_path = os.path.join(model_dir, model_name)

        # Load model
        self.model = self.load_model(self.model_path)

        # Set QoS for sensor data
        qos_profile = qos_profile_sensor_data
        qos_profile.depth = 1

        # ── Conditionally create publishers ──────────────────

        # Label mapping is needed when publishing masks (ints) or detections
        if self.publish_mask or self.publish_detections:
            qos_transient = QoSProfile(depth=1)
            qos_transient.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
            self.label_pub = self.create_publisher(LabelInfo, '/label_mapping', qos_transient)

        if self.publish_detections:
            self.detection2d_pub = self.create_publisher(Detection2DArray, '/detections_2d', qos_profile)

        if self.publish_plot:
            self.im_publisher = self.create_publisher(CompressedImage, '/yolo_overlay', qos_profile)

        if self.publish_mask:
            self.mask_publisher = self.create_publisher(CompressedImage, '/mask', qos_profile)

        # Subscribe to raw RGB camera image
        self.im_subscriber = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, qos_profile)

        # Set class ID mappings — shift model IDs by +1 to reserve 0 for background
        # e.g. model class 0 ('person') → published ID 1, background → 0
        self.id2label = {0: 'background', **{k + 1: v for k, v in self.model.names.items()}}
        self.label2id = {lbl: id for id, lbl in self.id2label.items()}

        self.get_logger().info(f"mapping (shifted): {self.id2label}")

        # Publish label mapping message (needed for mask ints and/or detections)
        if self.publish_mask or self.publish_detections:
            self.label_pub.publish(to_label_info(self.id2label))

        self.get_logger().info("YOLO object detection node started.")

    def load_model(self, filepath):
        model = YOLO(filepath)

        self.imgsz = model.args['imgsz'] # Get the image size (imgsz) the loaded model was trained on.

        # Init model
        self.get_logger().info("Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3)) # dummy image
        _ = model.predict(im, verbose=False)
        self.get_logger().info("Model initialization complete.")

        return model

    def image_callback(self, msg):

        # ros image to numpy
        image, timestamp_unix = image_to_np(msg)

        # run prediction
        predictions = self.model(image, verbose=False)

        # Parse + publish detections and/or mask only if needed
        if self.publish_detections or self.publish_mask:
            success, results, mask = parse_predictions(predictions)

            # Publish Bounding Boxes (Detection2DArray)
            if self.publish_detections:
                self.detection2d_pub.publish(to_detection2d_array(results, timestamp_unix))

            # Publish Mask (Compressed PNG)
            if self.publish_mask:
                self.mask_publisher.publish(np_to_compressedimage(mask, timestamp_unix))

        # Publish Debug Plot (Compressed JPEG)
        if self.publish_plot:
            plot = predictions[0].plot()
            plot_msg = np_to_compressedimage(plot, timestamp_unix)
            self.im_publisher.publish(plot_msg)


# ─── Main ────────────────────────────────────────────────
def main(args=None):
    pkg_path = get_package_prefix('yolo_vision').replace('install', 'src')
    model_path = pkg_path + '/models'
    model_name = 'best.pt'

    rclpy.init(args=args)
    node = YoloVision(model_path, model_name)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt: shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()