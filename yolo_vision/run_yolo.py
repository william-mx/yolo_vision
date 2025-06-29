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
from ros2_numpy import image_to_np, np_to_compressedimage, to_detection2d_array, to_label_info

from ultralytics import YOLO


# ─── Node Definition ─────────────────────────────────────
class YoloVision(Node):
    def __init__(self, model_dir, model_name):
        super().__init__('yolo_vision')

        # Ensure model directory exists
        if not os.path.exists(model_dir):
            self.get_logger().warn(f"Directory '{model_dir}' not found. Creating it.")
            os.makedirs(model_dir, exist_ok=True)

        # Resolve model path
        self.model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(self.model_path):
            model_name = 'yolo11n.pt'
            self.get_logger().warn(f"Model not found. Using fallback '{model_name}'.")
            self.model_path = os.path.join(model_dir, model_name)

        # Load model
        self.model = self.load_model(self.model_path)

        # Set QoS for sensor data
        qos_profile = qos_profile_sensor_data
        qos_profile.depth = 1

        # Publish detections
        self.detection2d_pub = self.create_publisher(Detection2DArray, '/detections_2d', qos_profile)

        # Subscribe to raw RGB camera image
        self.im_subscriber = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, qos_profile)

        # Publish label mapping once
        qos_transient = QoSProfile(depth=1)
        qos_transient.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.label_pub = self.create_publisher(LabelInfo, '/label_mapping', qos_transient)

        # Publish annotated images
        self.im_publisher = self.create_publisher(CompressedImage, '/result', qos_profile)

        # Set class ID mappings
        self.id2label = self.model.names
        self.label2id = {lbl: id for id, lbl in self.id2label.items()}

        # Publish label mapping message
        self.label_pub.publish(to_label_info(self.id2label))

        self.get_logger().info("YOLO object detection node started.")

    def load_model(self, filepath):
        model = YOLO(filepath)

        self.imgsz = model.args['imgsz'] # Get the image size (imgsz) the loaded model was trained on.

        # Init model
        print("Initializing the model with a dummy input...")
        im = np.zeros((self.imgsz, self.imgsz, 3)) # dummy image
        _ = model.predict(im, verbose = False)  
        print("Model initialization complete.")

        return model

    def image_callback(self, msg):
        image, timestamp_unix = image_to_np(msg)
        predictions = self.model(image, verbose=False)

        success, results = parse_predictions(predictions)
        plot = predictions[0].plot() if success else image.copy()

        msg = to_detection2d_array(results, timestamp_unix)
        self.detection2d_pub.publish(msg)

        im_msg = np_to_compressedimage(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB), timestamp_unix)
        self.im_publisher.publish(im_msg)


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
