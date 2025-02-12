import tensorflow as tf
import sys
import cv2

from yolo_v3 import Yolo_v3
from utils import load_images, load_class_names, draw_boxes, draw_frame

_MODEL_SIZE = (416, 416)
_CLASS_NAMES_FILE = './data/labels/coco.names'
_MAX_OUTPUT_SIZE = 20


def main(mode, iou_threshold, confidence_threshold, input_names):
    class_names = load_class_names(_CLASS_NAMES_FILE)
    n_classes = len(class_names)

    model = Yolo_v3(n_classes=n_classes, model_size=_MODEL_SIZE,
                    max_output_size=_MAX_OUTPUT_SIZE,
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

    if mode == 'images':
        batch_size = len(input_names)
        batch = load_images(input_names, model_size=_MODEL_SIZE)
        # Using TF 1.x placeholder. If you are on TF 2.x, consider tf.compat.v1.placeholder.
        inputs = tf.placeholder(tf.float32, [batch_size, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

        with tf.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')
            detection_result = sess.run(detections, feed_dict={inputs: batch})

        draw_boxes(input_names, detection_result, class_names, _MODEL_SIZE)
        print('Detections have been saved successfully.')

    elif mode == 'video':
        if len(input_names) < 1:
            print("Error: For video mode, please provide a video file path or camera index.")
            sys.exit(1)

        # If the provided video input is a digit, assume it's a camera index and convert it to an integer.
        video_input = input_names[0]
        if video_input.isdigit():
            video_input = int(video_input)

        inputs = tf.placeholder(tf.float32, [1, *_MODEL_SIZE, 3])
        detections = model(inputs, training=False)
        saver = tf.train.Saver(tf.global_variables(scope='yolo_v3_model'))

        with tf.Session() as sess:
            saver.restore(sess, './weights/model.ckpt')

            win_name = 'Video detection'
            cv2.namedWindow(win_name)

            cap = cv2.VideoCapture(video_input)
            if not cap.isOpened():
                print("Error: Cannot open video/camera stream!")
                sys.exit(1)

            frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                          cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter('./detections/detections.mp4', fourcc, fps,
                                  (int(frame_size[0]), int(frame_size[1])))

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    resized_frame = cv2.resize(frame, dsize=_MODEL_SIZE[::-1],
                                               interpolation=cv2.INTER_NEAREST)
                    detection_result = sess.run(detections,
                                                feed_dict={inputs: [resized_frame]})

                    draw_frame(frame, frame_size, detection_result,
                               class_names, _MODEL_SIZE)

                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    out.write(frame)
            finally:
                cv2.destroyAllWindows()
                cap.release()
                print('Detections have been saved successfully.')
    else:
        raise ValueError("Inappropriate mode. Please choose either 'video' or 'images'.")


if __name__ == '__main__':
    # Expected arguments:
    #   sys.argv[1]: mode ('images' or 'video')
    #   sys.argv[2]: iou_threshold (float)
    #   sys.argv[3]: confidence_threshold (float)
    #   sys.argv[4:]: input file(s) or camera index (for video)
    if len(sys.argv) < 5:
        print("Usage: python detect.py <mode: images|video> <iou_threshold> <confidence_threshold> <input_file(s)/camera_index>")
        sys.exit(1)

    mode = sys.argv[1]
    iou_threshold = float(sys.argv[2])
    confidence_threshold = float(sys.argv[3])
    input_names = sys.argv[4:]
    main(mode, iou_threshold, confidence_threshold, input_names)
