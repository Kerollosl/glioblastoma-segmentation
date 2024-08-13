from ultralytics import YOLO
import cv2
import os

def predict_glioblastoma(image_path, resize_shape=(840, 540), conf=0.25):

    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, resize_shape)

    if frame is None:
        print("Error: Could not read the image.")
    else:
        model.predict(frame, conf=conf, show=True, show_conf=False, show_labels=False, show_boxes=False)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


# Load a model
# weights_path = './yolov8n.pt'
weights_path = 'runs/segment/train5/weights/best.pt'


model = YOLO(weights_path)
print("Model successfully loaded")

# Predict for test images
for img in os.listdir("./test_images"):
    predict_glioblastoma(f"./test_images/{img}", resize_shape=(1200, 800), conf=0.2)
