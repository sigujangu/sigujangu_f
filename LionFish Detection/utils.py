import openvino as ov
import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from PIL import Image

# Load OpenVINO models for face detection and emotion recognition
core = ov.Core()

model_face = core.read_model(model="models/face-detection-adas-0001.xml")
compiled_model_face = core.compile_model(model=model_face, device_name="CPU")

input_layer_face = compiled_model_face.input(0)
output_layer_face = compiled_model_face.output(0)

model_emo = core.read_model(model="models/emotions-recognition-retail-0003.xml")
compiled_model_emo = core.compile_model(model=model_emo, device_name="CPU")

input_layer_emo = compiled_model_emo.input(0)
output_layer_emo = compiled_model_emo.output(0)

# Preprocess the image for model input
def preprocess(image, input_layer):
    N, input_channels, input_height, input_width = input_layer.shape
    resized_image = cv2.resize(image, (input_width, input_height))
    transposed_image = resized_image.transpose(2, 0, 1)
    input_image = np.expand_dims(transposed_image, 0)
    return input_image

# Find face boxes from the model output
def find_faceboxes(image, results, confidence_threshold):
    results = results.squeeze()
    scores = results[:, 2]
    boxes = results[:, -4:]
    face_boxes = boxes[scores >= confidence_threshold]
    scores = scores[scores >= confidence_threshold]
    image_h, image_w, image_channels = image.shape
    face_boxes = face_boxes * np.array([image_w, image_h, image_w, image_h])
    face_boxes = face_boxes.astype(np.int64)
    return face_boxes, scores

# Draw face boxes on the image
def draw_faceboxes(image, face_boxes, scores):
    show_image = image.copy()
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        cv2.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 200, 0), thickness=2)
    return show_image

# Draw emotion on the image
def draw_emotion(face_boxes, image):
    EMOTION_NAMES = 'LionFish'
    show_image = image.copy()
    for i in range(len(face_boxes)):
        xmin, ymin, xmax, ymax = face_boxes[i]
        face = image[ymin:ymax, xmin:xmax]
        input_image = preprocess(face, input_layer_emo)
        results_emo = compiled_model_emo([input_image])[output_layer_emo]
        results_emo = results_emo.squeeze()
        index = np.argmax(results_emo)
        label = EMOTION_NAMES
        fontScale = image.shape[1] / 750
        text = f"{label}"
        cv2.putText(show_image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 200, 0), 2)
        cv2.rectangle(img=show_image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 200, 0), thickness=2)
    return show_image

# Predict emotion for faces in the image
def predict_image(image, conf_threshold):
    input_image = preprocess(image, input_layer_face)
    results = compiled_model_face([input_image])[output_layer_face]
    face_boxes, scores = find_faceboxes(image, results, conf_threshold)
    visualize_image = draw_emotion(face_boxes, image)
    detected = len(face_boxes) > 0  # Check if any face boxes are detected
    return visualize_image, detected

# Load the Faster R-CNN model for lionfish detection
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# Make predictions for the lionfish detection model
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return img

def make_prediction(img, model):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img).permute(2, 0, 1)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, 
        boxes=prediction["boxes"], 
        labels=prediction["labels"],
        colors=["red" if label == "lionfish" else "green" for label in prediction["labels"]], 
        width=2
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

# Detect lionfish in the video frame
def detect_lionfish_in_frame(frame):
    img = Image.fromarray(frame)
    prediction = make_prediction(img, model)
    img_with_bbox = create_image_with_bboxes(np.array(img), prediction)
    lionfish_detected = any(label == "lionfish" for label in prediction["labels"])
    return img_with_bbox, lionfish_detected
