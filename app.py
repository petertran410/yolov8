import torch
from ultralytics import YOLO
from ultralytics import settings

model = YOLO('models/best.pt')

# results = model.predict(source="0", show=True)
# print(results)


input_path = 'Put your path image in here'

results = model(input_path)

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.save(filename="Change name of save image.jpg")
    print(result)
