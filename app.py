import torch
from ultralytics import YOLO
from ultralytics import settings

model = YOLO('models/warp.pt')

# results = model.predict(source="0", show=True)
# print(results)


input_path = 'prepared_data_all_MGS-05-Nov_02-40-55.jpg'

results = model(input_path)

for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.save(filename="File-1.jpg")
    print(result)
