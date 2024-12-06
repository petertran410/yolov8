from ultralytics import YOLO
import torch
import copy

model = YOLO('yolov8s.pt')

path = 'StableV5/data.yaml'

def put_in_eval_mode(trainer, n_layers=5):
    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split('.')[1]) < n_layers:
            module.eval()
            module.track_running_stats = False
            print(name, " put in eval mode.")
            
model.add_callback("on_train_epoch_start", put_in_eval_mode)

model.add_callback("on_pretrain_routine_start", put_in_eval_mode)

results = model.train(data=path, 
                      freeze=5,
                      imgsz=640, 
                      epochs=150,
                      patience=0,
                      device=0, 
                      name="FINETUNE-5layer",
                      batch=16, 
                      workers=4, 
                      optimizer="AdamW", 
                      lr0=0.0003,
                      weight_decay=0.0005,
                      warmup_epochs=5,
                      warmup_momentum=0.8,
                      val=True,
                      conf=0.5,
                      nms=True, 
                      cos_lr=True,
                      augment=True, 
                      verbose=True,
                      exist_ok=True,
                      plots=True
                     )
