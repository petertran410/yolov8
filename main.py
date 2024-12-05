from ultralytics import YOLO
import torch
import copy

model = YOLO('yolov8m.pt')

path = '/StableV3/data.yaml'

def put_in_eval_mode(trainer, n_layers=8):
    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split('.')[1]) < n_layers:
            module.eval()
            module.track_running_stats = False
            print(name, " put in eval mode.")
            
model.add_callback("on_train_epoch_start", put_in_eval_mode)

model.add_callback("on_pretrain_routine_start", put_in_eval_mode)

results = model.train(data=path, 
                      freeze=8, 
                      imgsz=640,
                      patience=0,
                      epochs=150,  
                      device=[0,1], 
                      name="FineTune-8Layer", 
                      batch=16,
                      workers=8,
                      plots=True,
                      optimizer="Adam",
                      momentum=0.95,
                      warmup_epochs=2,
                      warmup_momentum=0.5,
                      val=True,
                      close_mosaic=5,
                      cos_lr=True,
                      verbose=True,
                      exist_ok=True)