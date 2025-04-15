Instruction for installing and running the project:
To run the project, first clone the project from git: https://github.com/petertran410/yolov8.git

1. After clone the project, go the folder that have just cloned.
2. Create the environment by running: python -m venv venv
3. Activate the environment:

- For Win: venv\Scripts\activate
- For Linux/Mac: venv/bin/activate

4. Install the pip packages in file requirement.txt by running:
   pip install -r requirements.txt
5. There are 3 models, you can change the model but changing in the file app.py.
   At the: model = YOLO(‘models/best.pt’), you can change it into default or warp.
6. If you want to use model through your camera and detect live by using camera the uncomment the 2 line of code which are results and print(results).
   If you want to detect the image that you want to, then keep the line of code which are input_path and results = model(input_path).
   Make sure you copy the right path that paste it into: input_path = “paste the path in here”
7. After set up the path and model, run the command: python app.py
   This command will run the demo of the project.
   If you choose detect but a single image that you want, then configure the filename in result.save which is the name that you want to save after detecting.
