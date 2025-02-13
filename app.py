import PIL

import streamlit as st
from ultralytics import YOLO

model = YOLO('models/best.pt')