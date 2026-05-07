from roboflow import Roboflow

rf = Roboflow(api_key="jFxNKhqK5eJKo4zoqZVc")
project = rf.workspace("acne-vulgaris-detection").project("acne04-detection")
version = project.version(1)
dataset = version.download("yolov5")