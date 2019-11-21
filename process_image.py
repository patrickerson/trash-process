from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "separa_lixo.h5"
input_path = "teste_train.jpg"
output_path = "imagenew.jpg"

detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])