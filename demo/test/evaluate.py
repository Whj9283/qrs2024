from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()

# 加载预训练模型
model = YOLO(model_path / "class.pt", task='detect')
#***************************视频处理********************************
# 官网：https://docs.ultralytics.com/modes/predict/#inference-arguments

source = "../assets/class1.mp4"

results = model(source, stream=True, show=False, save=False)  # generator of Results objects

# 指定要保存的文件路径
file_path = "../output/output.txt"
with open(file_path, 'w', encoding='utf-8') as file:
    for r in results:
        # 写入每一帧的属性
        print(r.boxes.cls)
        # tensor_str = str(r.boxes.cls)  # 将Tensor转换为字符串
        # file.write(tensor_str + '\n')  # 每个属性后面添加换行符，以区分不同的帧
# 文件已经写入完成
# print(f"所有帧的属性已成功保存到 {file_path}")

#*****************************************************************

#***************************摄像头处理********************************
# cap = cv2.VideoCapture(1) # 0/1表示调用摄像头
# # 设置保存视频
# width  = 640
# height = 640
# fps    = cap.get(5)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter("../output/yx_out.mp4", fourcc, fps, (width, height))

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_resized = cv2.resize(frame, (640, 640))
#     results = model(frame_resized)
#     annotated_frame = results[0].plot()
#     writer.write(annotated_frame)    #写入
#     cv2.imshow('YOLOv8', annotated_frame)
#     if cv2.waitKey(1) == ord('q'):  # 按下q退出
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#*****************************************************************

#***************************识别图片*******************************
# # 读取图片
# img = cv2.imread(asset_path / 'xixi.jpg')
# # 使用cv2.resize()函数调整图片大小
# img_resized = cv2.resize(img, (640, 640))
# # 检测图片
# results = model(img_resized)
# res = results[0].plot()
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)
#****************************************************************

