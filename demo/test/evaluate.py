from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()

# 加载预训练模型
model = YOLO(model_path / "people_5.pt", task='detect')

cap = cv2.VideoCapture(1) # 0/1表示调用摄像头
# cap = cv2.VideoCapture('../assets/555.mp4')
# # 设置保存视频
# width  = 640
# height = 640
# fps    = cap.get(5)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# writer = cv2.VideoWriter("../output/555_out.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (640, 640))
    results = model(frame_resized)
    annotated_frame = results[0].plot()
    # writer.write(annotated_frame)    #写入
    cv2.imshow('YOLOv8', annotated_frame)
    if cv2.waitKey(1) == ord('q'):  # 按下q退出
        break

cap.release()
cv2.destroyAllWindows()
#
# 读取图片
# img = cv2.imread(asset_path / 'xixi.jpg')
# # 使用cv2.resize()函数调整图片大小
# img_resized = cv2.resize(img, (640, 640))
# # 检测图片
# results = model(img_resized)
# res = results[0].plot()
# cv2.imshow("YOLOv8 Inference", res)
# cv2.waitKey(0)