import cv2
import os
from os import path as osp
import numpy as np
import sys
data_dir = "/home/data/wudaokou/0807/"
input_views = ""
output_pic_view = "output_dir"
video_dir = "videos"
input = os.path.join(data_dir,input_views)
output = os.path.join(data_dir,output_pic_view)
output_video = os.path.join(data_dir,video_dir)
if not os.path.exists(output):
    os.makedirs(output)
if not os.path.exists(output_video):
    os.makedirs(output_video)
camera_lists = ['FW','FN','FL','FR','RL','RR','RN','RN','pcd_img']
all_pic_path=[]
pic_path = []
image_list = []
rows = 3
cols = 3
width = 2976
height = 2000
print("imcoming")

if not os.path.exists(output):
    os.makedirs(output)
sys.stdout.flush()

cam_list = ['FW','FN','FL','FR','RL','RR','RN','RN', "pcd_img"]
lidar_list = os.listdir(osp.join(data_dir, "pcd_img"))
result = dict()
for item in lidar_list:
    tmp = dict()
    lidar_time = float(item[:-4]) + 1.4
    for cam in cam_list:
        diff = 10000
        for img in os.listdir(osp.join(data_dir, "test", cam)):
            if abs(float(img[:-4])-lidar_time) < diff:
                diff = abs(float(img[:-4])-lidar_time)
                img_path = osp.join(data_dir, "test", cam, img[:-4]+".jpg")
            else:
                continue
        tmp[cam] = img_path
    result[item] = tmp
    
FW = []
FN = []
FL = []
FR = []
RL = []
RR = []
RN = []
pcd_img = []

order = list(result.keys())
order.sort()
for item in order:
    FW.append(result[item]["FW"])
    FN.append(result[item]["FN"])
    FL.append(result[item]["FL"])
    FR.append(result[item]["FR"])
    RL.append(result[item]["RL"])
    RR.append(result[item]["RR"])
    RN.append(result[item]["RN"])
    pcd_img.append(osp.join("/home/data/wudaokou/0807/pcd_img", item))

all_pic_path = [FW, FN, FL, FR, RL, RR, RN, RN, pcd_img]

# for cam in camera_lists:
#     print(cam)
#     pic_path.clear()
#     cam_pic = input + '/' + cam
#     print(cam_pic)
#     view_list = sorted(os.listdir(cam_pic))
#     for filename in view_list[:]:
#         # 检查文件是否是图片
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             pic_path.append(os.path.join(cam_pic, filename))
#     all_pic_path.append(pic_path.copy())
    
length = len(all_pic_path[0])
print("frame num :", length)
for n in range(length):
    try:
        if n >= length:
            continue
        i = -1
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for pic_path in all_pic_path:
            i = i + 1
            row = i // 2
            col = i % 2 
            image_temp=cv2.imread(pic_path[n])
            if i!=8:
                image_temp=cv2.resize(image_temp,(888,500))
                image[row * 500:(row + 1) * 500, col * 2088:(col * 2088 + 888), :] =image_temp
            if i == 8:
                image_temp=cv2.resize(image_temp,(1200,2000))
                image[0:2000, 888:2088, :] =image_temp
        image_last = cv2.resize(image,(width,height))
        output_path = output+"/"+f"{n:04d}"+".jpg"
        cv2.imwrite(output_path,image_last)
    except:
        print(n)

print("----------开始读取图片------------")
sys.stdout.flush()
for img in os.listdir(output):
    if img.endswith(".jpg"):
        image_list.append(img)
image_list.sort()
image = cv2.imread(os.path.join(output,image_list[0]))
height, width, layers = image.shape
print("------------正在创建视频------------")
sys.stdout.flush()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video+'/video.mp4', fourcc, 20, (width, height))
for img in image_list:
    images_path = os.path.join(output,img)
    frame = cv2.imread(images_path)
    out.write(frame)
out.release()
print("----------视频创建完成----------")
sys.stdout.flush()
print("视频已生成")
