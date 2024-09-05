import os
import cv2
import open3d as o3d
import yaml
import numpy as np
import argparse
import os
from os import path as osp


def read_camera_intrinsics(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    camera_matrix = {
        "rows": data['camera_matrix']['rows'],
        "cols": data['camera_matrix']['cols'],
        "data": data['camera_matrix']['data']
    }

    distortion_coefficients = {
        "rows": data['distortion_coefficients']['rows'],
        "cols": data['distortion_coefficients']['cols'],
        "data": data['distortion_coefficients']['data']
    }

    R = {
        "rows": data['R']['rows'],
        "cols": data['R']['cols'],
        "data": data['R']['data']
    }

    P = {
        "rows": data['P']['rows'],
        "cols": data['P']['cols'],
        "data": data['P']['data']
    }

    mat_in_K = np.array(camera_matrix['data']).reshape(camera_matrix['rows'], camera_matrix['cols'])
    mat_distort = np.array(distortion_coefficients['data']).reshape(distortion_coefficients['rows'],
                                                                    distortion_coefficients['cols'])
    mat_in_R = np.array(R['data']).reshape(R['rows'], R['cols'])
    mat_in_P = np.array(P['data']).reshape(P['rows'], P['cols'])
    return mat_in_K, mat_distort, mat_in_R, mat_in_P


def read_camera_extrinsics(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    rotation_matrix = {
        "rows": data['rotation_matrix']['rows'],
        "cols": data['rotation_matrix']['cols'],
        "data": data['rotation_matrix']['data']
    }
    translation_vector = {
        "rows": data['translation_vector']['rows'],
        "cols": data['translation_vector']['cols'],
        "data": data['translation_vector']['data']
    }
    # 获取 rotation_matrix 和 translation_vector 中的 data
    # rotation_matrix_data = data['rotation_matrix']['data']
    # translation_vector_data = data['translation_vector']['data']

    mat_r = np.array(rotation_matrix['data']).reshape(rotation_matrix['rows'], rotation_matrix['cols'])
    mat_t = np.array(translation_vector['data']).reshape(translation_vector['rows'], translation_vector['cols'])
    return mat_r, mat_t


def scale_to_range(value, min_val=0, max_val=100, new_min=0, new_max=255):
    # 计算原始范围和新范围的比例
    scale = (new_max - new_min) / (max_val - min_val)

    # 计算缩放后的值
    new_value = new_min + scale * (value - min_val)

    # 确保结果在新范围内（处理可能的浮点运算误差）
    new_value = max(new_min, min(new_max, new_value))

    return new_value


def jet_color_map(gray):
    if gray < 0 or gray > 255:
        return (0, 0, 0)

    if gray <= 31:
        r = 0
        g = 0
        b = 128 + 4 * (gray - 0)
    elif gray == 32:
        r = 0
        g = 0
        b = 255
    elif gray <= 95:
        r = 0
        g = 4 * (gray - 32)
        b = 255
    elif gray == 96:
        r = 2
        g = 255
        b = 254
    elif gray <= 158:
        r = 6 + 4 * (gray - 97)
        g = 255
        b = 250 - 4 * (gray - 97)
    elif gray == 159:
        r = 254
        g = 255
        b = 1
    elif gray <= 223:
        r = 255
        g = 252 - 4 * (gray - 160)
        b = 0
    elif gray <= 255:
        r = 252 - 4 * (gray - 224)
        g = 0
        b = 0
    else:
        # This else block is redundant due to the initial range check
        r = 0
        g = 0
        b = 0

    return (r, g, b)


def project(img_bgr, pcd, mat_cam_K, vec_cam_distort, transformation_matrix, draw_pt_radius, project_result_name):
    point_cloud_np = np.asarray(pcd.points)
    point_cloud_np = np.hstack([point_cloud_np, np.ones((point_cloud_np.shape[0], 1))])  # 齐次坐标形式
    point_cloud_np = np.dot(transformation_matrix, point_cloud_np.T).T[:, :3]  # [:, :3] 切片
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    points_2d, _ = cv2.projectPoints(point_cloud_np, rvec, tvec, mat_cam_K, vec_cam_distort)
    for point, point_3d in zip(points_2d, point_cloud_np):
        x, y = point.ravel()
        # 投影点云深度默认0-100m
        if 0 <= x < 10000000 and 0 <= y < 100000 and point_3d[2] > 5 and point_3d[2] < 100:
            # 对应的point_cloud_np的z值需要大于0
            # print(point)
            color_val = scale_to_range(point_3d[2])
            color_bgr = jet_color_map(color_val)
            cv2.circle(img_bgr, (int(x), int(y)), draw_pt_radius, color_bgr, -1)
    print(project_result_name)
    cv2.imwrite(project_result_name, img_bgr)


def calib_param_convert_switch(R_A2B, t_A2B):
    ## method1
    # B2A2 = np.eye(4)
    # B2A2[:3, :3] = np.linalg.inv(R_A2B)#np.linalg.inv(R_A2B)#cam2lidar -> lidar2cam
    # B2A2[:3, 3] = (np.dot(np.linalg.inv(R_A2B),t_A2B)*-1).flatten() #t_A2B[:, 0]        cam2lidar -> lidar2cam
    # print("transformation_matrix_cam2lidar:\n",B2A2)

    ## method2 equals to method1
    A2B = np.eye(4)
    A2B[:3, :3] = R_A2B  # np.linalg.inv(R_A2B)#cam2lidar -> lidar2cam
    A2B[:3, 3] = (t_A2B).flatten()  # t_A2B[:, 0]        cam2lidar -> lidar2cam
    B2A = np.linalg.inv(A2B)
    return B2A


def saveExtrinsicsXml(filename, transformation_matrix):
    # 创建FileStorage对象
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

    # 写入数据
    fs.write("tranformation_matrix", transformation_matrix)

    # 重要：关闭FileStorage对象
    fs.release()


def saveIntrinsicsXml(filename, dist, K, R, P):
    # 创建FileStorage对象
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    fs.write("dist", dist)
    fs.write("K", K)
    fs.write("R", R)
    fs.write("P", P)
    fs.release()


def find_coop_img_frame(ts_lidar, img_dir):
    # 将字符串时间戳转换为整数
    ts_lidar_int = int(ts_lidar)

    # 初始化最小差值变量和最匹配的文件名
    min_diff = float('inf')
    closest_img = None

    # 遍历给定文件夹内的所有文件
    for filename in os.listdir(img_dir):
        # 确保文件是.jpg格式
        if filename.endswith('.jpg'):
            # 获取不带扩展名的文件名，即时间戳部分
            timestamp = filename[:-4]

            try:
                # 将时间戳转换为整数
                timestamp_int = int(timestamp)

                # 计算当前文件时间戳与给定时间戳的差值
                diff = abs(timestamp_int - ts_lidar_int)

                # 更新找到的最小差值和相应的文件名
                if diff < min_diff and diff < 10000000:
                    min_diff = diff
                    closest_img = filename
                    # print("time sync diff----------:",min_diff)
            except ValueError:
                # 如果转换失败，则跳过当前文件
                continue

    # 返回最接近的图片时间戳
    return closest_img[:-4]


def matchLidarCam(lidarpath, campath, time_threshold):
    pcdfiles = [f for f in os.listdir(lidarpath)]  # if f.endwith(".pcd")
    imgfiles = [f for f in os.listdir(campath)]  # if f.endwith(".jpg")
    pcdfiles.sort()
    imgfiles.sort()

    pcdtimestamps = np.array([float(f.rsplit(".", 1)[0]) for f in pcdfiles])
    imgtimestamps = np.array([float(f.rsplit(".", 1)[0]) for f in imgfiles])
    # breakpoint()
    match_pcds = []
    match_imgs = []
    match_diff_s = []

    for pcd_time in pcdtimestamps:
        time_diff = np.abs(imgtimestamps - pcd_time)  # list
        min_diff_idx = np.argmin(time_diff)
        min_diff = time_diff[min_diff_idx]

        if (min_diff < time_threshold):
            match_pcds.append(pcdfiles[np.where(pcdtimestamps == pcd_time)[0][0]])
            match_imgs.append(imgfiles[min_diff_idx])
            match_diff_s.append(min_diff)

    return match_pcds, match_imgs, match_diff_s


def read_params(exfile, infile):
    fs = cv2.FileStorage(infile, cv2.FILE_STORAGE_READ)
    K = fs.getNode("K").mat()
    dist = fs.getNode("dist").mat()
    fs.release()

    fs = cv2.FileStorage(exfile, cv2.FILE_STORAGE_READ)
    transformation_matrix = fs.getNode("tranformation_matrix").mat()
    fs.release()

    if K is None or dist is None or transformation_matrix is None:
        print(f"Failed to read parameters from {infile} or {exfile}")
        return None, None, None

    return K, dist, transformation_matrix


def match(root_path, lidar_path, compensation=0.0):
    cam_list = list(cam_func_list.keys())  # 'FN','FL','FR','RL','RR','RN','RN'
    lidar_list = os.listdir(osp.join(root_path, lidar_path))
    result = dict()
    for item in lidar_list:
        tmp = dict()
        lidar_time = float(item[:-4]) + compensation
        for cam in cam_list:
            diff = 10000
            for img in os.listdir(osp.join(root_path, cam)):
                if abs(float(img[:-4]) - lidar_time) < diff:
                    diff = abs(float(img[:-4]) - lidar_time)
                    img_path = osp.join(root_path, cam, img)
                else:
                    continue
            tmp[cam] = img_path
        result[item] = tmp
    return result


def search_near_time(lidar_path, image_path):
    lidar_time = float(lidar_path[-24:-4])
    cam_folder = sorted(os.listdir(image_path))
    cam_result = []
    while len(cam_result) != 20:
        min_time = 100
        index = -1
        for i in range(len(cam_folder)):
            cam_time = float(cam_folder[i][:-4])
            if abs(cam_time - lidar_time) < min_time:
                min_time = abs(cam_time - lidar_time)
                index = i
        cam_result.append(index)
        for j in range(1, 10):
            cam_result.append(index + j)
    return [cam_folder[item] for item in cam_result]


if __name__ == "__main__":
    # purser = argparse.ArgumentParser(description="")
    # purser.add_argument("--bag_name",type=str,default="",help="ros包名,无需后缀")
    # purser.add_argument("--cam_view",type=str,default="/home/data/workshop/163741/FN",help="投影视角选择")
    # purser.add_argument("--lidar_dir",type=str,default="/home/xcc/data/rosbag/cloudpoints/0801-1/pcd",help="点云文件夹")
    # purser.add_argument("--param_dir",type=str,default="/home/data/workshop/163741/opencv_format_param/9006",help="投影cam参数文件夹的选择")
    # purser.add_argument("--draw_pt_radius",type=int,default=5,help="投影点绘制半径")
    # purser.add_argument("--project_dir",type=str,default="/home/xcc/data/rosbag/cloudpoints/0801-1/test/RN",help="输出投影结果文件夹")

    # args = purser.parse_args()
    # bag_name = args.bag_name
    # cam_view = args.cam_view
    # lidar_dir = args.lidar_dir
    # param_dir = args.param_dir
    # project_dir = args.project_dir
    # draw_pt_radius = args.draw_pt_radius
    cam_view = "/home/xcc/data/rosbag/cloudpoints/0816"
    lidar_dir = "/home/xcc/data/rosbag/cloudpoints/0816/pcd"
    param_dir = "/home/data/workshop/163741/opencv_format_param"
    project_dir = "/home/xcc/data/rosbag/cloudpoints/0816/test"
    draw_pt_radius = 2
    cam_func_list = {"FW": "9000",
                     "FL": "9002",
                     "FR": "9003",
                     "RL": "9004",
                     "RR": "9005",
                     "RN": "9006",
                     "FN": "9001"}

    # pcd_path = os.listdir(lidar_dir)
    # lidar_path = os.path.join(lidar_dir, "1723778825.500083208.pcd")
    # cam = "FL"
    # image_path = os.path.join(cam_view, cam)
    # img_folder = os.listdir(image_path)
    # img_folder = search_near_time(lidar_path, image_path)
    # for item in img_folder:
    #     img_bgr = cv2.imread(os.path.join(image_path, item))
    #     pcd = o3d.io.read_point_cloud(lidar_path)
    #     # read param
    #     param_in_path = os.path.join(param_dir, cam_func_list[cam], "{}_intrinsic.xml".format(cam_func_list[cam]))
    #     param_ex_path = os.path.join(param_dir, cam_func_list[cam], "{}_extrinsic.xml".format(cam_func_list[cam]))
    #     # check输出投影图像的文件
    #     project_frame_lidar = os.path.join(image_path, item)[-24:-4]
    #     project_result_dir = f"{project_dir}/{cam}"
    #     if not os.path.exists(project_result_dir):
    #         os.makedirs(project_result_dir)
    #     project_result_name = f"{project_result_dir}/{project_frame_lidar}.jpg"

    #     mat_cam_K, vec_cam_distort, transformation_matrix_lidar2cam = read_params(param_ex_path, param_in_path)

    #     img_bgr = cv2.undistort(img_bgr, mat_cam_K, vec_cam_distort)
    #     project(img_bgr, pcd, mat_cam_K, vec_cam_distort, transformation_matrix_lidar2cam, draw_pt_radius,
    #             project_result_name)

    match_relation = match(cam_view, lidar_dir, compensation=1.4)
    
    for item in list(match_relation.keys()):
        pcd_path = lidar_dir + "/" + item
        for cam in cam_func_list.keys():
            image_path = match_relation[item][cam]
    
            print(image_path)
            print(pcd_path)
            img_bgr = cv2.imread(image_path)
            pcd = o3d.io.read_point_cloud(pcd_path)
    
            # read param
            param_in_path = os.path.join(param_dir, cam_func_list[cam], "{}_intrinsic.xml".format(cam_func_list[cam]))
            param_ex_path = os.path.join(param_dir, cam_func_list[cam], "{}_extrinsic.xml".format(cam_func_list[cam]))
            # check输出投影图像的文件
            project_frame_lidar = image_path[-24:-4]
            project_result_dir = f"{project_dir}/{cam}"
            if not os.path.exists(project_result_dir):
                os.makedirs(project_result_dir)
            project_result_name = f"{project_result_dir}/{project_frame_lidar}.jpg"
    
            mat_cam_K, vec_cam_distort, transformation_matrix_lidar2cam = read_params(param_ex_path, param_in_path)
    
    
            img_bgr = cv2.resize(img_bgr, (3840, 2160), interpolation=cv2.INTER_LINEAR)
            # img_bgr = cv2.undistort(img_bgr, mat_cam_K, vec_cam_distort)
            project(img_bgr, pcd, mat_cam_K, vec_cam_distort, transformation_matrix_lidar2cam, draw_pt_radius,
                    project_result_name)
