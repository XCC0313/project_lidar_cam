import os
import open3d as o3d
# import pypcd
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def runm(pcd, save_name):
    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 创建一个不可见的窗口
    # save_name = os.path.join(save_folder, name+".png")

    # 将点云添加到可视化窗口
    vis.add_geometry(pcd)

    # 渲染点云
    # vis.poll_events()
    # vis.update_renderer()

    # 获取渲染的图像
    # image = vis.capture_screen_float_buffer(do_render=True)

    # 关闭可视化窗口
    # vis.destroy_window()
    # vis.set_full_screen(True)
    render_options = vis.get_render_option()
    render_options.point_size = 1
    render_options.light_on = False 
    ctr = vis.get_view_control()

    ctr.set_lookat(np.array([0, 0, 0]))
    ctr.set_up((1, 0, 1.3))  # 指向屏幕上方的向量；   1,0，0表示 lidar的x轴与pic的x轴平行，并且指向上方  0,1,0 表示lidarx轴指向图像y轴，沿屏幕向上
    ctr.set_front((-1.3, 0, 1))  # 垂直指向屏幕外的向量；初始值0,0,1 表示指向z轴,垂直屏幕
    ctr.set_zoom(0.2) # 控制远近（视野放大缩小）
    ctr.set_constant_z_near(1) 

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    # vis.run()
    # vis.capture_screen_image(save_name) # 保存当前画面
    

    # 将浮点缓冲区转换为 8 位图像
    image = vis.capture_screen_float_buffer(do_render=True)
    image = (255 * np.asarray(image)).astype(np.uint8)
    # save_name = os.path.join(save_folder, name+".png")
    cv2.imwrite(save_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # save_name23 = os.path.join(save_folder2, name+"_1.png")
    # cv2.imwrite(save_name23, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    vis.destroy_window()
    return

ldiar_folder = "/home/xcc/data/rosbag/cloudpoints/0816"
lidar_paths = glob.glob("{:s}/**/*.pcd".format(ldiar_folder), recursive=True)
save_folder = "/home/xcc/data/rosbag/cloudpoints/0816/pcdimg"
os.makedirs(save_folder, exist_ok=True)
for lpath in lidar_paths[:]:
    print(lpath)
    name = os.path.split(lpath)[1].split('.pcd')[0]
    
    if not os.path.exists(lpath):
        print(f"error")
    pcd = o3d.io.read_point_cloud(lpath)
    points = np.asarray(pcd.points)
    save_name = os.path.join(save_folder, name+".png")
    print(save_name)
    runm(pcd, save_name)
