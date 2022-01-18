# coding=gbk
import open3d as o3d  # 导入open3d库
import numpy as np  # 导入numpy

points = np.loadtxt(r'D:\Dataset\shapenetcore_partanno_segmentation_benchmark_v0\shapenetcore_partanno_segmentation_benchmark_v0\02773838\points\8ea3fa037107ec737426c116c412c132.pts')
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([point_cloud])