# improt odpowiednich modułów
from copyreg import pickle
import open3d as o3d
import pickle, torch, os, time
import numpy as np
import os
import pcd_utilities as util
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

DIR = os.getcwd()

model_path = os.path.join(DIR, "pointnet_model_pytorch", "saved_model", "pointnet_model")
path = os.path.join(DIR, "15_12_21PointClouds", "firstYLBR")

# Stałe
DOWNSAMPLING_VOXEL_VALUE = 0.04
OUTLIER_REMOVE_N_NEIGHBOURS=3
OUTLIER_REMOVE_STD_RATIO=1.2

roiParams = [15, 0, 2, -2, 1, -0.10]

DISTANCE_TH_RANSAC = 0.02
RANSAC_N = 100
N_ITERATIONS_RANSAC = 20

CLUSTER_EPS = 0.08
CLUSTER_MIN_POINTS=5

N_POINTS = 2500
# Wczytanie plik

def remove_ground_main(pcdNoGround):
    pcdNoGround, _ = util.removeGround(pcdNoGround)
    return pcdNoGround

def pcd_processing(pcd):
    functionStart = time.time()
    pcdMain = util.extractROI(pcd, util.roiParams)
    pcdNoGround = pcdMain

    pcdMain = remove_ground_main(pcdNoGround)

    print("Removed ground, took:", time.time() - functionStart)
    functionStart = time.time()

    """Removing outliers:"""
    pcdMain = close_far_split_cleanup(pcdMain)

    """Extracting clusters:"""
    flatPCDToPerformClustering = util.flattenPCD(pcdMain)
    clusterLabels = np.array(flatPCDToPerformClustering.cluster_dbscan(eps=0.1, min_points=5))

    print("Extracted clusters, took:", time.time() - functionStart)
    functionStart = time.time()

    """Paining clusters(could be removed):"""
    maxClusterLabel = clusterLabels.max()
    colors = plt.get_cmap("tab20")(clusterLabels / (maxClusterLabel if maxClusterLabel > 0 else 1))
    colors[clusterLabels < 0] = 0
    pcdMain.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    def extract_clusters(pcd, labels):
        clusters = np.array([])
        clusters_real = []
        for label in np.unique(labels):
            clusterIndexes = np.where(clusterLabels == label)[0]
            points = np.asarray(pcd.select_by_index(clusterIndexes).points)
            choice = np.random.choice(points.shape[0], N_POINTS, replace=True)
            points = points[choice, :]
            points = points - np.expand_dims(np.mean(points, axis = 0), 0) 
            dist = np.max(np.sqrt(np.sum(points ** 2, axis = 1)),0)
            points = points / dist #scale
            points = np.expand_dims(points, axis=0)
            points_real = pcd.select_by_index(clusterIndexes).points
            if clusters.size == 0:
                clusters = points
                clusters_real.append(points_real)
            else:
                clusters = np.vstack((clusters, points))
                clusters_real.append(points_real)
                
        return clusters, clusters_real
    
    clusters, clusters_real = extract_clusters(pcdMain, clusterLabels)

    showpcd(pcdMain, "Pcd after clustering")
    return torch.from_numpy(clusters), clusters_real


def close_far_split_cleanup(pcdMain):  # TODO optimize this as much as possible
    """Split pcd into far and close parts:"""
    closeFarDistance = 4
    pcdClose, closeInd = util.getPointsInRadius(pcdMain, closeFarDistance, removeInRadius=False,
                                                isCylinder=True)  # using default 000 origin
    pcdFar = pcdMain.select_by_index(closeInd, invert=True)
    """We will work on flat PCD's:"""
    pcdCloseFlat = util.flattenPCD(pcdClose)
    pcdCloseFlat, ind = pcdCloseFlat.remove_radius_outlier(nb_points=8, radius=0.025,
                                                        print_progress=False)  # radius outlier
    # display_inlier_outlier(pcdClose, ind)
    """Remove outliers from 3D cloud"""
    pcdClose = pcdClose.select_by_index(ind)
    # showpcd(pcdClose,"pcdClose first filter")

    _, ind = pcdCloseFlat.remove_statistical_outlier(nb_neighbors=2, std_ratio=20,
                                                    print_progress=False)  # statistical outlier
    # display_inlier_outlier(pcdClose, ind)
    pcdClose = pcdClose.select_by_index(ind)
    # showpcd(pcdClose,"pcdClose second filter")

    # clean up the far pcd:
    pcdFarFlat = util.flattenPCD(pcdFar)
    _, ind = pcdFarFlat.remove_radius_outlier(nb_points=4, radius=0.05)  # radius outlier
    # display_inlier_outlier(pcdFar, ind)

    # below step is necessary because open3d sucks and does not work
    # pcdClose, closeInd = getPointsInRadius(pcdMain, closeFarDistance, removeInRadius=False, isCylinder=True) # using default 000 origin
    """Remove outliers from 3D cloud"""
    pcdFar = pcdFar.select_by_index(ind, invert=False)
    # !TODO check if the line does not split any cone. STD of points in some range from the detector below the thresh for
    # object being there. If the cone is in there, move border closer, to apply less strict rules to the cones too far.
    # showpcd(pcdMain,'Main point cloud without outliers')

    mainPCD = pcdClose + pcdFar
    return mainPCD


def showpcd(pcdin, pcdName):
    o3d.visualization.draw_geometries([pcdin], window_name=pcdName)

# Wczytaj model
def load_model(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = torch.load(model_path)
readings_t = []
segmentations_t = []
predictions_t = []
full_time_t = []
i = 0
for element in os.listdir(path)[:10]:
    trained_path = os.path.join(path, element)
    pcd = o3d.io.read_point_cloud(trained_path)
    clusters, clusters_real  = pcd_processing(pcd)
    clusters = clusters.float()
    clusters = clusters.transpose(2, 1)
    clusters = clusters.cuda()
    model = model.eval()
    pred, _, _ = model(clusters)
    pred_choice = pred.data.max(1)[1]
    cones_id = pred_choice.nonzero(as_tuple=True)
    test = torch.arange(0,len(clusters)).cuda()
    i+=1
    for i in cones_id[0]:
        test = test[test != i]
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for cone in test:
        o3d_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(clusters_real[cone])
        bbox = np.asarray(o3d_bbox.get_box_points())
        # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc..
        lines = [[0, 1],[0, 3], [1, 7],
                [5, 4], [1, 6], [3, 6],
                [4, 5], [4, 7], [5, 2],
                [2, 0], [2, 7], [3, 5],
                [4, 6]]

        # Use the same color for all lines
        colors = [[1, 0, 0] for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        # Display the bounding boxes:
        vis.add_geometry(line_set)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
print(f"Sredni czas wczytywania pliku jednej iteracji: {np.mean(readings_t)}")
print(f"Sredni czas przeprowadzania operacji segmentacji: {np.mean(segmentations_t)}")
print(f"Sredni czas przeprowadzania predykcji: {np.mean(predictions_t)}")
print(f"Sredni czas przeprowadzenia jednego procesu: {np.mean(full_time_t)}")