import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def showpcd(pcdin, pcdName):
    o3d.visualization.draw_geometries([pcdin], window_name=pcdName)
DISTANCE_TH_RANSAC = 0.025  # distance threshold of RANSAC algorithm
# was 0.02 when calculating no of points for cone at distance
'''RANSAC_N = 3
N_ITERATIONS_RANSAC = 200'''

RANSAC_N = 3
N_ITERATIONS_RANSAC = 100

CLUSTER_EPS = 0.1
CLUSTER_MIN_POINTS = 15  # minimal number of points in cluster

"The lower the number, the higher the tollerance for tall objects:"
SENSOR_TO_GROUND_DISTANCE = 0.2  # blue cart
# SENSOR_TO_GROUND_DISTANCE = 0.23  # G3 cart
# SENSOR_TO_GROUND_DISTANCE = ???  # bolid ????

IS_CONE_STD_MIN = 0.1
IS_CONE_STD_MAX = 1.0
IS_CONE_SIZE_MIN = 0.20
IS_CONE_SIZE_MAX = 0.6
IS_CONE_MAX_OUTLIER_NUMBER = 150

CONE_RADIUS = 0.15
MAX_CONE_HEIGHT = 0.6

DOWNSAMPLING_VOXEL_VALUE = 0.02
OUTLIER_REMOVE_N_NEIGHBOURS = 3
OUTLIER_REMOVE_STD_RATIO = 1.2

# Format, x max, x min, y max... z min
roiParams = [15, 0, 2, -2, 1, -0.4]
roiExclude = [1, 0, 0.75, -0.75, 10, -10]

'''
# Do not use crop method, it's awfully slow
valsMin = ['0.0', '-2.0', '-0.4']
roiMin = np.array(valsMin, dtype=np.float64)
print(roiMin)
valsMax = ['15.0', '2.0', '1.0']
roiMax = np.array(valsMax, dtype=np.float64)
print(roiMax)
ROI = o3d.geometry.AxisAlignedBoundingBox(min_bound=roiMin, max_bound=roiMax)
print(ROI)
'''


def flattenPCD(pcdToFlatten):
    pointsARRAY = np.copy(np.asarray(pcdToFlatten.points))
    pointsARRAY[:, 2] = 0
    pcdFlat = o3d.geometry.PointCloud()
    O3DvectorForAssigningPoints = o3d.utility.Vector3dVector(pointsARRAY)
    pcdFlat.points = O3DvectorForAssigningPoints
    return pcdFlat


def removeGround(pcdin):
    _, inliers = pcdin.segment_plane(distance_threshold=DISTANCE_TH_RANSAC, ransac_n=RANSAC_N,
                                    num_iterations=N_ITERATIONS_RANSAC)
    inlier_cloud = pcdin.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = pcdin.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1, 0])
    print("Removing ", len(inliers), " ground points")
    return outlier_cloud, inlier_cloud

'''def removeGroundCpp(pcdNoGround):
    # pcdNoGround, groundPcd = util.removeGround(pcdNoGround)
    tic = time.time()
    pcdNoGroundArr = np.copy(np.asarray(pcdNoGround.points))
    pointNum, _ = pcdNoGroundArr.shape
    b = np.zeros((pointNum, 3 + 1))
    b[:, :-1] = pcdNoGroundArr
    # Passing XYZI with dummy I to the ground removal
    distance_th = 0.08
    iter = 200
    pcdNoGroundArrCleared = ground_removal_ext.ground_removal_kernel(b, distance_th, iter)
    # distance_th=0.025, iter=200
    # The last channel represents if this point is ground
    # 0: ground, 255: Non-ground
    non_ground_indices = np.where(pcdNoGroundArrCleared[:, 4] == 255)[0]
    # copy non ground points
    pcdNoGroundArrCleared = pcdNoGroundArrCleared[non_ground_indices, :]
    toc = time.time()
    print('Time used for ground removal: ', toc - tic)
    print(pcdNoGroundArr.shape[0])
    print(pcdNoGroundArrCleared.shape[0])
    print("Removed:", (pcdNoGroundArr.shape[0] - non_ground_indices.shape[0]), " points using C++ RANSAC")
    if (pcdNoGroundArr.shape[0] - non_ground_indices.shape[0] < 10000):
        removeFlatSurface = False
    else:
        pcdNoGround = pcdNoGround.select_by_index(non_ground_indices, invert=True)
    pcdNoGround = pcdNoGround.select_by_index(non_ground_indices, invert=True)
    showpcd(pcdNoGround, "pcd with ground removed")
    return pcdNoGround'''

# below solution could work
def removeGroundCutoff(pcdIn, groundLevel):
    pointsArr = np.asarray(pcdIn.points)
    groundIndicies = np.where(pointsArr[:, 2] < groundLevel)
    print("Removing ", len(groundIndicies[0]), " ground points")
    pcdOut = pcdIn.select_by_index(groundIndicies[0], invert=True)
    return pcdOut


def extractROI(pcd, roi=[], outsideROI=False):  # !TODO this is fucking dumb

    points = np.asarray(pcd.points)
    # Gets point indicies:
    roiIndicies = np.where(
        (points[:, 0] < roi[0]) & (points[:, 0] > roi[1]) & (points[:, 1] < roi[2]) & (points[:, 1] > roi[3]) & (
                points[:, 2] < roi[4]) & (points[:, 2] > roi[5]))
    pcd = pcd.select_by_index(roiIndicies[0], invert=outsideROI)

    return pcd


def getPointsInRadius(pcd, radius, centerPoint=[0, 0, 0], removeInRadius=True, isCylinder=False):
    pointsArr = np.asarray(pcd.points)
    if isCylinder:
        roiIndicies = np.where(np.sqrt(
            np.absolute((pointsArr[:, 0] - centerPoint[0]) ** 2 + (pointsArr[:, 1] - centerPoint[1]) ** 2)) < radius)
    else:
        roiIndicies = np.where(np.linalg.norm(pointsArr - centerPoint, axis=1) < radius)
    pcd = pcd.select_by_index(roiIndicies[0], invert=removeInRadius)
    return pcd, roiIndicies[0]


# Basically removing horizontal cylinder
def removeGroundLine(pcd, radius, leftPoint=[0, 0, 0], rightPoint=[0, 0, 0], removeInRadius=True):
    boundArr = [leftPoint[1], rightPoint[1]]
    centerPoint = (leftPoint + rightPoint) / 2
    pointsArr = np.asarray(pcd.points)
    roiHorizontalCylinder = np.where(np.sqrt(
        np.absolute((pointsArr[:, 0] - centerPoint[0]) ** 2 + (pointsArr[:, 2] - centerPoint[2]) ** 2)) < radius)
    roiDistanceBoundary = np.where(np.logical_and(pointsArr[:, 1] >= min(boundArr), pointsArr[:, 1] <= max(boundArr)))
    roiIndicies = np.intersect1d(roiHorizontalCylinder[0], roiDistanceBoundary[0])
    pcd = pcd.select_by_index(roiIndicies, invert=removeInRadius)
    return pcd, roiIndicies


'''def checkConePointNumberCondition(distance, numberOfPoints):
    isCone = True
    # Bottom exponential
    a1 = 4e+7  # max value
    b1 = 9.9  # left right shift
    c1 = 5  # height
    x = distance
    minimum = a1 * np.exp(-x - b1) + c1
    # Top exponential
    a2 = 3.9e+7  # max value
    b2 = 8  # left right shift
    c2 = 72  # height
    maximum = a2 * np.exp(-x - b2) + c2
    if (numberOfPoints < minimum or numberOfPoints > maximum):
        isCone = False
    return isCone'''

def checkConePointNumberCondition(distance, numberOfPoints):
    isCone = True
    # Bottom inverse square
    a1 = 4128
    b1 = 3.13
    c1 = -0.2
    d1 = -13
    x = distance
    minimum = inverse(distance, a1, b1, c1, d1)
    
    # Top inverse square
    a2 = 4425
    b2 = 1.24
    c2 = -0.2
    d2 = 15
    maximum = inverse(distance, a2, b2, c2, d2)
    if (numberOfPoints < minimum or numberOfPoints > maximum):
        isCone = False
    return isCone
    
def inverse(x, a, b, c, d):
    y = a/(b*(x*x) + c) + d
    return y
    


def extractClustersLabels(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label
                                             if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, labels


def extractConesByLabel(pcd, indicies):
    cluster = pcd.select_by_index(indicies)  # actual array is first element of tuple
    pointCloudCenter = cluster.get_center()
    # print("Point cloud center: ")
    # print(pointCloudCenter)
    clusterReferencePoint = pointCloudCenter

    '''
    pcd_tree = o3d.geometry.KDTreeFlann(cluster)
    origin = np.zeros(shape=(3,1))
    [k, idx, _] = pcd_tree.search_knn_vector_3d(origin, 1)
    print("idx return type: ")
    print(type(idx))
    closestPointPCD = pcd.select_by_index(idx)
    clusterClosestPointPCDArray = np.asarray(closestPointPCD.points)
    clusterClosestPoint = clusterClosestPointPCDArray[0]
    print("Cluster closest point: ")
    print(clusterClosestPoint)
    '''

    isClusterCone = checkIfClusterIsCone(pcd, cluster, clusterReferencePoint, indicies)
    return isClusterCone


def cone_statistical_filter(currentClusterPCD):  # TODO tune it so that it does not give false negatives
    isCone = True
    currentClusterPoints = np.asarray(currentClusterPCD.points)
    clusterCenter = currentClusterPCD.get_center()
    clusterCenterDistance = np.linalg.norm(clusterCenter)
    clusterNumberOfPoints = len(currentClusterPoints)

    maximum = currentClusterPCD.get_max_bound()
    minimum = currentClusterPCD.get_min_bound()
    clusterHeight = maximum[2] - minimum[2]
    clusterLength = maximum[1] - minimum[1]
    clusterWidth = maximum[0] - minimum[0]
    isCone = checkConePointNumberCondition(clusterCenterDistance, clusterNumberOfPoints)
    #if (maximum[2] > MAX_CONE_HEIGHT - SENSOR_TO_GROUND_DISTANCE):  # check if top of cone is higher than expected
        #isCone = False
    if (clusterHeight < 0.2 or clusterHeight > 0.5):  # check if cone cluster size is in expected range
        isCone = False
    if clusterWidth > 0.5:  # check if cone cluster width is in expected ranve
        isCone = False
    if clusterLength > 0.5:  # check if cone cluster length is in expected range
        isCone = False

    return isCone

def cone_statistical_filter_uncut(currentClusterPCD):  # TODO tune it so that it does not give false negatives
    isCone = True
    currentClusterPoints = np.asarray(currentClusterPCD.points)
    clusterCenter = currentClusterPCD.get_center()
    clusterCenterDistance = np.linalg.norm(clusterCenter)
    clusterNumberOfPoints = len(currentClusterPoints)

    maximum = currentClusterPCD.get_max_bound()
    minimum = currentClusterPCD.get_min_bound()
    clusterHeight = maximum[2] - minimum[2]
    clusterLength = maximum[1] - minimum[1]
    clusterWidth = maximum[0] - minimum[0]
    isCone = checkConePointNumberCondition(clusterCenterDistance, clusterNumberOfPoints)
    if (maximum[2] + SENSOR_TO_GROUND_DISTANCE > MAX_CONE_HEIGHT):  # check if top of cone is higher than expected
        isCone = False
    if (clusterHeight < 0.2 or clusterHeight > 0.55):  # check if cone cluster size is in expected range
        isCone = False
    '''if clusterWidth > 0.5:  # check if cone cluster width is in expected ranve
        isCone = False
    if clusterLength > 0.5:  # check if cone cluster length is in expected range
        isCone = False'''

    return isCone

def checkIfClusterIsCone(pcd, clusterPcd, referencePoint, indicies):
    isCone = True
    maximum = clusterPcd.get_max_bound()
    minimum = clusterPcd.get_min_bound()
    points = np.asarray(pcd.points)[indicies[0:], :]
    numberOfPoints = len(points)
    stdDev = np.std(points)
    clusterDistance = np.linalg.norm(referencePoint)
    print("Cluster distance: " + str(clusterDistance))
    print("Number of points: " + str(numberOfPoints))
    print("Standard deviation: " + str(stdDev))
    if (stdDev > IS_CONE_STD_MAX) or (stdDev < IS_CONE_STD_MIN):
        isCone = False
        print("Cluster not in standard deviation range")

    clusterSize = maximum[2] - minimum[2]
    if (clusterSize < IS_CONE_SIZE_MIN) or (clusterSize > IS_CONE_SIZE_MAX):
        isCone = False
        print("Cluster not in cone size range")
    print("Cluster size: " + str(clusterSize))
    print("Reference point: " + str(referencePoint))
    averageConeRadius = 0.07
    outliersNumber = 0
    maxOutlierNumber = IS_CONE_MAX_OUTLIER_NUMBER

    outlierArray = []
    # dist = np.linalg.norm(a - b)
    currentIndex = len(indicies) - 1
    if isCone:
        while currentIndex != -1:
            currentPoint = points[currentIndex]
            currentPoint[2] = referencePoint[2]  # TODO: refactor this shit XD
            dist = np.linalg.norm(referencePoint - currentPoint)  # Like seriously, what the fuck?
            # print(dist)
            if dist > averageConeRadius:  # remove points too far from center
                outliersNumber += 1
                outlierArray.append(currentIndex)
            if outliersNumber > maxOutlierNumber:
                isCone = False
                outlierArray = []
                break
            currentIndex += -1
    print(str(outliersNumber) + " points outside the bounds")
    """
    Essentially, check if cone has most points(%) in some range from center(Radius, point distance),
    and if it has a typical cone height, and number of points in cluster, adequate to the range.
    While doing so, extract points that are outliers, only up to the moment when there's more points
    than there should be. In this case, break the loop.
    Start from the end of indexes list if possible, as in Ouster lidar, the furthest points are the
    first ones on the list.

    Actually, start from the furthest points, and break the loop if there's already too
    much points for the object to be the cone. This way a lot of computational power will be saved.

    Check also the standard deviation of the cloud. If it's too high, then it's probably not a cones.

    Make two ranges, for big and small cones.

    Extract the position of cone clusters, then extract that same region from original point cloud,
    to get the intensity part.
    """
    if isCone:
        plt.plot(clusterDistance, numberOfPoints, 'ro')
    return isCone


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


# def create_cone_text_marker(message, color, position, idNumber, z_offset=0.5):
#     z_offset = np.float64(z_offset)
#     position[2] = position[2] + z_offset
#     marker = Marker()
#     marker.header.frame_id = "map"
#     marker.header.stamp = rospy.Time.now()
#     marker.type = 9  # text facing
#     # Set the scale of the marker
#     marker.scale.x = 0.2
#     marker.scale.y = 0.2
#     marker.scale.z = 0.2

#     marker.id = idNumber

#     # Set the color
#     marker.color.r = color[0]
#     marker.color.g = color[1]
#     marker.color.b = color[2]
#     marker.color.a = 1.0

#     # Set the pose of the marker
#     marker.pose.position.x = position[0]
#     marker.pose.position.y = position[1]
#     marker.pose.position.z = position[2]
#     marker.ns = "coneClusterInfo"
#     # Set text
#     marker.text = message
#     marker.lifetime = rospy.Duration.from_sec(0.15)
#     return marker


# def reset_markers(topic, max_marker_num = 150):
#     marker_array = MarkerArray()
#     for i in range(max_marker_num):
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = rospy.Time.now()
#         marker.type = 9
#         marker.scale.x = 0.2
#         marker.scale.y = 0.2
#         marker.scale.z = 0.2
#         marker.color.r = 0
#         marker.color.g = 0
#         marker.color.b = 0
#         marker.color.a = 0.01
#         marker.id = i
#         marker.lifetime = rospy.Duration.from_sec(0.0001)
#         # Set the pose of the marker
#         marker.pose.position.x = 0
#         marker.pose.position.y = 0
#         marker.pose.position.z = 0
#         marker.text = ''
#         marker.ns = "coneClusterInfo"
#         marker_array.markers.append(marker)
#     topic.publish(marker_array)

