import open3d as o3d
import numpy as np
import os, random, shutil, json
from sklearn.preprocessing import StandardScaler
os.chdir(os.path.dirname(__file__))


def showpcd(pcdin, pcdName):
    o3d.visualization.draw_geometries([pcdin], window_name=pcdName)
# Ładujemy przykładowy plik

skaler = StandardScaler()


save_file = "/home/rpomorski/Pulpit/Racing/driverless-software/PointCloud_Segmentation/pointnet_model_pytorch/racing_dataset2"
cones_id = "694202137"
other_id = "999999999"

os.makedirs(save_file, exist_ok=True)

#TODO usuwanie rzeczy z folderów
if cones_id in os.listdir(save_file):
    shutil.rmtree(os.path.join(save_file, cones_id))
if other_id in os.listdir(save_file):
    shutil.rmtree(os.path.join(save_file, other_id))

os.makedirs(os.path.join(save_file, cones_id, "points"), exist_ok=True)
os.makedirs(os.path.join(save_file, cones_id, "points_labels"), exist_ok=True)
os.makedirs(os.path.join(save_file, other_id, "points"), exist_ok=True)
os.makedirs(os.path.join(save_file, other_id, "points_labels"), exist_ok=True)

# Wyciągnięcie pachołków (samych) + utworzenie do nich plików z segmentacją
cones_path = os.path.join("./mldata", "pointnet", "cones")
for cone_ptd in os.listdir(cones_path):
    file = os.path.join(cones_path, cone_ptd)
    hash = random.getrandbits(50)
    point_cloud = o3d.io.read_point_cloud(file)
    point_cloud_np = np.asarray(point_cloud.points)
    if point_cloud_np.shape[0] != 1:
        np.savetxt(f"{save_file}/{cones_id}/points/{hash}.pts", point_cloud_np)
        segmentation_file = np.ones(point_cloud_np.shape[0], dtype=int)
        np.savetxt(f"{save_file}/{cones_id}/points_labels/{hash}.seg", segmentation_file)

others_path = os.path.join("./mldata", "pointnet", "other")
for other_ptd in os.listdir(others_path):
    file = os.path.join(others_path, other_ptd)
    hash = random.getrandbits(50)
    point_cloud = o3d.io.read_point_cloud(file)
    point_cloud_np = np.asarray(point_cloud.points)
    if point_cloud_np.shape[0] != 1:
        np.savetxt(f"{save_file}/{other_id}/points/{hash}.pts", point_cloud_np)
        segmentation_file = np.ones(point_cloud_np.shape[0], dtype=int)
        np.savetxt(f"{save_file}/{other_id}/points_labels/{hash}.seg", segmentation_file)

# # Przeniesienie plików z wcześniejszego folderu 
# os.makedirs(os.path.join(save_file, "999999999", "points"), exist_ok=True)
# os.makedirs(os.path.join(save_file, "999999999", "points_label"), exist_ok=True)
# old_data_path = "/home/rpomorski/Pulpit/Racing/driverless-software/PointCloud_Segmentation/pointnet_model_pytorch/shapenetcore_partanno_segmentation_benchmark_v0"
# for folder in os.listdir(old_data_path):
#     if "0" in folder:
#         points_path = os.path.join(old_data_path, folder, "points")
#         for file in os.listdir(points_path):
#             shutil.copyfile(os.path.join(points_path, file), os.path.join(save_file,"999999999", "points", file))
#         points_labels_path = os.path.join(old_data_path, folder, "points_label")
#         for file in os.listdir(points_labels_path):
#             shutil.copyfile(os.path.join(points_labels_path, file), os.path.join(save_file,"999999999", "points_label", file))

# Wygenerowanie plików jsona z podziałem na zbiory testowe i treningowe
list_of_cones = [f"shape_data/{cones_id}/" + i[:-4] for i in os.listdir(os.path.join(save_file, cones_id, "points"))]
list_of_others = [f"shape_data/{other_id}/" + i[:-4] for i in os.listdir(os.path.join(save_file, other_id, "points"))]
random.shuffle(list_of_cones)
random.shuffle(list_of_others)

train_list = list_of_cones[:int(len(list_of_cones)*0.6)]
val_list = list_of_cones[int(len(list_of_cones)*0.6):int(len(list_of_cones)*0.8)]
test_list = list_of_cones[int(len(list_of_cones)*0.8):]

train_list_others = list_of_others[:int(len(list_of_others)*0.6)]
val_list_others = list_of_others[int(len(list_of_others)*0.6):int(len(list_of_others)*0.8)]
test_list_others = list_of_others[int(len(list_of_others)*0.8):]

train_list = train_list + train_list_others
random.shuffle(train_list)
val_list = val_list + val_list_others
random.shuffle(val_list)
test_list = test_list + test_list_others
random.shuffle(test_list)

os.makedirs(os.path.join(save_file, "train_test_split"), exist_ok=True)
with open(os.path.join(save_file, "train_test_split","shuffled_test_file_list.json"), "w") as f:
    json.dump(test_list, f)

os.makedirs(os.path.join(save_file, "train_test_split"), exist_ok=True)
with open(os.path.join(save_file, "train_test_split","shuffled_train_file_list.json"), "w") as f:
    json.dump(train_list, f)

os.makedirs(os.path.join(save_file, "train_test_split"), exist_ok=True)
with open(os.path.join(save_file, "train_test_split","shuffled_val_file_list.json"), "w") as f:
    json.dump(val_list, f)