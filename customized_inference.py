import torch
import numpy as np
from pointnet.model import PointNetCls

#%% I/O functions
def read_off(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if 'OFF' != lines[0].strip():
        raise('Not a valid OFF header')
    
    n_vertices, _, _ = tuple(map(int, lines[1].strip().split()))
    vertices = np.array([[float(s) for s in line.strip().split()] for line in lines[2:2+n_vertices]])
    
    return vertices

# def write_features_to_txt(feature_list, filename):
#     with open(filename, 'w') as f:
#         for mesh_path, feature in feature_list:
#             feature_str = " ".join(map(str, feature))
#             f.write(f"{mesh_path} {feature_str}\n")

def write_features_to_txt(filename, feature_list, mesh_paths):
    with open(filename, 'w') as f:
        for i in range(len(feature_list)):
            feature_str = " ".join(map(str, feature_list[i]))
            f.write(f"{mesh_paths[i]} {feature_str}\n")

def read_features_from_txt(filename):
    mesh_features = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            mesh_path = parts[0]
            feature = np.array([float(x) for x in parts[1:]])
            mesh_features.append((mesh_path, feature))
    return [feature for _, feature in mesh_features]

#%% Data processing functions
def preProcessPC(vertices):
    """
    Input: 
        vertices: vertices read from file
    Output:
        point_set: proecess vertices
    """
    npoints=2500
    data_augmentation=True

    x = []
    y = []
    z = []
    for vertex in vertices:
        x.append(vertex[0])
        y.append(vertex[1])
        z.append(vertex[2])
    pts = np.vstack([x, y, z]).T

    choice = np.random.choice(len(pts), npoints, replace=True)
    point_set = pts[choice, :]

    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale

    if data_augmentation:
        theta = np.random.uniform(0, np.pi * 2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
        point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

    point_set = torch.from_numpy(point_set.astype(np.float32))
    # cls = torch.from_numpy(np.array([cls]).astype(np.int64))
    # return point_set, cls

    return point_set

#%% Inference
def inferenceBatch(model, pcArray):
    pcTensor = torch.tensor(pcArray, dtype=torch.float32).permute(0, 2, 1)
    print(f"pcTensor shape: {pcTensor.shape}")

    # Perform inference
    with torch.no_grad():
        output, trans, trans_feat = model(pcTensor, True)

    return output.numpy()

def main():
    #%% Load model
    modelPath = 'utils/cls/cls_model_75.pth'

    # Initialize the model and load the trained weights
    model = PointNetCls(k=5)  # Initialize with number of classes
    model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    model.eval()

    #%% Read from dataset and process pc
    data_path = '/Users/liujunyu/Data/Research/BVC/ITSS/ModelNet40/airplane/train/'
    mesh_names = range(1,601)
    mesh_paths = []
    pcList = []
    for mesh_name in mesh_names:
        meshPath = data_path + 'airplane_' + f"{mesh_name:04d}" + '.off'
        mesh_paths.append(meshPath)

        pc = read_off(meshPath)
        pcProcessed = preProcessPC(pc)
        pcList.append(pcProcessed)
    pcArray = np.array(pcList)
    print(f"pcArrray shape: {pcArray.shape}")

    #%% Read PC, process, and inferenece
    # feature_list = []
    # for mesh_path in mesh_paths:
    #     pc = read_off(mesh_path)
    #     pcProcessed = preProcessPC(model, pc)
    #     mesh_feature = inferenceFeature(pcProcessed)[0]
    #     # print(mesh_feature)
    #     feature_list.append((mesh_path, mesh_feature))
    featureArray = inferenceBatch(model, pcArray)
    featureList = featureArray.tolist()

    #%% Write features to file
    featureDataPath = "/Users/liujunyu/Data/Research/BVC/ITSS/" + "pointnet1_airplane_600.txt"
    # Writing features to txt
    write_features_to_txt(featureDataPath, featureList, mesh_paths)

    #%% Read features from file for inspection
    # read_features = read_features_from_txt(featureDataPath)
    # print("Read features:", read_features[0])

if __name__ == '__main__':
    main()

