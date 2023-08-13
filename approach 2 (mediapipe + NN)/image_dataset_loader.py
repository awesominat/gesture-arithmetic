import mediapipe as mp
import numpy as np
import os
import random
import torch
import glob
from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision.transforms import ToTensor/
from PIL import Image


dict_to_len = {
    '0': 1,
    '1': 1,
    '2': 1,
    '3': 1,
    '4': 1,
    '5': 1,
    '6': 2,
    '7': 2,
    '8': 2,
    '9': 2,
    '10': 2,
    '11': 1,
    '12': 2,
    '13': 1,

    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 2,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 1,
    12: 2,
    13: 1,
}

def process_image(image_path, label, ISTRAIN=False):
    base_options = mp.tasks.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                                            min_hand_detection_confidence=0.5,
                                                            min_hand_presence_confidence=0.5,
                                                            min_tracking_confidence=0.5,
                                            num_hands=2, running_mode=mp.tasks.vision.RunningMode.IMAGE)
    detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    # ISTRAIN => (expected_len = 1 or expected_len = 2)
    assert not ISTRAIN or (dict_to_len[label] == 1 or dict_to_len[label] == 2), 'unable to find correct matching from label ' + str(label)

    expected_len = dict_to_len[label]
    test_img = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(test_img)

    # unfortunate brute force approach since mediapipe hand detection isnt the most accurate for 2 hands if confidence is at 0.5
    # lowering it fixes it and brings hand #2 confidence >= 95%
    if not ISTRAIN and len(detection_result.handedness) != expected_len:
        for confidence_level in np.arange(.45, -.05, -.05):
            confidence_level = max(0, confidence_level)
            tempoptions = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                                                min_hand_detection_confidence=confidence_level,
                                                                min_hand_presence_confidence=confidence_level,
                                                                min_tracking_confidence=confidence_level,
                                                num_hands=2, running_mode=mp.tasks.vision.RunningMode.IMAGE)
            tempdetector = mp.tasks.vision.HandLandmarker.create_from_options(tempoptions)
            detection_result = tempdetector.detect(test_img)
            if len(detection_result.handedness) == expected_len:
                break
    
    concat_displays = ''
    hand_world_landmarks_tuples = np.zeros((1, 21, 2)) # left/right hand, 21 land marks, (x,y,z) pos

    running_ind = 0
    for handed in detection_result.handedness:
        display_name = handed[0].display_name
        # if display_name == 'Left':
        ind = 0
        # elif display_name == 'Right':
        #     ind = 1
        if concat_displays == '':
            concat_displays += display_name
        else:
            concat_displays += ', ' + display_name
        hand_world_landmarks_tuples[ind, :] = [(obj.x, obj.y) for obj in detection_result.hand_landmarks[running_ind]]
        # hand_world_landmarks_tuples[ind, :] = [(obj.x, obj.y, obj.z) for obj in detection_result.hand_world_landmarks[running_ind]]
        running_ind += 1

    # since it is not guaranteed for a left/right hand to be in the image, the np array is already padded
    # which hopefully silences the layers using the left/right hand

    # flatten the array to pass into the nn
    # flattened = hand_world_landmarks_tuples.reshape((126, 1))
    # flattened = torch.flatten(torch.from_numpy(hand_world_landmarks_tuples))

    # if len(detection_result.handedness) != expected_len:
    #     print('discarding ' + image_path)
    #     return np.zeros((1, 84))
    return hand_world_landmarks_tuples

def augment_landmarks(landmarks, flip_horizontal_prob=0.5, flip_vertical_prob=0.5, jitter_std=0.002):
    if random.random() < flip_horizontal_prob:
        landmarks[:, :, 0] = -landmarks[:, :, 0]

    if random.random() < flip_vertical_prob:
        landmarks[:, :, 1] = -landmarks[:, :, 1]

    non_zero_mask = landmarks != 0
    noise = torch.randn_like(landmarks) * jitter_std
    landmarks += noise * non_zero_mask.float()

    return landmarks

class CustomImageDataset(Dataset):
    image_paths: str
    labels: list

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        landmarks = torch.from_numpy(process_image(img_path, label, True))

        # if self.transform:
        #     landmarks = augment_landmarks(landmarks)

        flattened = torch.flatten(landmarks).to(torch.float32).to('cuda')

        return flattened, label

def create_datasets(root_dir, val_samples_per_class=3, test_samples_per_class=3, transform=None):
    all_image_paths = glob.glob(os.path.join(root_dir, '*.jpg'))
    label_to_paths = {}
    
    for path in all_image_paths:
        label = int(os.path.basename(path).split()[0])
        if label not in label_to_paths:
            label_to_paths[label] = []
        # to_append = process_image(path, label, True)
        label_to_paths[label].append(path) # to_append

    train_paths, val_paths, test_paths = [], [], []
    for label, paths in label_to_paths.items():
        random.shuffle(paths)
        val_paths.extend(paths[:val_samples_per_class])
        test_paths.extend(paths[val_samples_per_class:val_samples_per_class + test_samples_per_class])
        train_paths.extend(paths[val_samples_per_class + test_samples_per_class:])

    train_labels = [int(os.path.basename(path).split()[0]) for path in train_paths]
    val_labels = [int(os.path.basename(path).split()[0]) for path in val_paths]
    test_labels = [int(os.path.basename(path).split()[0]) for path in test_paths]

    train_dataset = CustomImageDataset(train_paths, train_labels, transform)
    val_dataset = CustomImageDataset(val_paths, val_labels, transform)
    test_dataset = CustomImageDataset(test_paths, test_labels, transform)

    return train_dataset, val_dataset, test_dataset
