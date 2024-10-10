import numpy as np
import torch
import pickle
import cv2

train_data = []
test_data = []


def find_bounding_box(binary_mask_path):
    # Read the binary mask
    mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes
    
for index in range(1,15+1):
    class_id=index
    print(class_id)
    folder = "data/lm_train/train"+os.sep+"{:06d}".format(class_id) +os.sep+ "rgb"
    mask_folder = "data/lm_train/train"+os.sep+"{:06d}".format(class_id) +os.sep+ "mask"
    files = os.listdir(folder)
    for i in range(len(files)):
        file = files[i]
        image_full_path = folder+os.sep+file
        mask_file = file.split('.')[0] 
        mask_full_path = mask_folder+os.sep+mask_file+'_000000.png'
        img = Image.open(image_full_path)
        x, y, w, h = find_bounding_box(mask_full_path)[0]
        crop = np.array(img)[y-8:y+h+8, x-8:x+w+8]
        arr = np.array(Image.fromarray(crop).resize((64,64)))
        torch_arr = torch.from_numpy(arr).float()
        label = torch.tensor([class_id]).long()
        if i<1000:
            train_data.append([torch_arr, label])
        else:
            test_data.append([torch_arr, label])
            
with open('linemod_masked_train_dataset','wb') as f:
    pickle.dump(train_data, f)
f.close()
with open('linemod_masked_test_dataset','wb') as f:
    pickle.dump(test_data, f)
f.close()