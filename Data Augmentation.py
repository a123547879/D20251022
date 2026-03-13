import os
import joblib
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# from keras.utils import image_dataset_from_directory
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from sklearn.cluster import MeanShift, estimate_bandwidth

model_vgg = VGG16(weights= 'imagenet', include_top= False)

def add_salt_pepper_noise(image):
    # 添加椒盐噪声
    s_vs_p = 0.5
    amount = 0.04
    noisy = np.copy(image)
    
    # 盐噪声（白点）
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 255
    
    # 椒噪声（黑点）
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy


def DataEnhance(dataPath, savePath, isAdd= False):
    if not isAdd:
        datagen = ImageDataGenerator(rotation_range= 20, 
                                     width_shift_range= 0, 
                                     height_shift_range= 0, 
                                     horizontal_flip= False, 
                                     vertical_flip= False)
    else:
        datagen = ImageDataGenerator(rotation_range= 20,  
                                     preprocessing_function= add_salt_pepper_noise, 
                                     width_shift_range= 0,
                                     height_shift_range= 0, 
                                     horizontal_flip= False, 
                                     vertical_flip= False)
    gen = datagen.flow_from_directory(dataPath, target_size= (224, 224), batch_size= 2, save_to_dir= savePath, save_prefix= 'gen', save_format= 'png')
    for i in range(2000):
        batch = next(gen)


def MSModelCreate(dataPath, savePath):
    data = joblib.load(dataPath)
    bw = estimate_bandwidth(data, n_samples= int(data.shape[0] * 0.6))
    model = MeanShift(bandwidth= bw)
    model.fit(data)
    joblib.dump(model, savePath)

def VisualData(imgFolder, tip, row, cols):
    img_paths = []
    for img_path in os.listdir(imgFolder):
        img_paths.append(f'{imgFolder}/{img_path}')

    normal_apple_id = 0

    for i in range(row):
        for j in range(cols):
            img = load_img(img_paths[i * cols + j])
            plt.subplot(row, cols, i * cols + j + 1)
            plt.title('apple' if tip[i * cols + j] == normal_apple_id else 'other')
            plt.imshow(img)
            plt.axis('off')
    plt.show()

for i in range(1, 10):
    originalPath = f'D:/DY/num/{i}'
    genPath = f'dataFolder/num/{i}'

    DataEnhance(originalPath, genPath)

# trainPath = 'D20240529/train_data'
# testPath = 'D20240529/test_data'
# saveDataPath = 'D20240531/data.pkl'
# saveTestPath = 'D20240531/test_data.pkl'

# loadData(trainPath, model_vgg, saveDataPath)
# loadData(testPath, model_vgg, saveTestPath)

# saveModelPath = 'D20240531/model.pkl'
# MSModelCreate(saveDataPath, saveModelPath)

# x_train = joblib.load(saveDataPath)
# x_test = joblib.load(saveTestPath)

# model = joblib.load(saveModelPath)

# y_predict_train = model.predict(x_train)
# print(Counter(y_predict_train))

# y_predict_test = model.predict(x_test)

# VisualData(testPath, y_predict_test, 3, 4)



    