import os
from typing import Tuple
import cv2
import shutil
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as IMG
from PIL import Image
import numpy as np
from skimage import transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

DATA_DIR = os.path.join("/content/drive/MyDrive/Dataset")
DATA_DIR_NOT_DIVYANK = os.path.join(r"D:\Studium\MAsterarbeit\Udemzy Kurs ML\Python Programms\MasterArbeit\Dataset\LFW_subDATASET")
DATA_DIR_LFW = os.path.join(r"D:\Studium\MAsterarbeit\Udemzy Kurs ML\Python Programms\lfw_funneled")
FACES_DIVYANK = os.path.join(r"D:\Studium\MAsterarbeit\Udemzy Kurs ML\Python Programms\MasterArbeit\Dataset\DivyankFaces")
FACES_LFW = os.path.join(r"D:\Studium\MAsterarbeit\Udemzy Kurs ML\Python Programms\MasterArbeit\Dataset\LFW_subDATASETFaces")
DEST_LFW = os.path.join(r"D:\Studium\MAsterarbeit\Udemzy Kurs ML\Python Programms\MasterArbeit\Dataset\not_Divyank")
RESHAPED_DIVYANK = os.path.join(r"D:\Studium\MAsterarbeit\Udemzy Kurs ML\Python Programms\MasterArbeit\Dataset\Divyank_resized")
X_FILE_PATH = os.path.join (DATA_DIR, "x.npy")
Y_FILE_PATH = os.path.join(DATA_DIR,"y.npy")

IMG_SHAPE_ROW = 96
IMG_SHAPE_COL = 96
IMG_DEPTH_RGB = 3   # RGB Photos
IMG_DEPTH_GRAYSCALE = 1 # GRAYSCALE Photos
IMG_SIZE_RGB = [IMG_SHAPE_ROW, IMG_SHAPE_COL, IMG_DEPTH_RGB]
IMG_SIZE_GRAYSCALE  = [IMG_SHAPE_ROW, IMG_SHAPE_COL, IMG_DEPTH_GRAYSCALE]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

divyank_DIR = os.path.join(DATA_DIR, "Divyank_resized" ) 
not_divyank_DIR = os.path.join( DATA_DIR, "LFW_subDATASET_Resized")
dirs = [divyank_DIR, not_divyank_DIR]
classes = ["Divyank", "not_Divyank"]

#  ITU-R 601-2 luma transform:
#  L = R * 299/1000 + G * 587/1000 + B * 114/1000

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2990, 0.5870, 0.1140])



def extract_divyank_not_divyank() -> None:  
    num_divyank = len(os.listdir(divyank_DIR))
    num_not_divyank = len(os.listdir(not_divyank_DIR))
    total_num_photos = num_divyank + num_not_divyank
    

    x = np.zeros(
        shape = (total_num_photos, IMG_SHAPE_ROW, IMG_SHAPE_COL, IMG_DEPTH_GRAYSCALE),
        dtype = np.float32
    )
    y = np.zeros(
        shape = (total_num_photos,),
        dtype = np.float32
    )
    number_of_images_read = 0
    number_of_images_not_read = 0
    cnt = 0
    for d ,class_names in zip(dirs, classes):
        for f in os.listdir(d):         
            image_file_path = os.path.join(d , f)
            try:    
                number_of_images_read += 1
                img = IMG.imread(image_file_path)
                img = rgb2gray(img)
                x[cnt] =  transform.resize(
                    image = img,
                    output_shape = IMG_SIZE_GRAYSCALE
                    )
                if class_names == "Divyank":
                    y[cnt] = 0
                elif class_names == "not_Divyank":

                    y[cnt] = 1
                else:
                    print("Invalid Class name!!")
                cnt +=1
                
                                
            except: #noqa: E722
                number_of_images_not_read += 1
                print(f"Image{f} cant be read")
                #os.remove(image_file_path)

    # Dropping not readable
    x = x[:cnt]
    y = y[:cnt]
    print(f"Number of Images saved:{number_of_images_read}")
    print(f"Number of Images not saved:{number_of_images_not_read}")
    np.save(X_FILE_PATH,x)
    np.save(Y_FILE_PATH,y)


def reshaping_the_photos()-> None:
# Since the shape of the divyank photos will be matched to not_divyank photos
# hence the transform function will be applied only to divyank_Dir
    cnt = 0
    for f in os.listdir(FACES_DIVYANK):
        image_file_path = os.path.join(FACES_DIVYANK , f)
        resizing_img = Image.open(image_file_path)
        img_resized = resizing_img.resize((96 , 96))
        
        img_resized.save(f)
        

def crop_and_save_photos() -> None:
    for i in os.listdir(DATA_DIR_NOT_DIVYANK):
        image_file_path = os.path.join(DATA_DIR_NOT_DIVYANK, i)
        #image_file_path = os.path.join(i)
        try:
            img = cv2.imread(image_file_path,1)
            gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img,1.1,5, minSize = (150 , 150))
            for(x,y,w,h) in faces:
                img = cv2.rectangle(img,(x,y), (x+w,y+h), (255,0,0),2)
                croped_img = img[ y:y+h, x: x+w]   #Targeting the required area
                cv2.imwrite(f"not_Divyank_image{i}", croped_img)
        except:
            print(f"Image {i} cant be read!")
            #os.remove(image_file_path)

def saving_lfw_photos_in_one_folder() -> None:
    for root, dirs, file in os.walk(DATA_DIR_LFW, topdown= False):
        for f in file:
            file_path = os.path.join(root ,f)
            DEST_PATH = os.path.join( DEST_LFW , f)
            shutil.copyfile(file_path, DEST_PATH)

class DivyankNotDivyank:
    def __init__(self, test_size: float = 0.2, validation_size: float = 0.33) -> None:
        # User-definen constants
        self.num_classes = 2
        self.batch_size = 128
        # Load the data set
        x = np.load(X_FILE_PATH)
        y = np.load(Y_FILE_PATH)
        # Split the dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size)
        # Preprocess x data
        self.x_train = x_train.astype(np.float32)
        self.x_test = x_test.astype(np.float32)
        self.x_val = x_val.astype(np.float32)
        # Preprocess y data
        self.y_train = to_categorical(y_train, num_classes=self.num_classes)
        self.y_test = to_categorical(y_test, num_classes=self.num_classes)
        self.y_val = to_categorical(y_val, num_classes=self.num_classes)
        # Dataset attributes
        self.train_size = self.x_train.shape[0]
        self.test_size = self.x_test.shape[0]
        self.val_size = self.x_val.shape[0]
        self.width = self.x_train.shape[1]
        self.height = self.x_train.shape[2]
        self.depth = self.x_train.shape[3]
        self.img_shape = (self.width, self.height, self.depth)

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_train, self.y_train

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_test, self.y_test

    def get_val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.x_val, self.y_val
      
    def load_and_preprocess_custom_image(image_file_path:str )-> np.ndarray:
      
      img = IMG.imread(image_file_path)
      img = rgb2gray(img)
      img =  transform.resize(
          image = img,
          output_shape = IMG_SIZE_GRAYSCALE
                    )
      return img

    def data_augmentation(self, augment_size: int = 5_000) -> None:
        image_generator = ImageDataGenerator(
            rotation_range=5,
            zoom_range=0.08,
            width_shift_range=0.08,
            height_shift_range=0.08
        )
        # Fit the data generator
        image_generator.fit(self.x_train, augment=True)
        # Get random train images for the data augmentation
        rand_idxs = np.random.randint(self.train_size, size=augment_size)
        x_augmented = self.x_train[rand_idxs].copy()
        y_augmented = self.y_train[rand_idxs].copy()
        x_augmented = image_generator.flow(
            x_augmented,
            np.zeros(augment_size),
            batch_size=augment_size,
            shuffle=False
        ).next()[0]
        # Append the augmented images to the train set
        self.x_train = np.concatenate((self.x_train, x_augmented))
        self.y_train = np.concatenate((self.y_train, y_augmented))
        self.train_size = self.x_train.shape[0]

if __name__ == "__main__":
    extract_divyank_not_divyank()

    # data = DivyankNotDivyank()
    # print(data.y_train.shape)
    # print(data.y_val.shape)
