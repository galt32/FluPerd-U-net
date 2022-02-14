# for data load
import os

# for reading and processing images
#import imageio
from PIL import Image
from skimage.io import imread
from skimage.transform import rescale, resize
# for visualizations
import matplotlib.pyplot as plt
import re
import numpy as np # for using np arrays
import glob
# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split


def LoadData (path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively
    """
    # Make a list for images and masks filenames
    bright_img = []
    fluor_img = []
    
    r = re.compile(r'r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)sk(\d+)')
    def custom_key(x):
        val = int(r.search(x).group(5)) * 10000 + int(r.search(x).group(1)) * 1000 + int(r.search(x).group(2)) * 100 + int(r.search(x).group(3)) * 10 + int(r.search(x).group(6))
        return val
    def custom_key2(x):
        val = int(r.search(x).group(1)) * 100000 + int(r.search(x).group(2)) * 10000 + int(r.search(x).group(3)) * 1000 + int(r.search(x).group(6))
        return val
    # Read the images folder like a list
    for filename in sorted(glob.glob(path1 + '*.tiff'), key = lambda x: custom_key(x)):
        #print (filename)
        if (int(r.search(filename).group(2)) == 6 or int(r.search(filename).group(2)) == 7):
            if int(r.search(filename).group(6)) > 180:
                continue
        if (int(r.search(filename).group(2)) == 8):
            if int(r.search(filename).group(6)) > 180:
                continue
        if (int(r.search(filename).group(5)) == 2):
            bright_img.append(filename)
        elif (int(r.search(filename).group(5)) == 1):
            fluor_img.append(filename)
    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    bright_img.sort(key = lambda x: custom_key2(x))
    fluor_img.sort(key = lambda x: custom_key2(x))
    
    return bright_img, fluor_img

def PreprocessData(bright, fluor, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    
    
     image_bright_field  = imread(filename)
            image_bright_field = np.float64(image_bright_field)
            image_bright_field = image_bright_field * 255.0 / image_bright_field.max()
            image_bright_field = np.uint8(image_bright_field)
            
            image_fluor  = imread(filename)
            image_fluor = np.float64(image_fluor)
            image_fluor = image_fluor * 255.0 / image_fluor.max()
            image_fluor = np.uint8(image_fluor)
    """
    # Pull the relevant dimensions for image and mask
    m = len(bright)                     # number of images
    i_h, i_w, i_c = target_shape_img   # pull height, width, and channels of image
    m_h, m_w, m_c = target_shape_mask  # pull height, width, and channels of mask
    print (i_h, i_w, i_c)
    print (m_h, m_w, m_c)
    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)
    
    # Resize images and masks
    for file in bright:
        # convert image into an array of desired shape (3 channels)
        index = bright.index(file)
        #path = os.path.join(path1, file)
        
        #image_bright_field = Image.open(file)#.convert('RGB') #imread(file) #Image.open(file).convert('RGB')
        image_bright_field = imread(file)
        image_bright_field = resize(image_bright_field, (i_h, i_w),
                       anti_aliasing=True)
        #print (image_bright_field.shape)
        #image_bright_field = image_bright_field.resize((i_h, i_w))
        image_bright_field = np.reshape(image_bright_field,(i_h, i_w, i_c)) 
        #image_bright_field = single_img/256.
        image_bright_field = np.float64(image_bright_field)
        image_bright_field = image_bright_field / image_bright_field.max()
        #image_bright_field = np.uint8(image_bright_field)
        X[index] = image_bright_field
        #print (image_bright_field.shape)
        
        
        # convert mask into an array of desired shape (1 channel)
        
        single_mask_ind = fluor[index]
        #image_fluor  = Image.open(file) # imread(single_mask_ind)
        image_fluor  = imread(single_mask_ind)
        image_fluor = resize(image_fluor, (i_h, i_w),
                       anti_aliasing=True)
        #image_fluor = image_fluor.resize((i_h, i_w))
        image_fluor = np.reshape(image_fluor,(i_h, i_w, i_c)) 
        image_fluor = np.float64(image_fluor)
        image_fluor = image_fluor * 15.0 / image_fluor.max()
        image_fluor = np.uint8(image_fluor)
        #image_bright_field = single_img/256.
        y[index] = image_fluor
        
    return X, y


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow 
    # Proper initialization prevents from the problem of exploding and vanishing gradients 
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size) 
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions      
    skip_connection = conv
    
    return next_layer, skip_connection



def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)
    
    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv



def UNetCompiled(input_size=(540, 540, 1), n_filters=32, n_classes=16):
   """
   Combine both encoder and decoder blocks according to the U-Net research paper
   Return the model as output 
   """
    # Input size represent the size of 1 image (the size used for pre-processing) 
   inputs = Input(input_size)
    
    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
   cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
   cblock2 = EncoderMiniBlock(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
   cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
   cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
   cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
   ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
   ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
   ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
   ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size. 
    # Observe the number of channels will be equal to number of output classes
   conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

   conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
    
    # Define the model
   model = tf.keras.Model(inputs=inputs, outputs=conv10)

   return model


# Call the apt function
path1 = '/media/nk/TNF_experiments_II/NG/JW162__2022-02-04T18_35_53-Measurement 1/Images/'
path2 = '/media/nk/TNF_experiments_II/NG/JW162__2022-02-04T18_35_53-Measurement 1/Images/'

bright, fluor = LoadData (path1, path2)


# View an example of image and corresponding mask 
show_images = 1
print(bright[101])
print(fluor[101])

    
    
# Define the desired shape
target_shape_img = [512,512, 1]
target_shape_mask = [512, 512, 1]

# Process data using apt helper function
X, y = PreprocessData(bright, fluor, target_shape_img, target_shape_mask, path1, path2)

# QC the shape of output and classes in output dataset 
print("X Shape:", X.shape)
print("Y shape:", y.shape)
# There are 3 classes : background, pet, outline



# Use scikit-learn's function to split the dataset
# Here, I have used 20% data as test/valid set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state = 123)


# Call the helper function for defining the layers for the model, given the input image size
unet = UNetCompiled(input_size=(512,512,1), n_filters=32, n_classes=16)

unet.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
results = unet.fit(X_train, y_train, batch_size=4, epochs=1, validation_data=(X_valid, y_valid))
unet.evaluate(X_valid, y_valid)

# Results of Validation Dataset
def VisualizeResults(index):
    img = X_valid[index]
    img = img[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    fig, arr = plt.subplots(1, 3, figsize=(35, 35))
    arr[0].imshow(X_valid[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_valid[index,:,:,0])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:,:,0])
    arr[2].set_title('Predicted Masked Image ')
    plt.savefig('plot{}.png'.format(index))
count = 100
# Add any index to contrast the predicted mask with actual mask
for i in range (len(X_valid)):
    if i >100:
        break
    VisualizeResults(i) 



