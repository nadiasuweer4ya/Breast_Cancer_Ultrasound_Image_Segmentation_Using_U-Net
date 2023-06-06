import gradio as gr
from PIL import Image
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

size = 128

def preprocess_image(image, size=128):
    image = image.resize((size, size))
    image = image.convert("L")
    image = np.array(image) / 255.0
    return image

def conv_block(input, num_filters):
    conv = Conv2D(num_filters, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(input)
    conv = Conv2D(num_filters, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(conv)
    return conv

def encoder_block(input, num_filters):
    conv = conv_block(input, num_filters)
    pool = MaxPooling2D((2, 2))(conv)
    return conv, pool

def decoder_block(input, skip_features, num_filters):
    uconv = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    con = concatenate([uconv, skip_features])
    conv = conv_block(con, num_filters)
    return conv

def build_model(input_shape):
    input_layer = Input(input_shape)
    
    s1, p1 = encoder_block(input_layer, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    output_layer = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)                                                               
    model = Model(input_layer, output_layer, name="U-Net")
    model.load_weights('BreastCancerSegmentation.h5')
    return model
    
def preprocess_image(image, size=128):
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255.
    return image

def segment(image):
    image = preprocess_image(image, size=size)
    image = np.expand_dims(image, 0)
    output = model.predict(image, verbose=0)
    mask_image = output[0]
    mask_image = np.squeeze(mask_image, -1)
    mask_image *= 255
    mask_image = mask_image.astype(np.uint8)
    mask_image = Image.fromarray(mask_image).convert("L")
    return mask_image

if __name__ == "__main__":
    model = build_model(input_shape=(size, size, 1))
    gr.Interface(
        fn=segment,
        inputs="image",
        outputs=gr.Image(type="pil", label="Breast Cancer Mask"),
        examples=[["benign(10).png"], ["benign(109).png"]],
        title = '<h1 style="text-align: center;">Breast Cancer Ultrasound Image Segmentation! 💐 </h1>',
        description = """
        Check out this exciting development in the field of breast cancer diagnosis and treatment!
        A demo of Breast Cancer Ultrasound Image Segmentation has been developed.
        Upload image file, or try out one of the examples below! 🙌
        """
    ).launch(debug=True)
