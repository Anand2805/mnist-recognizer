from PIL import Image, ImageFilter
import numpy as np
import csv
import png
import randomForest as rfc

def readAndSendRecognizedData(imgBase64: Image, nRFC):
    wbground = Image.new("RGB", imgBase64.size, (255, 255, 255))
    wbground.paste(imgBase64, mask=imgBase64.split()[3])
    greys_image = wbground.convert(mode='L')
    
    # save and show the image
    #gs_image.save('d.jpg')
    #image2 = Image.open('d.jpg')
    #image2.show()

    print(greys_image)
    dQueryData = greyScaleToMnist(greys_image)
    predicted = rfc.trainAndPredictData(nRFC, dQueryData)
    return predicted

def greyScaleToMnist(img_grey):
    value = np.asarray(img_grey)
    value = value.flatten()
    value = ~value  # invert B&W
    #value[value > 0] = 1
    print(value)
    print(np.reshape(value, (28, 28)))
    return value
    #png.from_array(np.reshape(value, (28, 28)) , 'L').save("small_smiley.png")
    #with open("img_pixels.csv", 'a') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(value)
