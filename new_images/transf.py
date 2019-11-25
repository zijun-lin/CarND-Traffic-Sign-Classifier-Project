import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2


for i, fimg in enumerate(glob.glob('*.ppm')):
    image = mpimg.imread(fimg)
    label = (fimg.strip('new_images/')).strip('.ppm')
    print('Image Name: ', fimg, label)

    plt.imshow(image)
    plt.show()

    # cv2.imwrite(label+'.jpg', image)
    plt.imsave(label+'.jpg', image)