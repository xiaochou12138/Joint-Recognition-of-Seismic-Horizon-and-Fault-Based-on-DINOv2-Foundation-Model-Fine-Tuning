from tqdm import tqdm
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')

# 弹性变换
def elastic_transform(data, alpha, sigma):
    '''

    :param data:
    :param alpha:
    :param sigma:
    :return:
    '''
    #
    elastic = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
    #
    augmented_data = elastic.augment_image(data)
    return augmented_data

#
seis_path = "synthetic_data/valid/seis/0.dat"
label_path = "synthetic_data/valid/horizon/0.dat"
seis = np.fromfile(seis_path, np.float32).reshape(224, 224)
label = np.fromfile(label_path, np.float32).reshape(224, 224)

index = 0
for i in tqdm(np.arange(0,10,0.1)):
    for j in range(0,3,1):
        seis_transform = elastic_transform(seis, i, j)
        label_transform = elastic_transform(label, i, j)
        mask1 = (label_transform > 0.5)
        postproduce_grad1 = np.where(mask1, 1, 0)
        label_transform = postproduce_grad1

        # #
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(seis, cmap='seismic')
        # plt.title('Original Data')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(seis_transform, cmap='seismic')
        # plt.title('Augmented Data')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        #
        # #
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(label, cmap='seismic')
        # plt.title('Original Data')
        # plt.axis('off')
        # plt.subplot(1, 2, 2)
        # plt.imshow(label_transform, cmap='seismic')
        # plt.title('Augmented Data')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        seis_transform.tofile("synthetic_data/train/seis_horizon/" + str(index) + ".dat")
        label_transform.tofile("synthetic_data/train/horizon/" + str(index) + ".dat")
        index = index + 1