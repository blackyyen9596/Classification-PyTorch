from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import os
import glob

dataset_dir = r'D:\Github\BlackyYen\BlackyYen-public\machine_learning\Classification\ntut-ml-2020-classification\simpsons\train'
total_number = 3500

fileLists = os.listdir(dataset_dir)

for fileList in fileLists:
    i = 0
    j = 0
    dataset_class_dir = os.path.join(dataset_dir, fileList)
    dirPathPattern = os.path.join(dataset_class_dir, '*.jpg')
    samples = glob.glob(dirPathPattern)

    while True:
        try:
            fname = os.path.basename(samples[i])
            img_path = os.path.join(dataset_class_dir, samples[i])
            img = load_img(img_path)
        except:
            i = 0
            j += 1
            fname = os.path.basename(samples[i])
            img_path = os.path.join(dataset_class_dir, samples[i])
            img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1, ) + x.shape)
        # print(x.shape)

        datagen = ImageDataGenerator(rotation_range=15,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     shear_range=0.1,
                                     zoom_range=[1, 1.2],
                                     fill_mode="wrap",
                                     horizontal_flip=True,
                                     vertical_flip=False)

        k = 0
        for suffix in range(0, 1):
            for batch_img in datagen.flow(
                    x,
                    batch_size=1,
                    save_to_dir=dataset_class_dir,
                    save_prefix=os.path.splitext(samples[i])[0] + '_%d' % j,
                    save_format="jpg"):
                # plt.imshow(batch_img[0].astype("int"))
                # plt.axis("off")
                # plt.show()
                k += 1
                if k >= 0:
                    break
        i += 1
        dirPathPattern = os.path.join(dataset_class_dir, '*.jpg')
        sample = glob.glob(dirPathPattern)
        sample_number = len(sample)
        print(sample_number)
        if sample_number == total_number:
            break
