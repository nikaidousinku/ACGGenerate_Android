import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from imageio import imsave
from PIL import Image
import datetime

batch_size = 64
z_dim = 128
LABEL = 34
all_tags = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair', 'purple hair', 'green hair',
            'red hair', 'silver hair', 'white hair', 'orange hair', 'aqua hair', 'grey hair', 'long hair',
            'short hair', 'twintails', 'drill hair', 'ponytail', 'blue eyes', 'red eyes', 'brown eyes',
            'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes',
            'blush', 'smile', 'open mouth', 'hat', 'ribbon', 'glasses']


def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    # print(images.shape)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))  # 8
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    elif len(images.shape) == 4 and images.shape[3] == 1:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 1)) * 0.5
    elif len(images.shape) == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    else:
        raise ValueError('Could not parse image shape of {}'.format(images.shape))
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


def get_random_tags():
    y = np.random.uniform(0.0, 1.0, [batch_size, LABEL]).astype(np.float32)
    p_other = [0.6, 0.6, 0.25, 0.04488882, 0.3, 0.05384738]
    for i in range(batch_size):
        for j in range(len(p_other)):
            if y[i, j + 28] < p_other[j]:
                y[i, j + 28] = 1
            else:
                y[i, j + 28] = 0

    phc = [0.15968645, 0.21305391, 0.15491921, 0.10523116, 0.07953927, 0.09508879, 0.03567429, 0.07733163, 0.03157895,
           0.01833307, 0.02236442, 0.00537514, 0.00182371]
    phs = [0.52989922, 0.37101264, 0.12567589, 0.00291153, 0.00847864]
    pec = [0.28350664, 0.15760678, 0.17862742, 0.13412254, 0.14212126, 0.0543913, 0.01020637, 0.00617501, 0.03167493,
           0.00156775]
    for i in range(batch_size):
        y[i, :28] = 0

        hc = np.random.random()
        for j in range(len(phc)):
            if np.sum(phc[:j]) < hc < np.sum(phc[:j + 1]):
                y[i, j] = 1
                break

        hs = np.random.random()
        for j in range(len(phs)):
            if np.sum(phs[:j]) < hs < np.sum(phs[:j + 1]):
                y[i, j + 13] = 1
                break

        ec = np.random.random()
        for j in range(len(pec)):
            if np.sum(pec[:j]) < ec < np.sum(pec[:j + 1]):
                y[i, j + 18] = 1
                break
    return y


def orient_generate(tags, now_time="201904031"):
    # for i, tags in enumerate([['blonde hair', 'twintails', 'blush', 'smile', 'ribbon', 'red eyes'],
    # ['silver hair', 'long hair', 'blush', 'smile', 'open mouth', 'blue eyes']]):
    z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
    # print(z_samples)
    # print(len(z_samples),type(z_samples))
    # print(z_samples.shape)
    y_samples = np.zeros([1, LABEL])
    # print('分割线',y_samples)
    # print(len(y_samples))

    for tag in tags:
        y_samples[0, all_tags.index(tag)] = 1
    y_samples = np.repeat(y_samples, batch_size, 0)
    # print(y_samples)
    # print(type(y_samples[0][0]))
    #
    #
    # print('z_samples.shape', z_samples.shape)
    # print('y_samples.shape',y_samples.shape)
    gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
    # print(gen_imgs.shape)
    # print('分割线2')
    # print(is_training)

    gen_imgs = (gen_imgs + 1) / 2

    imgs = [img[:, :, :] for img in gen_imgs]

    # imgs= np.array(imgs)
    r_img = random.choice(imgs)
    # print(imgs.shape)
    # gen_imgs = montage(imgs)
    r_img = np.clip(r_img, 0, 1)

    imsave('../data/images/{}.jpg'.format(now_time), r_img)


def Generate(tags, now_time):
    pb_path = "./models/anime_acgan-60000.pb"

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            g = sess.graph.get_tensor_by_name('generator/g/Tanh:0')
            noise = sess.graph.get_tensor_by_name('noise:0')
            noise_y = sess.graph.get_tensor_by_name('noise_y:0')
            is_training = sess.graph.get_tensor_by_name('is_training:0')

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # saver = tf.train.import_meta_graph('./models/anime_acgan-60000.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('./models'))
    #
    # graph = tf.get_default_graph()
    # g = graph.get_tensor_by_name('generator/g/Tanh:0')
    # noise = graph.get_tensor_by_name('noise:0')
    # noise_y = graph.get_tensor_by_name('noise_y:0')
    # is_training = graph.get_tensor_by_name('is_training:0')

            z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
            y_samples = np.zeros([1, LABEL])
            for tag in tags:
                y_samples[0, all_tags.index(tag)] = 1
            y_samples = np.repeat(y_samples, batch_size, 0)
            gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
            gen_imgs = (gen_imgs + 1) / 2
            imgs = [img[:, :, :] for img in gen_imgs]
            r_img = random.choice(imgs)
            r_img = np.clip(r_img, 0, 1)
            imsave('./data/images/{}.jpg'.format(now_time), r_img)
            print('文件名', './data/images/{}.jpg'.format(now_time))
            # orient_generate(tags,now_time)


if __name__ == '__main__':
    pb_path = "../models/anime_acgan-60000.pb"
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            g = sess.graph.get_tensor_by_name('generator/g/Tanh:0')
            noise = sess.graph.get_tensor_by_name('noise:0')
            noise_y = sess.graph.get_tensor_by_name('noise_y:0')
            is_training = sess.graph.get_tensor_by_name('is_training:0')

            tags = []
            add_list = ['black hair', 'blue eyes', 'smile']
            tags = tags + add_list
            orient_generate(tags)
            # z_samples = np.random.uniform(-1.0, 1.0, [batch_size, z_dim]).astype(np.float32)
            # #y_samples = get_random_tags()
            # y_samples = np.zeros([1, 34])
            # for tag in tags:
            #   y_samples[0, all_tags.index(tag)] = 1
            #
            # y_samples=np.repeat(y_samples,64,0)
            # gen_imgs = sess.run(g, feed_dict={noise: z_samples, noise_y: y_samples, is_training: False})
            # gen_imgs = (gen_imgs + 1) / 2
            # imgs = [img[:, :, :] for img in gen_imgs]
            # # print(imgs)
            # # print(len(imgs))
            # #test_img1 = imgs[1]
            # # single_save(test_img1)
            # gen_imgs = montage(imgs)
            # gen_imgs = np.clip(gen_imgs, 0, 1)
            # imsave('../images/pb文件保存的测试3.jpg', gen_imgs)
