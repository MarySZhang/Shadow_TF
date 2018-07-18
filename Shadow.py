from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--max_epochs", type=int)

a = parser.parse_args()

summary_freq = 100
progress_freq = 50
trace_freq = 0
display_freq = 0
save_freq = 5000
aspect_ratio = 1.0
batch_size = 1
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
l1_weight = 100.0
gan_weight = 1.0
EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, targets, masks, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")

def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2

def localize(images, masks):
    imgs = []
    for i in range(len(images)):
        temp = images[i]
        center_idx = center_idxes[i]

        if center_idx['x'] < 64:
            left_bound = 0
            right_bound = 127
        elif center_idx['x'] > 191:
            left_bound = 128
            right_bound = 255
        else:
            left_bound = center_idx['x'] - 64
            right_bound = center_idx['x'] + 64

        if center_idx['y'] < 64:
            up_bound = 0
            low_bound = 127
        elif center_idx['y'] > 191:
            up_bound = 128
            low_bound = 255
        else:
            up_bound = center_idx['y'] - 64
            low_bound = center_idx['y'] + 64

        img = temp[up_bound:low_bound, left_bound:right_bound, : ]
        imgs.append(img)
    return imgs

def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernal_size=4, strides=(stride, stride), padding="valid", kernal_initializer=tf.random_normal_initializer(0, 0.02))
        
def gen_conv(batch_input, out_channels):
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernal_initializer=initializer)

def gen_deconv(batch_input, out_channels):
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    def get_path(name):
        return glob.glob(os.path.join(a.input_dir, name))

    img_input_paths = get_path("*.png")
    obj_1_input_paths = get_path("*objectMask1.png")
    obj_2_input_paths = get_path("*objectMask2.png")
    obj_3_input_paths = get_path("*objectMask3.png")
    sha_1_input_paths = get_path("*shadowMask1.png")
    sha_2_input_paths = get_path("*shadowMask2.png")
    sha_3_input_paths = get_path("*shadowMask3.png")
    obj_0_input_paths = get_path("*objectMask.png")
    gt_input_paths = get_path("*truth.png")
    
    all_paths = [img_input_paths,
                 obj_1_input_paths,
                 obj_2_input_paths,
                 obj_3_input_paths,
                 sha_1_input_paths,
                 sha_2_input_paths,
                 sha_3_input_paths,
                 obj_0_input_paths,
                 gt_input_paths]
    
    decode = tf.image.decode_png

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    for input_paths in all_paths:
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)

    def load_images(input_paths):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        raw_input = preprocess(raw_input)
        return raw_input

    with tf.name_scope("load_images"):
        images = load_images(img_input_paths)
        #separate three channels of images
        img_r = images[:,:,0]
        img_g = images[:,:,1]
        img_b = images[:,:,2]
        obj_1 = load_images(obj_1_input_paths)
        obj_2 = load_images(obj_2_input_paths)
        obj_3 = load_images(obj_3_input_paths)
        sha_1 = load_images(sha_1_input_paths)
        sha_2 = load_images(sha_2_input_paths)
        sha_3 = load_images(sha_3_input_paths)
        obj_0 = load_images(obj_0_input_paths)
        targets = load_images(gt_input_paths)
        targets.set_shape([None, None, 3])

        #putting all inputs together
        inputs = tf.concat([img_r, img_g, img_b, obj_1, obj_2, obj_3, sha_1, sha_2, sha_3, obj_0], axis=2)

        paths_batch, inputs_batch, targets_batch, masks_batch = tf.train.batch([paths, inputs, targets, obj_0], batch_size = batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            masks=masks_batch
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

def generator(generator_inputs, generator_outputs_channels):
    layers = []

    output = gen_conv(generator_inputs, ngf)
    layers.append(output)

    layer_specs = [
        ngf * 2,
        ngf * 4,
        ngf * 8,
        ngf * 8,
        ngf * 8,
        ngf * 8,
        ngf * 8,
    ]

    for out_channels in layer_specs:
        rectified = lrelu(layers[-1], 0.2)
        convolved = gen_conv(rectified, out_channels)
        output = batchnorm(convolved)
        layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),
        (ngf * 8, 0,5),
        (ngf * 8, 0.5),
        (ngf * 8, 0.0),
        (ngf * 4, 0.0),
        (ngf * 2, 0.0),
        (ngf, 0.0),
    ]

    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = len(layers) - decoder_layer - 1
        if decoder_layer == 0:
            input = layers[-1]
        else:
            input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, out_channels)
        output = batchnorm(output)

        if dropout > 0.0:
            output = tf.nn.dropout(output, keep_prob=1 - dropout)

        layers.append(output)

    input = tf.concat([layers[-1], layers[0]], axis=3)
    rectified = tf.nn.relu(input)
    output = gen_deconv(rectified, generator_outputs_channels)
    output = tf.tanh(output)
    layers.append(output)

    return layers[-1]

def create_model(inputs, targets, masks):
    def local_discriminator(inputs, targets, masks):
        loc_inputs = localize(inputs, masks)
        loc_targets = localize(targets, masks)

        # [128 * 128 * 6]
        loc_dis_inputs = tf.concat([loc_inputs, loc_targets], axis=3)

        layers = []

        #layer 1 => [64 * 64 * 64]
        convolved = discrim_conv(loc_dis_inputs, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

        #layer 2 => [32 * 32 * 128]
        convolved = discrim_conv(layers[-1], ndf * 2, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

        #layer 3 => [31 * 31 * 256]
        convolved = discrim_conv(layers[-1], ndf * 4, stride=1)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

        #layer 4 => [30 * 30 * 1]
        convolved = discrim_conv(layers[-1], out_channels=1, stride=1)
        output = tf.sigmoid(convolved)
        layers.append(output)
        return layers[-1]

        
    def global_discriminator(inputs, targets):
        glob_inputs = tf.concat([inputs, targets])
        layers = []

        #layer 1
        convolved = discrim_conv(glob_inputs, ndf, stride=2)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

        #layers 2-6
        for i in range(5):
            out_channels = ndf * min(2**(i+1), 8)
            convolved = discrim_conv(layers[-1], out_channels, stride=2)
            normalized = batchnorm(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

        #layer 7: flatten
        flatten = tf.contrib.layers.flatten(layers[-1])
        output = linear(flatten, 1024, name="flatten")
        return output

    def discriminator(inputs, targets, center_idxes):
        loc = local_discriminator(inputs, targets, center_idxes)
        glob = global_discriminator(inputs, targets)
        output = tf.concat([loc, glob], axis=1)
        return output

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = generator(inputs, out_channels)

    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            predict_real = discriminator(inputs, targets, center_idxes)
    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            predict_fake = discriminator(inputs, outputs, center_idxes)

    with tf.name_scope("discriminator_loss"):
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )

def same_images(fetches, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets

def main():
    seed = random.randint(0, 2**31 - 1)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test":
        if a.checkpoint is None:
            raise Exception("checkpoint required  for test mode")

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    model = create_model(examples.inputs, examples.targets, examples.center_idxes)
    inputs = deprocess(examples.inputs)
    targets = deprocess(examples.targets)
    outputs = deprocess(model.outputs)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_images"):
        converted_inputs = convert(inputs)
        converted_targets = convert(targets)
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }
    #summaries
    with tf.name_scope("summaries"):
        tf.summary.image("inputs", converted_inputs)
        tf.summary.image("targets", converted_targets)
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))
    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * max_epochs


        if a.mode == "test":
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
            print("rate", (time.time() - start) / max_steps)

        else:
            #training
            start = time.time()
            for step in range(max_steps):
                def should(freq):
                    return freq  > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(summary_freq):
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches, options=None, run_metadata=None)

                if should(summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                if should(progress_freq):
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * batch_size / (time.time() - start)
                    remaining = (max_steps - step) * batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, ramaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

        main()
                

        



















