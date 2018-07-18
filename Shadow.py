'''
working log （2018/07/18）：
to do：
    load images and json files (SYNCHRONIZE!)
    conditional input for discriminators
    processing 4-channel output of generator
    changing loss functions
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tfimage as im
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

Examples = collections.namedtuple("Examples", "paths, inputs, targets, bounds, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, local_predict_real, local_predict_fake, global_predict_real, global_predict_fake, local_discrim_loss, local_discrim_grads_and_vars, global_discrim_loss, global_discrim_grads_and_vars, gen_loss_local, gen_loss_global, gen_loss_L1, gen_grads_and_vars, train")

def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2

def localize(images, bounds):
    #as long as images and bounds are paired up
    imgs = []
    for i in range(len(images)):
        temp = images[i]
        c_row = bounds[i][0] + (bounds[i][1] - bounds[i][0]) // 2
        c_col = bounds[i][2] + (bounds[i][3] - bounds[i][2]) // 2
        if c_row < 128:
            c_row = 128
        elif c_row > 383:
            c_row = 383

        if c_col < 128:
            c_col = 128
        elif: c_col > 383:
            c_col = 383
            
        img = temp[c_row-128:c_row+128, c_col-128:c_col+128, : ]
        imgs.append(img)
    return imgs

def discrim_conv(batch_input, out_channels, stride, name="dis_conv2d"):
    input_shape = batch_input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [5, 5, input_shape[-1], out_channels], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(batch_input, w, strides=[1, stride, stride, 1], padding="VALID")
        output = batchnorm(output)
        output = tf.nn.relu(output)
        return output
            
def gen_conv(batch_input, out_channels, kernel=3, stride=2, function="relu", name="gen_conv2d"):
    input_shape = batch_input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [kernel, kernel, input_shape[-1], out_channels], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(batch_input, w, strides=[1, stride, stride, 1], padding="SAME")
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        output = batchnorm(output)
        if function == "relu":
            output = tf.nn.relu(output)
        elif function == "tanh":
            output = tf.nn.tanh(output)
        else:
            raise Exception("Invalid function")

        return output

def gen_deconv(batch_input, connect_input, out_channels, out_height, name="gen_deconv"):
    if connect_input == None:
        inputs = batch_input
    else:
        inputs = tf.concat([batch_input, connect_input], axis=3)
    inputs = tf.nn.relu(inputs)
    input_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [4, 4, out_channels, input_shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(inputs, w, output_shape=[batch_size, out_height, out_height, out_channels], strides=[1, 2, 2, 1], padding="SAME")
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        output = batchnorm(deconv)
        return output

def gen_dilate_conv(batch_input, rate, name="gen_dilate_conv"):
    input_shape = batch_input.get_shape().as_list()
    out_shape = [batch_size, input_shape[1], input_shape[2], input_shape[-1]]
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [3, 3, input_shape[-1], out_shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_shape[-1]], initializer=tf.constant_initializer(0.0))
        dilate_conv = tf.nn.atrous_conv2d(batch_input, w, rate=rate, padding="SAME")
        dilate_conv = tf.reshape(tf.nn.bias_add(dilate_conv, b), dilate_conv.get_shape())
        return dilate_conv
    

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


    '''
    -----------------------------------------------------------------------
    modify paths here
        -input(3 channels)
        -o1, s1, o2, s2, o3, s3
        -O
        -gt(3 channels)
        problems:
            *using tf.train.string_input_producer, randomly shuffles each paths every epoch. So image sets are not in sync
            *try to put images together (e.g. in a folder, or as a larger image) -- then okay
            *the implementation below assumes images are concatenated into a 512 by (512*13) greyscale image

            *but *bounds.json files still cannot be synchronized with shuffle
    read in all .json files, store bounds in list
    -----------------------------------------------------------------------
    '''
 
    input_paths = get_path("*.png")

    bounds = []
    for i in range(len(input_paths)):
        filename = "%dbounds.json" % i
        with open(filename, "w") as json_file:
            data = json.load(json_file)
            bound = [data["up"], data["low"], data["left"], data["right"]]
            bounds.append(bound)
    
    decode = tf.image.decode_png

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    input_paths = sorted(input_paths, key=lambda path: int(get_name(input_paths)))


    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
        raw_input = preprocess(raw_input)

        width = raw_input.get_shape()[1]
        seg = []
        for i in range(13):
            left_bound = (width * i) // 13
            right_bound = (width * (i + 1)) // 13
            seg[i] = raw_input[:,left_bound:right_bound,:]
        

        #putting all inputs together
        inputs = tf.concat([seg[0], seg[1], seg[2], seg[3], seg[4], seg[5], seg[6], seg[7], seg[8], seg[9]], axis=2)
        targets = tf.concat([seg[10], seg[11], seg[12]], axis=2)

        paths_batch, inputs_batch, targets_batch, bounds_batch = tf.train.batch([paths, inputs, targets, bounds], batch_size = batch_size)
        steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            bounds=bounds_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

def generator(generator_inputs):
    layers = []

    # encoders:
    # 1: [batch, 512, 512, 10] => [batch, 256, 256, 64]
    conv1 = gen_conv(generator_inputs, 64, 5, 1, "relu", "conv1")
    layers.append(conv1)
    # 2: [batch, 256, 256, 64] => [batch, 128, 128, 128]
    conv2 = gen_conv(conv1, 128, 3, 2, "relu", "conv2")
    layers.append(conv2)
    # 3: [batch, 128, 128, 128] => [batch, 64, 64, 256]
    conv3 = gen_conv(conv2, 256, 3, 2, "relu", "conv3")
    layers.append(conv3)
    # 4: [batch, 64, 64, 256] => [batch, 32, 32, 512]
    conv4 = gen_conv(conv3, 512, 3, 2, "relu", "conv4")
    layers.append(conv4)

    # dilators:
    dilate_conv1 = gen_dilate_conv(conv4, 2, "dilate_conv1")
    layers.append(dilate_conv1)

    dilate_conv2 = gen_dilate_conv(dilate_conv1, 4, "dilate_conv2")
    layers.append(dilate_conv2)

    dilate_conv3 = gen_dilate_conv(dilate_conv2, 8, "dilate_conv3")
    layers.append(dilate_conv3)

    dilate_conv4 = gen_dilage_conv(dilate_conv3, 16, "dilate_conv4")
    layers.append(dilate_conv4)

    # decoders:
    # 4: [batch, 32, 32, 512] => [batch, 64, 64, 256] (no skip layer)
    deconv4 = gen_deconv(dilate_conv4, connect_input=None, 256, 64, "deconv4")
    layers.append(deconv4)
    # 3: [batch, 64, 64, 256] => [batch, 128, 128, 128] (connect with layers[2])
    deconv3 = gen_deconv(deconv4, connect_input=layers[2], 128, 128, "deconv3")
    layers.append(deconv3)
    # 2: [batch, 128, 128, 128] => [batch, 256, 256, 64] (connect with layers[1])
    deconv2 = gen_deconv(deconv3, connect_input=layers[1], 64, 256, "deconv2")
    layers.append(deconv2)
    # 1: [batch, 256, 256, 64] => [batch, 512, 512, 4] (connect with layers[0])
    deconv1 = gen_deconv(deconv2, connect_input=layers[0], 4, 512, "deconv1")
    layers.append(deconv1)
        
    return layers[-1]

def create_model(inputs, targets, bounds):
    def local_discriminator(inputs, targets, bounds):
        #here, inputs are conditional inputs for discriminator
        #modify later
        loc_inputs = localize(inputs, bounds)
        loc_targets = localize(targets, bounds)

        # [256 * 256 * 6]
        loc_dis_inputs = tf.concat([loc_inputs, loc_targets], axis=3)
        layers = []

        #layer 1 => [128 * 128 * 64]
        l_1 = discrim_conv(loc_dis_inputs, 64, 2, "l_1")
        layers.append(l_1)
        #layer 2 => [64 * 64 * 128]
        l_2 = discrim_conv(l_1, 128, 2, "l_2")
        layers.append(l_2)
        #layer 3 => [32 * 32 * 256]
        l_3 = discrim_conv(l_2, 256, 2, "l_3")
        layers.append(l_3)
        #layer 4 => [31 * 31 * 512]
        l_4 = discrim_conv(l_3, 512, 1, "l_4")
        layers.append(l_4)
        #layer 5 => [30 * 30 * 1]
        l_5 = discrim_conv(l_4, 1, 1, "l_5")
        layers.append(l_5)
        
        return layers[-1]

        
    def global_discriminator(inputs, targets):
        #here, inputs are conditional inputs for discriminator
        #modify later
        
        #[512 * 512 * 6]
        glob_inputs = tf.concat([inputs, targets], axis=3)
        layers = []
                
        #layer 1 => [256 * 256 * 64]
        g_1 = discrim_conv(loc_dis_inputs, 64, 2, "g_1")
        layers.append(g_1)
        #layer 2 => [128 * 128 * 128]
        g_2 = discrim_conv(l_1, 128, 2, "g_2")
        layers.append(g_2)
        #layer 3 => [64 * 64 * 256]
        g_3 = discrim_conv(l_2, 256, 2, "g_3")
        layers.append(g_3)
        #layer 4 => [32 * 32 * 512]
        g_4 = discrim_conv(l_3, 512, 2, "g_4")
        layers.append(g_4)
        #layer 5 => [31 * 31 * 512]
        g_5 = discrim_conv(l_4, 512, 1, "g_5")
        layers.append(g_5)
        #layer 6 => [30, 30, 1]
        g_6 = discrim_conv(l_5, 1, 1, "g_6")
        layers.append(g_6)
        
        return layers[-1]

    with tf.variable_scope("generator"):
        outputs = generator(inputs)
        '''
        ----------------------------------------------------------------
        insert operations for 3-channel img and mask
        result will be [batch, 512, 512, 3] images
        ----------------------------------------------------------------
        '''

    with tf.name_scope("real_local_discriminator"):
        with tf.variable_scope("local_discriminator"):
            local_predict_real = local_discriminator(inputs, targets, bounds)
    with tf.name_scope("fake_local_discriminator"):
        with tf.variable_scope("local_discriminator", reuse=True):
            local_predict_fake = local_discriminator(inputs, outputs, bounds)

    with tf.name_scope("real_global_discriminator"):
        with tf.variable_scope("global_discriminator"):
            global_predict_real = global_discriminator(inputs, targets)
    with tf.name_scope("fake_global_discriminator"):
        with tf.variable_scope("global_discriminator"):
            global_predict_fake = global_discriminator(inputs, outputs)

    #loss functions: FIXME
    with tf.name_scope("local_discriminator_loss"):
        local_discrim_loss = tf.reduce_mean(-(tf.log(local_predict_real + EPS) + tf.log(1 - local_predict_fake + EPS)))
    with tf.name_scope("global_discriminator_loss"):
        global_discrim_loss = tf.reduce_mean(-(tf.log(global_predict_real + EPS) + tf.log(1 - global_predict_fake + EPS)))
    with tf.name_scope("generator_loss"):
        gen_loss_local = tf.reduce_mean(-tf.log(local_predict_fake + EPS))
        gen_loss_global = tf.reduce_mean(-tf.log(global_predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_local * local_weight + gen_loss_global * global_weight + gen_loss_L1 * l1_weight

    with tf.name_scope("local_discriminator_train"):
        local_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("local_discriminator")]
        local_discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        local_discrim_grads_and_vars = local_discrim_optim.compute_gradients(local_discrim_loss, var_list=local_discrim_tvars)
        local_discrim_train = local_discrim_optim.apply_gradients(local_discrim_grads_and_vars)
    with tf.name_scope("global_discriminator_train"):
        global_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("global_discriminator")]
        global_discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        global_discrim_grads_and_vars = global_discrim_optim.compute_gradients(global_discrim_loss, var_list=global_discrim_tvars)
        global_discrim_train = global_discrim_optim.apply_gradients(local_discrim_grads_and_vars)
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([local_discrim_train, global_disrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([local_discrim_loss, global_discrim_loss, gen_loss_local, gen_loss_global, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        local_predict_real=local_predict_real,
        local_predict_fake=local_predict_fake,
        global_predict_real=global_predict_real,
        global_predict_fake=global_predict_fake,
        local_discrim_loss=ema.average(local_discrim_loss),
        local_discrim_grads_and_vars=local_discrim_grads_and_vars,
        global_discrim_loss=ema.average(global_discrim_loss),
        global_discrim_grads_and_vars=global_discrim_grads_and_vars,
        gen_loss_local=ema.average(gen_loss_local),
        gen_loss_global=ema.average(gen_loss_global),
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

    model = create_model(examples.inputs, examples.targets, examples.bounds)
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

    with tf.name_scope("local_predict_real_summary"):
        tf.summary.image("local_predict_real", tf.image.convert_image_dtype(model.local_predict_real, dtype=tf.uint8))
    with tf.name_scope("local_predict_fake_summary"):
        tf.summary.image("local_predict_fake", tf.image.convert_image_dtype(model.local_predict_fake, dtype=tf.uint8))
    with tf.name_scope("global_predict_real_summary"):
        tf.summary.image("global_predict_real", tf.image.convert_image_dtype(model.global_predict_real, dtype=tf.uint8))
    with tf.name_scope("global_predict_fake_summary"):
        tf.summary.image("global_predict_fake", tf.image.convert_image_dtype(model.global_predict_fake, dtype=tf.uint8))

    tf.summary.scalar("local_discriminator_loss", model.local_discrim_loss)
    tf.summary.scalar("global_discriminator_loss", model.global_discrim_loss)
    tf.summary.scalar("generator_loss_local", model.gen_loss_local)
    tf.summary.scalar("generator_loss_global", model.gen_loss_global)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.local_discrim_grads_and_vars + model.global_discrim_grads_and_vars + model.gen_grads_and_vars:
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
                    fetches["local_discrim_loss"] = model.local_discrim_loss
                    fetches["global_discrim_loss"] = model.global_discrim_loss
                    fetches["gen_loss_local"] = model.gen_loss_local
                    fetches["gen_loss_global"] = model.gen_loss_global
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
                    print("local_discrim_loss", results["local_discrim_loss"])
                    print("global_discrim_loss", results["global_discrim_loss"])
                    print("gen_loss_local", results["gen_loss_local"])
                    print("gen_loss_global", results["gen_loss_global"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

        main()
                

        



















