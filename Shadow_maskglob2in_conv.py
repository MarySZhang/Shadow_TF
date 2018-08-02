'''
working log (2018/07/23)
change parameters:
    batch_size, loss weights, stddev for noise in mask_discriminator
    MAKE MASK ANOTHER L1 LOSS, WEIGHT=100
    Structure change (which one?)
    Dropout?
'''

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
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir")
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--max_epochs", type=int)

a = parser.parse_args()

# change parameters here
summary_freq = 50
progress_freq = 100
trace_freq = 0
display_freq = 0
save_freq = 5000
aspect_ratio = 1.0
batch_size = 1
ngf = 64
ndf = 64
lr = 0.0002
beta1 = 0.5
l1_weight = 1.0
mask_weight = 0.1
mask_l1_weight = 1.0
EPS = 1e-12
CROP_SIZE = 256
noise_dev = 0.02

Examples = collections.namedtuple("Examples", "paths, inputs, targets, bounds, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, mask_outputs, mask_predict_real, mask_predict_fake, mask_discrim_loss, gen_loss_mask, gen_loss_L1, gen_loss_mask_L1, train")

def preprocess(image):
    with tf.name_scope("preprocess"):
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        return (image + 1) / 2


#FIXME!!!!
def localize(image, bound):
    bds = tf.cast(bound, tf.float32)
    bds = bds / 256
    img = tf.image.crop_and_resize(image, bds, list(range(batch_size)), [128, 128])
    return img


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)



def discrim_conv(batch_input, out_channels, stride, name="dis_conv2d", relu=True, norm=True):
    with tf.variable_scope(name) as scope:
        inputs = batch_input
        if relu == True:
            inputs = lrelu(batch_input, 0.2)
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        output = tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        if norm == True:
            output = batchnorm(output)
        return output
            
def gen_conv(batch_input, out_channels, kernel=3, stride=2, name="gen_conv2d", norm=True):
    with tf.variable_scope(name) as scope:
        if norm == True:
            inputs = lrelu(batch_input, 0.2)
        else:
            inputs = batch_input
        output = tf.layers.conv2d(inputs, out_channels, kernel_size=kernel, strides=(stride, stride), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        if norm == True:
            output = batchnorm(output)
        return output

def gen_deconv(batch_input, connect_input, out_channels, name="gen_deconv", norm=True):
    with tf.variable_scope(name) as scope:
        if connect_input == None:
            inputs = batch_input
        else:
            inputs = tf.concat([batch_input, connect_input], axis=3)
        deconv = tf.layers.conv2d_transpose(inputs, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        if norm == True:
            deconv = batchnorm(deconv)
        return deconv

def gen_dilate_conv(batch_input, rate, name="gen_dilate_conv"):
    input_shape = batch_input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        dilate_conv = tf.layers.conv2d(batch_input, input_shape[-1], kernel_size=3, strides=(1, 1), padding="same", dilation_rate=rate, kernel_initializer=tf.random_normal_initializer(0, 0.02))
        return dilate_conv
    

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name
    
    def get_num(path, post):
        name = get_name(path)
        spt = name.split(post)
        return spt[0]

    def sort_num(paths, post):
        return sorted(paths, key=lambda path: int(get_num(path, post)))

    def get_path(name):
        paths = glob.glob(os.path.join(a.input_dir, "*"+name+".png"))
        return sort_num(paths, name)

    input_paths = get_path("threeAddedOne")
    obj_0_paths = get_path("objectMask_0")
    obj_1_paths = get_path("objectMask_1")
    obj_2_paths = get_path("objectMask_2")
    obj_3_paths = get_path("objectMask_3")
    sha_0_paths = get_path("shadowMask_0")
    sha_1_paths = get_path("shadowMask_1")
    sha_2_paths = get_path("shadowMask_2")
    sha_3_paths = get_path("shadowMask_3")
    target_paths = get_path("image")


    
    with tf.name_scope("load_images"):
        
        def read_input(paths_):
            path_queue = tf.train.string_input_producer(paths_, shuffle=False)
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = tf.image.decode_png(contents)
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
            raw_input = preprocess(raw_input)
            raw_input = tf.image.resize_images(raw_input, [256, 256])
            return raw_input, paths
        
        input_imgs, paths = read_input(input_paths)
        obj_0, _ = read_input(obj_0_paths)
        obj_1, _ = read_input(obj_1_paths)
        obj_2, _ = read_input(obj_2_paths)
        obj_3, _ = read_input(obj_3_paths)
        sha_0, _ = read_input(sha_0_paths)
        sha_1, _ = read_input(sha_1_paths)
        sha_2, _ = read_input(sha_2_paths)
        sha_3, _ = read_input(sha_3_paths)
        targets, _ = read_input(target_paths)
        targets = targets[:,:,0:3]

        #putting all inputs together
        inputs = tf.concat([input_imgs[:,:,0:3], obj_0, obj_1, obj_2, sha_0, sha_1, sha_2, obj_3], axis=2)
        targets = tf.concat([targets, sha_3], axis=2)
        
        inputs.set_shape([256, 256, 10])
        targets.set_shape([256, 256, 4])
        #bounds
        bound_paths = glob.glob(os.path.join(a.input_dir, "*bound.json"))
        bound_paths = sort_num(bound_paths, "bound")
        
        def map_jsonfile_to_bound(filename):
            with open(filename) as f:
                data = json.load(f)
            return [data["top"], data["left"], data["bottom"], data["right"]]
        
        bound_list = list(map(map_jsonfile_to_bound, bound_paths))
        
        def bound_queue(bound_list):
            bound_tensor = tf.convert_to_tensor(bound_list, dtype=tf.int32)
            fq = tf.FIFOQueue(capacity=32, dtypes=tf.int32)
            fq_enqueue_op = fq.enqueue_many([bound_tensor])
            tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_enqueue_op] * 1))
            return fq
        
        bound_q = bound_queue(bound_list)
        bds = bound_q.dequeue()#actually only a bound at once
        bds.set_shape([4])#needed by tf.train.shuffle_batch


        paths_batch, inputs_batch, targets_batch, bounds_batch = tf.train.batch([paths, inputs, targets, bds], batch_size = batch_size)
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
    # 0: [batch, 256, 256, 10] => [batch, 128, 128, 64]
    conv1 = gen_conv(generator_inputs, 64, 5, 2, "conv1", norm=False)
    layers.append(conv1)
    # 1: [batch, 128, 128, 64] => [batch, 64, 64, 128]
    conv2 = gen_conv(conv1, 128, 3, 2, "conv2")
    layers.append(conv2)
    # 2: [batch, 64, 64, 128] => [batch, 32, 32, 256]
    conv3 = gen_conv(conv2, 256, 3, 2, "conv3")
    layers.append(conv3)
    # 3: [batch, 32, 32, 256] => [batch, 16, 16, 512]
    conv4 = gen_conv(conv3, 512, 3, 2, "conv4")
    layers.append(conv4)

    # 4-7 dilators:
    dilate_conv1 = gen_dilate_conv(conv4, 2, "dilate_conv1")
    layers.append(dilate_conv1)

    dilate_conv2 = gen_dilate_conv(dilate_conv1, 4, "dilate_conv2")
    layers.append(dilate_conv2)

    dilate_conv3 = gen_dilate_conv(dilate_conv2, 8, "dilate_conv3")
    layers.append(dilate_conv3)

    dilate_conv4 = gen_dilate_conv(dilate_conv3, 16, "dilate_conv4")
    layers.append(dilate_conv4)

    # decoders:
    # 8: [batch, 16, 16, 512] => [batch, 32, 32, 256] (connect with layers[3])
    deconv0 = gen_deconv(dilate_conv4, layers[3], 256, "deconv0")
    layers.append(deconv0)
    # 9: [batch, 32, 32, 256] => [batch, 64, 64, 128] (connect with layers[2])
    deconv1 = gen_deconv(deconv0, layers[2], 128, "deconv1")
    layers.append(deconv1)
    # 10: [batch, 64, 64, 128] => [batch, 128, 128, 64] (connect with layers[1])
    deconv2 = gen_deconv(deconv1, layers[1], 64, "deconv2")
    layers.append(deconv2)
    # 11: [batch, 128, 128, 64] => [batch, 256, 256, 4] (connect with layers[0])
    deconv3 = gen_deconv(deconv2, layers[0], 4, "deconv3", norm=False)
    deconv3 = tf.tanh(deconv3)
    layers.append(deconv3)   
        
    return layers[-1]

def create_model(inputs, targets, bounds):
    masks = targets[:,:,:,3:]
    targets = targets[:,:,:,:3]
        

    
    def mask_discriminator(inputs, masks):
        #noise
        noise = tf.random_normal(shape=tf.shape(masks), mean=0.0, stddev=noise_dev, dtype=tf.float32)
        targets = masks + noise
        con_inputs = tf.concat([inputs[:,:,:,9:], targets], axis=3)
        
        #[128 * 128 * 11]
        layers = []
                
        #layer 1 => [128 * 126 * 32]
        m_1 = discrim_conv(con_inputs, 32, 2, "m_1", relu=False, norm=False)
        layers.append(m_1)
        #layer 2 => [64 * 64 * 64]
        m_2 = discrim_conv(m_1, 64, 2, "m_2")
        layers.append(m_2)
        #layer 3 => [32 * 32 * 128]
        m_3 = discrim_conv(m_2, 128, 2, "m_3")
        layers.append(m_3)
        #layer 4 => [16 * 16 * 256]
        m_4 = discrim_conv(m_3, 512, 2, "m_4")
        layers.append(m_4)
        #layer 5 => [8 * 8 * 512]
        m_5 = discrim_conv(m_4, 512, 2, "m_5")
        layers.append(m_5)
        #layer 6 => [4, 4, 512]
        m_6 = discrim_conv(m_5, 512, 2, "m_6")
        layers.append(m_6)
        #layer 7 => [2, 2, 512]
        m_7 = discrim_conv(m_6, 512, 2, "m_7")
        layers.append(m_7)
        #layer 8 => [1, 1, 512]
        m_8 = discrim_conv(m_7, 512, 2, "m_8", norm=False)
        m_8 = tf.contrib.layers.flatten(m_8)
        layers.append(m_8)
        
        return layers[-1]


    with tf.variable_scope("generator"):
        gen_outputs = generator(inputs)
        img_outputs = gen_outputs[:,:,:,:3]
        mask_outputs = gen_outputs[:,:,:,3:]
        #outputs = inputs * (1-mask) + gen * mask
        mask_3outs = tf.image.grayscale_to_rgb(mask_outputs)
        mask_multiplyer = deprocess(mask_3outs)
        
        outputs = tf.multiply(inputs[:,:,:,:3], (1 - mask_multiplyer)) + tf.multiply(img_outputs, mask_multiplyer)
        outputs.set_shape([batch_size, 256, 256, 3])

    with tf.name_scope("real_mask_discriminator"):
        with tf.variable_scope("mask_discriminator"):
            mask_predict_real = mask_discriminator(inputs, masks)
    with tf.name_scope("fake_mask_discriminator"):
        with tf.variable_scope("mask_discriminator", reuse=True):
            mask_predict_fake = mask_discriminator(inputs, mask_outputs)

    #loss functions: FIXME
    with tf.name_scope("mask_discriminator_loss"):
        mask_discrim_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_predict_real, labels=tf.ones_like(mask_predict_real)))
        mask_discrim_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_predict_fake, labels=tf.zeros_like(mask_predict_fake)))
        mask_discrim_loss = mask_discrim_real_loss + mask_discrim_fake_loss
    with tf.name_scope("generator_loss"):
        gen_loss_mask = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_predict_fake, labels=tf.ones_like(mask_predict_fake)))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss_mask_L1 = tf.reduce_mean(tf.abs(masks - mask_outputs))
        gen_loss = gen_loss_mask * mask_weight + gen_loss_L1 * l1_weight + gen_loss_mask_L1 * mask_l1_weight

    with tf.name_scope("mask_discriminator_train"):
        mask_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("mask_discriminator")]
        mask_discrim_optim = tf.train.AdamOptimizer(lr, beta1)
        mask_discrim_grads_and_vars = mask_discrim_optim.compute_gradients(mask_discrim_loss, var_list=mask_discrim_tvars)
        mask_discrim_train = mask_discrim_optim.apply_gradients(mask_discrim_grads_and_vars)
    with tf.name_scope("generator_train"):
        with tf.control_dependencies([mask_discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(lr, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        mask_predict_real=mask_predict_real,
        mask_predict_fake=mask_predict_fake,
        mask_discrim_loss=mask_discrim_loss,
        gen_loss_mask=gen_loss_mask,
        gen_loss_L1=gen_loss_L1,
        gen_loss_mask_L1=gen_loss_mask_L1,
        outputs=outputs,
        mask_outputs=mask_outputs,
        train=tf.group(incr_global_step, gen_train),
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets", "masks"]:
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
    masks = deprocess(model.mask_outputs)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_images"):
        converted_inputs = convert(inputs[:,:,:,:3])
        converted_targets = convert(targets[:,:,:,:3])
        converted_outputs = convert(outputs)
        converted_masks = convert(tf.image.grayscale_to_rgb(masks))

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
            "masks": tf.map_fn(tf.image.encode_png, converted_masks, dtype=tf.string, name="mask_pngs"),
        }
   #summaries
    with tf.name_scope("summaries"):
        tf.summary.image("inputs", converted_inputs)
        tf.summary.image("targets", converted_targets)
        tf.summary.image("outputs", converted_outputs)
        tf.summary.image("masks", converted_masks)

#    with tf.name_scope("local_predict_real_summary"):
#        tf.summary.image("local_predict_real", tf.image.convert_image_dtype(model.local_predict_real, dtype=tf.uint8))
#    with tf.name_scope("local_predict_fake_summary"):
#        tf.summary.image("local_predict_fake", tf.image.convert_image_dtype(model.local_predict_fake, dtype=tf.uint8))
#    with tf.name_scope("global_predict_real_summary"):
#        tf.summary.image("global_predict_real", tf.image.convert_image_dtype(model.global_predict_real, dtype=tf.uint8))
#    with tf.name_scope("global_predict_fake_summary"):
#        tf.summary.image("global_predict_fake", tf.image.convert_image_dtype(model.global_predict_fake, dtype=tf.uint8))
#    with tf.name_scope("mask_predict_real_summary"):
#        tf.summary.image("mask_predict_real", tf.image.convert_image_dtype(model.mask_predict_real, dtype=tf.uint8))
#    with tf.name_scope("mask_predict_fake_summary"):
#        tf.summary.image("mask_predict_fake", tf.image.convert_image_dtype(model.mask_predict_fake, dtype=tf.uint8))


    tf.summary.scalar("mask_discriminator_loss", model.mask_discrim_loss)
    tf.summary.scalar("generator_loss_mask", model.gen_loss_mask)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
    tf.summary.scalar("generator_loss_mask_L1", model.gen_loss_mask_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    #for grad, var in model.local_discrim_grads_and_vars + model.global_discrim_grads_and_vars + model.gen_grads_and_vars:
#        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (trace_freq > 0 or summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs


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
                    fetches["mask_discrim_loss"] = model.mask_discrim_loss
                    fetches["gen_loss_mask"] = model.gen_loss_mask
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["gen_loss_mask_L1"] = model.gen_loss_mask_L1
                
                if should(summary_freq):
                    fetches["summary"] = sv.summary_op
                
                try:
                    results = sess.run(fetches, options=None, run_metadata=None)
                except Exception as ex:
                    print(ex)
                    return
                
                if should(summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                
                if should(progress_freq):
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * batch_size / (time.time() - start)
                    remaining = (max_steps - step) * batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("mask_discrim_loss", results["mask_discrim_loss"])
                    print("gen_loss_mask", results["gen_loss_mask"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                    print("gen_loss_mask_L1", results["gen_loss_mask_L1"])
                    sys.stdout.flush()
                
                if should(save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                
#                if sv.should_stop():
#                    break

main()
print('the end')
                

 
