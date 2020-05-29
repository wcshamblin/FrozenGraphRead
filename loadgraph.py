#!/usr/bin/python3
import tensorflow as tf
import argparse
ps = argparse.ArgumentParser(description='Load a frozen graph')
ps.add_argument("fgraph", type=str, help='Path to frozen graph')
ps.add_argument("-i", "--inode", type=str, help="Name of input tensor node (if not declared, list tensors)")
ps.add_argument("-o", "--onode", type=str, help="Name of output tensor node (if not declared, list tensors)")

args=ps.parse_args()

def load_graph(frozen_graph_filename):
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

tf.compat.v1.disable_eager_execution()

def readimg(file_name,
            input_height,
            input_width,
            input_mean=0,
            input_std=255):
    image_str_tensor = tf.compat.v1.placeholder(tf.string, shape=[None], name= 'encoded_image_string_tensor')
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.compat.v1.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    elif file_name.endswith(".jpg") or file_name.endswith(".jpeg"):
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
    else:
        print("File format not recognized")
        return 1
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.compat.v1.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    image = map_fn(normalized, image_str_tensor, back_prop=False, dtype=tf.uint8)
    # image = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(normalized, image_str_tensor))
    # sess = tf.compat.v1.Session()
    # result = sess.run(normalized)
    return image


fgraph = load_graph(args.fgraph)

if args.inode and args.onode:
    img = readimg("cardinal.jpg", 720, 540)
    inode = fgraph.get_operation_by_name(args.inode)
    onode = fgraph.get_operation_by_name(args.onode)
    with tf.compat.v1.Session(graph=fgraph) as tfs:
        output = tfs.run(onode.outputs[0], {inode.outputs[0]: img})
    print(output)
else:
    print("Nodes:")
    nodes = [op.name for op in fgraph.get_operations()]
    for node in nodes:
        print(node)