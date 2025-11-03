#!python3

# pylint: disable=R, W0401, W0614, W0703
from enum import Enum
from lib.meta import Meta
from os import path

alt_names = None

trt_ready = True
darknet_ready = False
onnx_ready = False

try:
    from lib.tensorrt_net import TrtNet
except Exception as e:
    print(f'Error during importing TrtNet! - {e}')
    trt_ready = False


def load_net(config_path, meta_path, weights_path=None):

    def try_loading_net(net_config_priority):
        for net_config in net_config_priority:
            weights = net_config['weights_path']
            use_gpu = net_config.get('use_gpu', True)

            try:
                print(f'----- Trying to load weights: {weights} - use_gpu = {use_gpu} -----')
                if weights.endswith(".engine"):
                    if not trt_ready:
                        raise Exception('Not loading TensorRT net due to previous import failure.')
                    net_main = TrtNet(weights, meta_path)
                else:
                    raise Exception(f'Can not recognize net from weights file suffix: {weights}')

                print('Succeeded!')
                return net_main
            except Exception as e:
                print(f'Failed! - {e}')

        raise Exception(f'Failed to load any net after trying: {net_config_priority}')

    global alt_names  # pylint: disable=W0603

    model_dir = path.join(path.dirname(path.realpath(__file__)), '..', 'model')
    net_config_priority = [
        dict(weights_path=path.join(model_dir, 'model.engine'), use_gpu=True),
        dict(weights_path=path.join(model_dir, 'model-weights.onnx'), use_gpu=True),
        dict(weights_path=path.join(model_dir, 'model-weights.onnx'), use_gpu=False),
        dict(weights_path=path.join(model_dir, 'model-weights.darknet'), use_gpu=True),
        dict(weights_path=path.join(model_dir, 'model-weights.darknet'), use_gpu=False),
    ]

    if weights_path is not None:
        # if user explicitly pointed to a file, try TRT first if it is an engine
        net_config_priority = [
            dict(weights_path=weights_path, use_gpu=True),
            dict(weights_path=weights_path, use_gpu=False)
        ]

    net_main = try_loading_net(net_config_priority)

    if alt_names is None:
        try:
            meta = Meta(meta_path)
            alt_names = meta.names
        except Exception:
            pass

    return net_main


def detect(net, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    return net.detect(net.meta, image, alt_names, thresh, hier_thresh, nms, debug)

