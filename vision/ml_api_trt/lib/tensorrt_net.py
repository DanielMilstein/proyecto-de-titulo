# lib/tensorrt_net.py
import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context
from lib.meta import Meta


class TrtNet:
    def __init__(self, engine_path: str, meta_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.meta = Meta(meta_path)

        # Discover I/O tensor names (TRT 10+ API)
        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise RuntimeError(f"Expected exactly 1 input tensor, found {len(self.input_names)}: {self.input_names}")
        self.input_name = self.input_names[0]

        # Resolve input shape; if dynamic, we’ll set it on every call
        # Typical YOLO: [N,3,H,W]
        eng_shape = self.engine.get_tensor_shape(self.input_name)
        self.has_dynamic = any(dim == -1 for dim in eng_shape)

        # CUDA stream
        self.stream = cuda.Stream()

    def _preprocess(self, image, target_hw):
        h, w = target_hw
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # CHW
        img = np.expand_dims(img, axis=0)  # NCHW
        img /= 255.0
        return img

    def detect(self, meta, image, alt_names, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
        # Figure out input spatial size
        # Prefer engine static dims; otherwise assume 416x416 unless meta says otherwise
        in_shape = self.engine.get_tensor_shape(self.input_name)  # e.g., [-1, 3, 416, 416] or [1,3,640,640]
        if self.has_dynamic:
            # Guess from meta or fall back to common YOLO sizes
            H = 416
            W = 416
            # If engine gave non-negative H/W, use those
            if len(in_shape) == 4 and in_shape[2] > 0 and in_shape[3] > 0:
                H, W = int(in_shape[2]), int(in_shape[3])
        else:
            H, W = int(in_shape[2]), int(in_shape[3])

        inp = self._preprocess(image, (H, W))
        nbytes = inp.nbytes

        # If dynamic, set the input shape on the context now
        if self.has_dynamic:
            self.context.set_input_shape(self.input_name, inp.shape)  # (1,3,H,W)

        # Allocate device input and copy
        d_input = cuda.mem_alloc(nbytes)
        cuda.memcpy_htod_async(d_input, inp, self.stream)
        self.context.set_tensor_address(self.input_name, int(d_input))

        # Prepare outputs: shapes are known after we set input shape
        d_outputs = []
        h_outputs = []
        for name in self.output_names:
            oshape = tuple(self.context.get_tensor_shape(name))  # e.g., (1, num, 85) or (1, n, 4), etc.
            size = int(np.prod(oshape))
            host = np.empty(size, dtype=np.float32)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            d_outputs.append((name, dev))
            h_outputs.append((name, host, oshape))

        # Run
        self.context.execute_async_v3(self.stream.handle)

        # D2H
        for (_, dev), (name, host, _) in zip(d_outputs, h_outputs):
            cuda.memcpy_dtoh_async(host, dev, self.stream)
        self.stream.synchronize()

        # Reshape to tensors
        np_outputs = [host.reshape(shape) for (name, host, shape) in h_outputs]

        # Post-process with your existing ONNX path
        width = image.shape[1]
        height = image.shape[0]
        dets = post_processing(np_outputs, width, height, thresh, nms, self.meta.names)
        return dets[0]


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)

def post_processing(output, width, height, conf_thresh, nms_thresh, names):
    box_array = output[0]
    confs = output[1]

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # [batch, num, 4]
    box_array = box_array[:, :, 0]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    box_x1x1x2y2_to_xcycwh_scaled = lambda b: \
        (
            float(0.5 * width * (b[0] + b[2])),
            float(0.5 * height * (b[1] + b[3])),
            float(width * (b[2] - b[0])),
            float(width * (b[3] - b[1]))
         )
    dets_batch = []
    for i in range(box_array.shape[0]):

        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms for each class
        for j in range(num_classes):

            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)

            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])

        detections = [(names[b[6]], float(b[4]), box_x1x1x2y2_to_xcycwh_scaled((b[0], b[1], b[2], b[3]))) for b in bboxes]
        dets_batch.append(detections)


    return dets_batch

