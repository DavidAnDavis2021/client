#!usr/bin/env python
import argparse
import numpy as np
from PIL import Image
import os
import sys
import struct

import grpc

from tritonclient.grpc import service_pb2, service_pb2_grpc
import tritonclient.grpc.model_config_pb2 as mc

FLAGS = None


def model_dtype_to_np(model_dtype):
    if model_dtype == "BOOL":
        return bool
    elif model_dtype == "INT8":
        return np.int8
    elif model_dtype == "INT16":
        return np.int16
    elif model_dtype == "INT32":
        return np.int32
    elif model_dtype == "INT64":
        return np.int64
    elif model_dtype == "UINT8":
        return np.uint8
    elif model_dtype == "UINT16":
        return np.uint16
    elif model_dtype == "FP16":
        return np.float16
    elif model_dtype == "FP32":
        return np.float32
    elif model_dtype == "FP64":
        return np.float64
    elif model_dtype == "BYTES":
        return np.dtype(object)
    return None


def deserialize_bytes_tensor(encoded_tensor):
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return (np.array(strs, dtype=np.object_))


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception("expecting 1 output, got {}".format(
            len(model_metadata.outputs)))

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception("expecting output datatype to be FP32, model '" +
                        model_metadata.name + "' output type is " +
                        output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    return (input_metadata.name, output_metadata.name, input_metadata.datatype)



def postprocess(responses):
    for response in responses:
        print(response)
        batched_result = deserialize_bytes_tensor(response.raw_output_contents[0])
        contents = np.reshape(batched_result, response.outputs[0].shape)
        for (index, results) in enumerate(contents):
            #print("Image '{}':".format(filenames[index]))
            print("Image '{}':".format("TEST"))
            for result in results:
                cls = "".join(chr(x) for x in result).split(':')
                print(cls)
                #print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))

    """
    Post-process response to show classifications.
    """
    if len(response.outputs) != 1:
        raise Exception("expected 1 output, got {}".format(len(
            response.outputs)))

    if len(response.raw_output_contents) != 1:
        raise Exception("expected 1 output content, got {}".format(
            len(response.raw_output_contents)))


    #if len(contents) != batch_size:
    #    raise Exception("expected {} results, got {}".format(
    #        batch_size, len(contents)))
    #if len(filenames) != batch_size:
    #    raise Exception("expected {} filenames, got {}".format(
    #        batch_size, len(filenames)))


def requestGenerator(input_name, output_name, dtype, FLAGS,
                     result_filenames):
    request = service_pb2.ModelInferRequest()
    request.model_name = FLAGS.model_name
    request.model_version = FLAGS.model_version

    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = output_name
    output.parameters['classification'].int64_param = FLAGS.classes
    #print("TEST ", output)
    request.outputs.extend([output])

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = input_name
    input.datatype = dtype


    # Preprocess image into input data according to model requirements
    # Preprocess the images into input data according to model
    # requirements
    image_data = []
    for filename in filenames:
        #img = Image.open(filename)
        #input_bytes = img.tobytes()
        file = open(filename,"rb")
        input_bytes = file.read()

        request.ClearField("inputs")
        request.ClearField("raw_input_contents")
        input.shape.extend([1,len(input_bytes)])
        request.inputs.extend([input])
        result_filenames.append(filename)
        request.raw_input_contents.extend([input_bytes])
        yield request



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use streaming inference API')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument(
        '-x',
        '--model-version',
        type=str,
        required=False,
        default="",
        help='Version of model. Default is to use latest version.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c',
                        '--classes',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument(
        '-s',
        '--scaling',
        type=str,
        choices=['NONE', 'INCEPTION', 'VGG'],
        required=False,
        default='NONE',
        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(FLAGS.url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Make sure the model matches our requirements, and get some
    # properties of the model that we need for preprocessing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=FLAGS.model_name, version=FLAGS.model_version)
    metadata_response = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(name=FLAGS.model_name,
                                                    version=FLAGS.model_version)
    config_response = grpc_stub.ModelConfig(config_request)

    input_name, output_name, dtype = parse_model(
        metadata_response, config_response.config)

    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    result_filenames = []

    # Send request
    for request in requestGenerator(input_name, output_name, dtype, FLAGS, result_filenames):
        print("CHECK THIS VALUE", request.outputs[0].parameters['classification'])
        f = open("PYTHON_MODEL_REQUEST.pb", "wb")
        f.write(request.SerializeToString())
        f.close()
        responses.append(grpc_stub.ModelInfer(request))


    error_found = False

    postprocess(responses)

    print("PASS")
