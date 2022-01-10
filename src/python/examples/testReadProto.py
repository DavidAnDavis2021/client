import tritonclient.grpc.service_pb2 as service
import tritonclient.grpc.model_config_pb2 as model_config


#import tritonclient.grpc as grpcclient
#import tritonclient.http as httpclient
#from tritonclient.utils import triton_to_np_dtype
#from tritonclient.utils import InferenceServerException

#filename = "./TMP_inception21k_onnx_ensemble_old.pb"
filename = "./PYTHON_MODEL_REQUEST.pb"
#filename = "./TMP_inception21k_onnx_ensemble.pb"

def printInputMsg(filename):
    s = service.ModelInferRequest()
    f = open(filename, "rb")
    s.ParseFromString(f.read())
    #s.inputs[0].contents.Clear()
    #s.ClearField("raw_input_contents")
    print(s)

print(filename)
printInputMsg(filename)
