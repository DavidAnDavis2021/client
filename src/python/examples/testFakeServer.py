#Modified from https://github.com/grpc/grpc/blob/master/examples/python/helloworld/greeter_server.py

from concurrent import futures
import logging

import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc


class Greeter(service_pb2_grpc.GRPCInferenceServiceServicer):
    def ServerLive(self, request, context):
        print("HELLO Live")
        response = service_pb2.ServerLiveResponse()
        response.live = True
        return response
        

    def ServerReady(self, request, context):
        print("HELLO Ready")
        return

    def ModelReady(self, request, context):
        print("HELLO Model ready")
        return "HELLO model Ready"


    def ServerMetadata(self, request, context):
        print("HELLO metadata ")
        return "HELLO metadata"

    def ModelMetadata(self, request, context):
        print("HELLO model metadata")
        return "HELLO model metadata"


    def ModelInfer(self, request, context):
        print("HELLO Infer")
        print(request)
        f = open("PYTHON_SERVER.pb", "wb")
        f.write(request.SerializeToString())
        f.close()
        print("Written results")
        print(context)
        return "hello infer"

    def ModelStreamInfer(self, request, context):
        print("HELLO Stream infer")
        return "HELLO stream infer"

    def ModelConfig(self, request, context):
        print("HELLO model config")
        return "HELLO model config"

    def ModelStatistics(self, request, context):
        print("HELLO stats")
        return "HELLO model stats"

    def RepositoryIndex(self, request, context):
        print("HELLO repo")
        return "HELLO repo"



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:8101')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
