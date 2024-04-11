import time

from onnxruntime import InferenceSession
from transformers import AutoConfig, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

from ortrunner.utils.utils import logger
from ortrunner.custom_ortrunner import CustomORTModelRunner


class ONNXRunner:
    def __init__(self, init=False, use_cache=True, use_io_binding=False):
        self.init = init
        if self.init:
            self.load(use_cache=use_cache, use_io_binding=use_io_binding)

    def load(self, use_cache=True, use_io_binding=False):
        self.use_cache = use_cache
        self.use_io_binding = use_io_binding
        logger.info('loading model')
        # [Method-1]
        # sess_options = onnxruntime.SessionOptions()
        # sess_options.intra_op_num_threads = 10
        # sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        # sess_options.inter_op_num_threads = 10
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # [Method-2]
        # cuda_provider_options = {
        #     "arena_extend_strategy": "kSameAsRequested",
        #     "do_copy_in_default_stream": False,
        #     "cudnn_conv_use_max_workspace": "1"
        # }
        # cpu_provider_options = {"arena_extend_strategy": "kSameAsRequested", "do_copy_in_default_stream": False}
        # execution_providers = [
        #     ("CUDAExecutionProvider", cuda_provider_options), ("CPUExecutionProvider", cpu_provider_options)]
        self.sess = InferenceSession(
            "/data1/marco.wu/models/Large_Language_Model/breeze_7b_instruct_v1_0_onnx/model.onnx",
            # sess_options=sess_options,
            # providers=execution_providers,
            # providers=[
            #     # (
            #     #     'CUDAExecutionProvider', {
            #     #         'device_id': 0,
            #     #         'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
            #     #         'arena_extend_strategy': "kNextPowerOfTwo",
            #     #         'cudnn_conv_algo_search': 'EXHAUSTIVE',
            #     #         'do_copy_in_default_stream': True
            #     #     }
            #     # ),
            #     # 'CUDAExecutionProvider',
            #     # 'CPUExecutionProvider',
            # ],
        )
        logger.info('loading config')
        self.config = AutoConfig.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")
        logger.info('loading ORT')
        self.model = ORTModelForCausalLM(self.sess, self.config, use_cache=True, use_io_binding=False)
        logger.info('loading tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-Instruct-v1_0")
        logger.info('loading CustomORT Runner')
        self.ort_runner = CustomORTModelRunner(
            self.tokenizer, self.sess, self.config, use_cache=True, use_io_binding=False)

    def run_onnx(self, text):
        logger.info('starting inference by ORTModel')
        inputs = self.tokenizer(text, return_tensors="pt")
        t0 = time.time()
        outputs = self.model.generate(**inputs)
        t1 = time.time()
        logger.info('----ORTModel Output----')
        logger.info(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        logger.info(f'Used Time: {t1 - t0}')

    def run_cus_onnx(self, text):
        logger.info('starting inference by CustomORTModel')
        t0 = time.time()
        output_texts = self.ort_runner.run(text)
        t1 = time.time()
        logger.info('----CustomORTModel Output----')
        logger.info(output_texts)
        logger.info(f'Used Time: {t1 - t0}')


def main():
    # Load
    onnx_runner = ONNXRunner(init=True, use_cache=True, use_io_binding=False)

    # Run
    onnx_runner.run_onnx("請問台灣最高的山是?\nOutput:")

    # Run ORTModel
    onnx_runner.run_cus_onnx("請問台灣最高的山是?\nOutput:")


if __name__ == "__main__":
    main()
