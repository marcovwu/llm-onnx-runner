import numpy as np
import onnxruntime as ort

from transformers import LlamaTokenizer


class MistralRunner:
    SEQ_LENGTH = 2048
    HIDDEN_SIZE = 4096
    NUM_ATTENTION_HEADS = 32
    HIDDEN_DIM = 128  # HIDDEN_SIZE // NUM_ATTENTION_HEADS
    NUM_LAYERS = 32
    NUM_KEY_VALUE_HEADS = 8
    VOCAB_SIZE = 61952
    PROVIDERS = ['CPUExecutionProvider']  # CUDAExecutionProvider, CPUExecutionProvider

    def __init__(
        self,
        onnx_path='/alghome/marco.wu/models/Large_Language_Model/breeze_7b_instruct_v1_0_onnx_fixed/tmp/onnx/',
        tokenizer_path='MediaTek-Research/Breeze-7B-Instruct-v1_0',
        sample_method_name='greedy_head'  # greedy_head, penalty_sample_head
    ):
        # load tokenizer
        print('load tokenizer')
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.end_token_id = self.tokenizer.eos_token_id

        # load embedding
        print('load embedding')
        self.embedding_sess = ort.InferenceSession(onnx_path + 'embedding.onnx')
        input_name = [self.embedding_sess.get_inputs()[i].name for i in range(len(self.embedding_sess.get_inputs()))]
        print("embedding_input_name:{}".format(input_name))

        # load block
        # self.block_sess_list = []
        # for i in range(MistralRunner.NUM_LAYERS):
        #     print(f'load block_{i}')
        #     self.block_sess_list.append(
        #         ort.InferenceSession(onnx_path + 'block_' + str(i) + '.onnx', providers=MistralRunner.PROVIDERS))
        # input_name = [
        #     self.block_sess_list[0].get_inputs()[i].name for i in range(len(self.block_sess_list[0].get_inputs()))]
        # print("block_input_name:{}".format(input_name))

        # load block_cache
        self.block_cache_sess_list = []
        for i in range(MistralRunner.NUM_LAYERS):
            print(f'load block_cache_{i}')
            self.block_cache_sess_list.append(
                ort.InferenceSession(onnx_path + 'block_cache_' + str(i) + '.onnx', providers=MistralRunner.PROVIDERS))
        input_name = [self.block_cache_sess_list[0].get_inputs()[i].name for i in range(
            len(self.block_cache_sess_list[0].get_inputs()))]
        print("block_input_name:{}".format(input_name))

        # load lm_head
        print('load lm_head')
        self.lm_head_sess = ort.InferenceSession(onnx_path + 'lm_head.onnx')
        input_name = [self.lm_head_sess.get_inputs()[i].name for i in range(len(self.lm_head_sess.get_inputs()))]
        print("lm_head_input_name:{}".format(input_name))

        # load sample head
        print(f'load {sample_method_name}')
        self.sample_method_name = sample_method_name
        self.sample_head_sess = ort.InferenceSession(onnx_path + f'{sample_method_name}.onnx')
        input_name = [
            self.sample_head_sess.get_inputs()[i].name for i in range(len(self.sample_head_sess.get_inputs()))]
        print("lm_head_input_name:{}".format(input_name))

        # init parameters
        self.k_cache_list = np.zeros(
            (
                MistralRunner.NUM_LAYERS,
                1,
                MistralRunner.SEQ_LENGTH,
                MistralRunner.NUM_KEY_VALUE_HEADS,
                MistralRunner.HIDDEN_DIM
            )
        ).astype(np.float32)
        self.v_cache_list = np.zeros(
            (
                MistralRunner.NUM_LAYERS,
                1,
                MistralRunner.SEQ_LENGTH,
                MistralRunner.NUM_KEY_VALUE_HEADS,
                MistralRunner.HIDDEN_DIM
            )
        ).astype(np.float32)

    def check_token_length(self, token_ids_length, max_generate_len):
        if token_ids_length >= MistralRunner.SEQ_LENGTH:
            raise ValueError(
                f'Prompt length {token_ids_length} is larger than maximal model input length {MistralRunner.SEQ_LENGTH}'
            )
        if max_generate_len + token_ids_length >= MistralRunner.SEQ_LENGTH:
            print(
                'Warning: Generate length is too large, will be truncated to '
                f'{MistralRunner.SEQ_LENGTH - token_ids_length}'
            )
            max_generate_len = MistralRunner.SEQ_LENGTH - token_ids_length
        return max_generate_len

    def _init_input_to_kv_list(self, i, token_len, present_k_cache, present_v_cache):
        present_k_cache[:, MistralRunner.SEQ_LENGTH - token_len:] = present_k_cache[:, :token_len]
        present_v_cache[:, MistralRunner.SEQ_LENGTH - token_len:] = present_v_cache[:, :token_len]
        present_k_cache[:, :MistralRunner.SEQ_LENGTH - token_len] = 0
        present_v_cache[:, :MistralRunner.SEQ_LENGTH - token_len] = 0
        self.k_cache_list[i] = present_k_cache
        self.v_cache_list[i] = present_v_cache

    def _update_new_to_kv_list(self, i, token_len, present_k_cache, present_v_cache, block_onnx_inputs):
        new_k = np.zeros(block_onnx_inputs["history_k"].shape).astype(np.float32)
        new_v = np.zeros(block_onnx_inputs["history_v"].shape).astype(np.float32)
        new_k[:, MistralRunner.SEQ_LENGTH - token_len: MistralRunner.SEQ_LENGTH - 1] = block_onnx_inputs[
            "history_k"][:, MistralRunner.SEQ_LENGTH - token_len + 1:]
        # token_len=2时，相当于把history_k有效的最后2个放到new_k倒数3,2位置
        new_v[:, MistralRunner.SEQ_LENGTH - token_len: MistralRunner.SEQ_LENGTH - 1] = block_onnx_inputs[
            "history_v"][:, MistralRunner.SEQ_LENGTH - token_len + 1:]
        new_k[:, MistralRunner.SEQ_LENGTH - 1:] = present_k_cache[:, -1:]  # [1, 1, NUM_KEY_VALUE_HEADS, HIDDEN_SIZE
        # 新生成的第2049放到new_k倒数1
        new_v[:, MistralRunner.SEQ_LENGTH - 1:] = present_v_cache[:, -1:]
        self.k_cache_list[i] = new_k
        self.v_cache_list[i] = new_v

    def _inference_input_kv(self, input_states, token_len):
        # position ids
        position_ids = list(range(token_len)) + (MistralRunner.SEQ_LENGTH - token_len) * [0]
        position_ids = np.array([position_ids], dtype=np.int64)
        # attention mask
        attention_mask = -1000 * np.ones((1, 1, MistralRunner.SEQ_LENGTH, MistralRunner.SEQ_LENGTH)).astype(np.float32)
        for i in range(token_len):
            for j in range(token_len):
                if j <= i:
                    attention_mask[:, :, i, j] = 0

        for i in range(MistralRunner.NUM_LAYERS):
            block_onnx_inputs = {
                "input_states": input_states,  # [1, SEQ_LENGTH, HIDDEN_SIZE]
                "position_ids": position_ids,  # [1, SEQ_LENGTH]
                "attention_mask": attention_mask,  # [1, 1, SEQ_LENGTH, SEQ_LENGTH]
            }

            # present_k/v_cache: (1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HIDDEN_DIM)
            hidden_state, present_k_cache, present_v_cache = self.block_sess_list[i].run(None, block_onnx_inputs)

            # update to k/v_cache_list
            self._init_input_to_kv_list(i, token_len, present_k_cache, present_v_cache)

            # update input_states for next layer
            input_states = hidden_state

        return input_states

    def _inference_input(self, input_states, token_len, mode='update'):
        """
        :param input_states: (list) size: [1, 1, HIDDEN_SIZE]
        "param token_len: (int) current token index + 1
        :param mode: (str) used to process kv_list. choices: 'init' or 'update'
        """

        attention_mask = -1000 * np.ones((1, 1, 1, MistralRunner.SEQ_LENGTH + 1)).astype(np.float32)
        attention_mask[:, :, :, MistralRunner.SEQ_LENGTH + 1 - token_len:] = 0  # [[[[-1000, ..., -1000, 0]]]]
        # attention_mask = np.zeros((1, 1, 1, MistralRunner.SEQ_LENGTH + 1)).astype(np.float32)
        # attention_mask[:, :, :, MistralRunner.SEQ_LENGTH + 1 - token_len:] = 1

        for i in range(MistralRunner.NUM_LAYERS):
            block_onnx_inputs = {
                "input_states": input_states,  # [1, 1, HIDDEN_SIZE]
                "position_ids": [[(token_len - 1)]],  # [1, 1]
                "attention_mask": attention_mask,  # [1, 1, 1, SEQ_LENGTH + 1]
                "history_k": self.k_cache_list[i],  # [1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HIDDEN_DIM]
                "history_v": self.v_cache_list[i],  # [1, SEQ_LENGTH, NUM_KEY_VALUE_HEADS, HIDDEN_DIM]
            }

            # hidden_state: (1, 1, SEQ_LENGTH), present_k/v_cache: (1, SEQ_LENGTH + 1, NUM_KEY_VALUE_HEADS, HIDDEN_DIM)
            hidden_state, present_k_cache, present_v_cache = self.block_cache_sess_list[i].run(None, block_onnx_inputs)

            # update to k/v_cache_list
            if mode == 'init':
                self._init_input_to_kv_list(i, token_len, present_k_cache[
                    :, :MistralRunner.SEQ_LENGTH], present_v_cache[:, :MistralRunner.SEQ_LENGTH])
            else:
                self._update_new_to_kv_list(i, token_len, present_k_cache, present_v_cache, block_onnx_inputs)

            # update input_states for next layer
            input_states = hidden_state

        return hidden_state

    def _generate_token(self, hidden_state, input_ids):
        """
        :param hidden_state: (list) [1, 1, HIDDEN_SIZE]
        :param input_ids: (list) [SEQ_LENGTH]

        :return generated_token: (int) token id
        """
        lm_head_result = self.lm_head_sess.run(None, {"hidden_states": hidden_state[0]})  # [array(1, VOCAB_SIZE)]
        if self.sample_method_name == 'greedy_head':
            probs, generated_token = None, self.sample_head_sess.run(None, {"m_logits": lm_head_result[0]})
            generated_token = int(generated_token[0].item())
        elif self.sample_method_name == 'penalty_sample_head':
            probs, generated_token = self.sample_head_sess.run(
                None,
                {
                    "m_logits": lm_head_result[0],  # [1, VOCAB_SIZE]
                    "input_ids": np.array([input_ids]),  # [1, SEQ_LENGTH]
                    "top_p": np.array([0.8], dtype=np.float32),
                    "temperature": np.array([0.98], dtype=np.float32),
                    "penalty": np.array([0.98], dtype=np.float32)
                }
            )
        return probs, generated_token

    def run(self, prompt, max_generate_len=10, show=False):
        # Tokenization
        token_ids = self.tokenizer.encode(prompt)
        token_ids_length = len(token_ids)
        if show:
            print("input ids:{}".format(token_ids))  # TODO: only for show
            print("input ids len:{}".format(token_ids_length))  # TODO: only for show

        # Check
        max_generate_len = self.check_token_length(token_ids_length, max_generate_len)

        # Generate past keys and values of input prompt
        input_ids = token_ids + [0] * (MistralRunner.SEQ_LENGTH - token_ids_length)
        embedding_result = self.embedding_sess.run(None, {"input_ids": [input_ids]})  # [1, SEQ_LENGTH, HIDDEN_SIZE]
        # [block method]
        # hidden_state = self._inference_input_kv(input_states=embedding_result[0], token_len=token_ids_length)
        # [block cache method]
        for token_inx in range(token_ids_length - 1):
            _ = self._inference_input(
                input_states=[[embedding_result[0][0][token_inx]]], token_len=token_inx + 1, mode="update"
            )

        # Start generations from last token_ids
        gene_count = 0
        while token_ids[-1] != self.end_token_id and gene_count <= max_generate_len:
            # obtain current token index
            current_token_inx = gene_count + token_ids_length - 1
            input_ids[current_token_inx] = token_ids[-1]
            embedding_result = self.embedding_sess.run(None, {"input_ids": [input_ids]})  # [1, SEQ_LENGTH, HIDDEN_SIZE]

            # obtain hidden state
            hidden_state = self._inference_input(
                input_states=[[embedding_result[0][0][current_token_inx]]], token_len=current_token_inx + 1
            )

            # update to outputs
            _, generated_token = self._generate_token(hidden_state, input_ids)
            token_ids.append(generated_token)
            if show:
                print(self.tokenizer._convert_id_to_token(generated_token))  # TODO: only for show
            gene_count += 1

        # Decode the whole token list
        predict = self.tokenizer.decode(token_ids)
        return predict


if __name__ == "__main__":
    mistral_runner = MistralRunner()
    predict = mistral_runner.run("請問台灣最高的山是?\nOutput:", max_generate_len=10, show=True)
    print("\nPredict:\n{}".format(predict))
    import pdb
    pdb.set_trace()
