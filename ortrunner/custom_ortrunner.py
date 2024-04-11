import re
import torch
import onnxruntime
import numpy as np

from typing import Optional, Tuple, Union, Dict, List, Set

from ortrunner.transformers_tools.helper import TypeHelper, IOBindingHelper
from ortrunner.transformers_tools.modeling_outputs import CausalLMOutputWithPast
from ortrunner.transformers_tools.normalized_config import NormalizedConfigManager


class BreezeORTModelForCausalLM:
    def __init__(
        self,
        model: onnxruntime.InferenceSession,
        config,
        use_io_binding: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ):
        if use_io_binding is None:
            use_io_binding = model.get_providers()[0] in ["CPUExecutionProvider", "CUDAExecutionProvider"]

        # TODO: Marco
        super().__init__()
        self.normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        self.num_key_value_heads = (
            config.num_kv_heads if config.model_type != "mistral" and (
                config.new_decoder_architecture or not config.multi_query) else 1
        )
        self.config = config
        self.model = model
        self.inputs_names = {
            'input_ids': 0, 'attention_mask': 1, 'position_ids': 2, 'past_key_values.0.key': 3,
            'past_key_values.0.value': 4, 'past_key_values.1.key': 5, 'past_key_values.1.value': 6,
            'past_key_values.2.key': 7, 'past_key_values.2.value': 8, 'past_key_values.3.key': 9,
            'past_key_values.3.value': 10, 'past_key_values.4.key': 11, 'past_key_values.4.value': 12,
            'past_key_values.5.key': 13, 'past_key_values.5.value': 14, 'past_key_values.6.key': 15,
            'past_key_values.6.value': 16, 'past_key_values.7.key': 17, 'past_key_values.7.value': 18,
            'past_key_values.8.key': 19, 'past_key_values.8.value': 20, 'past_key_values.9.key': 21,
            'past_key_values.9.value': 22, 'past_key_values.10.key': 23, 'past_key_values.10.value': 24,
            'past_key_values.11.key': 25, 'past_key_values.11.value': 26, 'past_key_values.12.key': 27,
            'past_key_values.12.value': 28, 'past_key_values.13.key': 29, 'past_key_values.13.value': 30,
            'past_key_values.14.key': 31, 'past_key_values.14.value': 32, 'past_key_values.15.key': 33,
            'past_key_values.15.value': 34, 'past_key_values.16.key': 35, 'past_key_values.16.value': 36,
            'past_key_values.17.key': 37, 'past_key_values.17.value': 38, 'past_key_values.18.key': 39,
            'past_key_values.18.value': 40, 'past_key_values.19.key': 41, 'past_key_values.19.value': 42,
            'past_key_values.20.key': 43, 'past_key_values.20.value': 44, 'past_key_values.21.key': 45,
            'past_key_values.21.value': 46, 'past_key_values.22.key': 47, 'past_key_values.22.value': 48,
            'past_key_values.23.key': 49, 'past_key_values.23.value': 50, 'past_key_values.24.key': 51,
            'past_key_values.24.value': 52, 'past_key_values.25.key': 53, 'past_key_values.25.value': 54,
            'past_key_values.26.key': 55, 'past_key_values.26.value': 56, 'past_key_values.27.key': 57,
            'past_key_values.27.value': 58, 'past_key_values.28.key': 59, 'past_key_values.28.value': 60,
            'past_key_values.29.key': 61, 'past_key_values.29.value': 62, 'past_key_values.30.key': 63,
            'past_key_values.30.value': 64, 'past_key_values.31.key': 65, 'past_key_values.31.value': 66
        }
        self.output_names = {
            'logits': 0, 'present.0.key': 1, 'present.0.value': 2, 'present.1.key': 3, 'present.1.value': 4,
            'present.2.key': 5, 'present.2.value': 6, 'present.3.key': 7, 'present.3.value': 8, 'present.4.key': 9,
            'present.4.value': 10, 'present.5.key': 11, 'present.5.value': 12, 'present.6.key': 13,
            'present.6.value': 14, 'present.7.key': 15, 'present.7.value': 16, 'present.8.key': 17,
            'present.8.value': 18, 'present.9.key': 19, 'present.9.value': 20, 'present.10.key': 21,
            'present.10.value': 22, 'present.11.key': 23, 'present.11.value': 24, 'present.12.key': 25,
            'present.12.value': 26, 'present.13.key': 27, 'present.13.value': 28, 'present.14.key': 29,
            'present.14.value': 30, 'present.15.key': 31, 'present.15.value': 32, 'present.16.key': 33,
            'present.16.value': 34, 'present.17.key': 35, 'present.17.value': 36, 'present.18.key': 37,
            'present.18.value': 38, 'present.19.key': 39, 'present.19.value': 40, 'present.20.key': 41,
            'present.20.value': 42, 'present.21.key': 43, 'present.21.value': 44, 'present.22.key': 45,
            'present.22.value': 46, 'present.23.key': 47, 'present.23.value': 48, 'present.24.key': 49,
            'present.24.value': 50, 'present.25.key': 51, 'present.25.value': 52, 'present.26.key': 53,
            'present.26.value': 54, 'present.27.key': 55, 'present.27.value': 56, 'present.28.key': 57,
            'present.28.value': 58, 'present.29.key': 59, 'present.29.value': 60, 'present.30.key': 61,
            'present.30.value': 62, 'present.31.key': 63, 'present.31.value': 64
        }
        self._ordered_input_names = [
            'input_ids', 'attention_mask', 'position_ids', 'past_key_values.0.key', 'past_key_values.0.value',
            'past_key_values.1.key', 'past_key_values.1.value', 'past_key_values.2.key', 'past_key_values.2.value',
            'past_key_values.3.key', 'past_key_values.3.value', 'past_key_values.4.key', 'past_key_values.4.value',
            'past_key_values.5.key', 'past_key_values.5.value', 'past_key_values.6.key', 'past_key_values.6.value',
            'past_key_values.7.key', 'past_key_values.7.value', 'past_key_values.8.key', 'past_key_values.8.value',
            'past_key_values.9.key', 'past_key_values.9.value', 'past_key_values.10.key', 'past_key_values.10.value',
            'past_key_values.11.key', 'past_key_values.11.value', 'past_key_values.12.key', 'past_key_values.12.value',
            'past_key_values.13.key', 'past_key_values.13.value', 'past_key_values.14.key', 'past_key_values.14.value',
            'past_key_values.15.key', 'past_key_values.15.value', 'past_key_values.16.key', 'past_key_values.16.value',
            'past_key_values.17.key', 'past_key_values.17.value', 'past_key_values.18.key', 'past_key_values.18.value',
            'past_key_values.19.key', 'past_key_values.19.value', 'past_key_values.20.key', 'past_key_values.20.value',
            'past_key_values.21.key', 'past_key_values.21.value', 'past_key_values.22.key', 'past_key_values.22.value',
            'past_key_values.23.key', 'past_key_values.23.value', 'past_key_values.24.key', 'past_key_values.24.value',
            'past_key_values.25.key', 'past_key_values.25.value', 'past_key_values.26.key', 'past_key_values.26.value',
            'past_key_values.27.key', 'past_key_values.27.value', 'past_key_values.28.key', 'past_key_values.28.value',
            'past_key_values.29.key', 'past_key_values.29.value', 'past_key_values.30.key', 'past_key_values.30.value',
            'past_key_values.31.key', 'past_key_values.31.value'
        ]
        self.device = torch.device('cpu')
        self.use_io_binding = use_io_binding
        self.output_shape_inference_pattern = re.compile(r"([a-zA-Z_]+)|([0-9]+)|([+-/*])|([\(\)])")

        self.num_pkv = 2
        self.key_value_input_names = [key for key in self.inputs_names if (".key" in key) or (".value" in key)]
        self.key_value_output_names = [key for key in self.output_names if (".key" in key) or (".value" in key)]
        self.use_cache = len(self.key_value_input_names) > 0

        self.use_merged = "use_cache_branch" in self.inputs_names
        self.model_type = self.config.model_type

        self.use_fp16 = False
        for inp in model.get_inputs():
            if (
                inp.name == "past_key_values" or inp.name in self.key_value_input_names
            ) and inp.type == "tensor(float16)":
                self.use_fp16 = True
                break

        # Reference: https://github.com/huggingface/optimum/pull/1381
        # model_type = config.model_type.replace("_", "-")
        # if model_type in MODEL_TYPES_REQUIRING_POSITION_IDS and "position_ids" not in self.inputs_names:
        #     msg = "ORTModelForCausalLM loaded a legacy ONNX model with no position_ids input, although this input" +\
        #             f" is required for batched generation for the architecture {model_type}. We strongly encourage" +\
        #             "t o re-export the model with optimum>=1.14 for position_ids and batched inference support."
        #     print(msg)

        if use_cache ^ self.use_cache:
            raise ValueError(
                f"`use_cache` was set to `{use_cache}` but the loaded model only supports `use_cache={self.use_cache}`."
                f"Please load your current model with `use_cache={self.use_cache}` or export the original model "
                f"once again with `use_cache={use_cache}` when calling the `from_pretrained` method. "
                "To export your model, simply set `export=True`."
            )

        if use_io_binding and not use_cache:
            raise ValueError(
                "The parameters combination use_cache=False, use_io_binding=True is not supported. "
                "Please either pass use_cache=True, use_io_binding=True (default), or use_cache=False, "
                "use_io_binding=False."
            )

    def _prepare_output_buffer(self, model: onnxruntime.InferenceSession, output_shape: Tuple[int], output_name: str):
        """Prepares the buffer of output_name with a 1D tensor."""
        ort_type = TypeHelper.get_output_type(model, output_name)
        torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
        if len(output_shape) > 0:
            output_buffer = torch.empty(np.prod(output_shape), dtype=torch_type, device=self.device).contiguous()
        else:
            # Case when the output is a scalar
            output_buffer = torch.tensor(0, dtype=torch_type, device=self.device).contiguous()
        return output_buffer

    def _output_shape_inference(self, axis_name: Union[str, int], dimensions: Dict[str, int]) -> Union[str, int]:
        """
        Infers the output shape of a given dynamic axis by using the `dimensions` mapping.

        For instance, for the following inputs:
            axis_name = "past_sequence_length + sequence_length"
            dimensions = {"batch_size": 2, "sequence_length": 3, "past_sequence_length": 7}

        The inferred shape is 3 + 7 = 10.
        """
        if isinstance(axis_name, int):
            return axis_name
        # It is actually covered below, but this is to make things faster.
        elif axis_name in dimensions:
            return dimensions[axis_name]

        # Tokens is going to be populated by iterating over every match for the self.output_shape_inference_pattern.
        # This pattern matches 4 things: axis names, integer values, operators (+, -, *, /) and parenthesis.
        tokens = []
        for idx, match_ in enumerate(re.finditer(self.output_shape_inference_pattern, axis_name)):
            groups = match_.groups()
            matched_group = None
            for idx, group in enumerate(groups):
                if group is not None:
                    matched_group = idx
                    break

            # For every match except an axis name, we simply append the content of the match to the tokens list.
            # For an axis name, we check if it is specified in the `dimensions` dictionary. If for some reason it is
            # not there, or its value not an integer, the shape inference process stops and we return the axis name as
            # is.
            if matched_group == 0:
                dim = dimensions.get(groups[0], None)
                if dim is None or not isinstance(dim, int):
                    return axis_name
                tokens.append(str(dim))
            else:
                tokens.append(groups[matched_group])

        # Here it should not be problematic to use eval since anything not matching the pattern would trigger an
        # exception.
        return int(eval(" ".join(tokens)))

    def _prepare_io_binding(
        self,
        model: onnxruntime.InferenceSession,
        *model_inputs: torch.Tensor,
        ordered_input_names: List[str],
        known_output_shapes: Optional[Dict[str, Tuple[int]]] = None,
        outputs_to_not_bind: Optional[Union[Set[str], str]] = None,
    ) -> Tuple[onnxruntime.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]]:
        """
        Prepares IO binding for ONNX Runtime.

        Args:
            model (`onnxruntime.InferenceSession`):
                The model for which we want to bind the inputs and outputs.
            *model_inputs:
                The inputs of the model.
            ordered_input_names (`List[str]`):
                Names of the inputs, that must match with the order of model_inputs.
            known_output_shapes (`Optional[Dict[str, Tuple[int]]]`, defaults to `None`):
                It can be hard to infer all the output shapes from the inputs only. For instance for the past key /
                values. It is possible to explicitely pass the shape via this argument.
            outputs_to_not_bind (`Optional[Union[Set[str], str]]`, defaults to `None`):
                The names of the outputs that should not be bound.

        Returns:
            `Tuple[onnxruntime.IOBinding, Dict[str, Tuple[int]], Dict[str, torch.Tensor]`: The IOBinding object, a 
                dictionary
            containing the shape of each output, and another one pointing to the buffers containing the outputs data.
        """
        io_binding = model.io_binding()

        name_to_np_type = TypeHelper.get_io_numpy_type_map(model)

        input_name_to_shape = {}
        for idx, tensor in enumerate(model_inputs):
            if tensor is None:
                continue
            name = ordered_input_names[idx]
            tensor = tensor.contiguous()
            input_name_to_shape[name] = tensor.shape

            data_ptr = tensor.data_ptr()
            if "past" in name and data_ptr == 0:
                # During first generation, sequence_length can be 0 when use_cache=True, which results in data_ptr to
                #   also be 0.
                # To keep compatibility with IO binding, we pass the data pointer of input_ids instead. This will have
                #   no impact because past_key_values will not be used during the first generation.
                data_ptr = model_inputs[0].data_ptr()

            io_binding.bind_input(
                name,
                tensor.device.type,
                IOBindingHelper.get_device_index(self.device),
                name_to_np_type[name],
                tuple(tensor.shape),
                data_ptr,
            )
        dimensions = {}
        for input_ in model.get_inputs():
            shape = input_.shape
            for idx, axis in enumerate(shape):
                if isinstance(axis, str):
                    dimensions[axis] = input_name_to_shape[input_.name][idx]

        output_shapes = {}
        output_buffers = {}

        if known_output_shapes is None:
            known_output_shapes = {}

        if outputs_to_not_bind is None:
            outputs_to_not_bind = set()
        elif isinstance(outputs_to_not_bind, str):
            outputs_to_not_bind = {outputs_to_not_bind}

        for output_node in model.get_outputs():
            output_name = output_node.name
            if output_name in outputs_to_not_bind:
                continue
            if output_name in known_output_shapes:
                output_shape = known_output_shapes[output_name]
            else:
                output_shape = []
                for axis_name in output_node.shape:
                    output_shape.append(self._output_shape_inference(axis_name, dimensions))
            output_buffer = self._prepare_output_buffer(model, output_shape, output_name)

            io_binding.bind_output(
                output_name,
                output_buffer.device.type,
                IOBindingHelper.get_device_index(self.device),
                name_to_np_type[output_name],
                output_shape,
                output_buffer.data_ptr(),
            )
            output_shapes[output_name] = output_shape
            output_buffers[output_name] = output_buffer

        return io_binding, output_shapes, output_buffers

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache_branch: bool = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # adding use_cache_branch in the signature here is just a hack for IO Binding
        use_torch = isinstance(input_ids, torch.Tensor)
        # TODO: marco self.raise_on_numpy_input_io_binding(use_torch)

        inputs = {}
        known_output_shapes = {}
        use_cache_branch = None
        loss = None
        if self.use_cache:
            if past_key_values is not None:
                # Flatten the past_key_values (gpt_bigcode has fused key/value cache, so no need to flatten it)
                # (1) size of past_key_values: 32, 2, 1 --> 1, 8, 12, 128 (O)
                # (2) size of past_key_values: 32, 2, 1 --> 1, 8, 13, 128 (O)
                if self.model_type != "gpt_bigcode":
                    past_key_values = tuple(
                        past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                    )

            # Create dummy past_key_values for decoder first generation step if none given
            use_cache_branch, past_key_values, known_output_shapes = self.prepare_past_key_values(
                input_ids, past_key_values, use_torch
            )

        if self.use_io_binding:
            # TODO: fix transformers generate to have contiguous input_ids here already
            # For an unknown reason, calling `contiguous()` here is necessary to not have errors
            # on CPU EP with batch size > 1, despite it being also called in _prepare_io_binding.
            # I suspect the reason is the contiguous python list that messes something up?
            model_inputs = [input_ids.contiguous()]

            if "attention_mask" in self.inputs_names:
                model_inputs.append(attention_mask)

            if "position_ids" in self.inputs_names:
                if position_ids is None:
                    raise ValueError("position_ids was not passed but is a required input for this ONNX model.")
                model_inputs.append(position_ids.contiguous())

            if past_key_values is not None:
                model_inputs += past_key_values

            if use_cache_branch is not None:
                model_inputs.append(use_cache_branch)

            if "labels" in self.inputs_names:
                model_inputs.append(labels)
                known_output_shapes.update({"loss": []})

            io_binding, output_shapes, output_buffers = self._prepare_io_binding(
                self.model,
                *model_inputs,
                known_output_shapes=known_output_shapes,
                ordered_input_names=self._ordered_input_names,
            )

            if self.device.type == "cpu":
                self.model.run_with_iobinding(io_binding)
            else:
                io_binding.synchronize_inputs()
                self.model.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer(2)
                past_key_values = ()
                for name in self.key_value_output_names:
                    past_key_values += (output_buffers[name].view(output_shapes[name]),)

            logits = output_buffers["logits"].view(output_shapes["logits"])

            if "loss" in self.output_names:
                loss = output_buffers["loss"].view(output_shapes["loss"])
        else:
            inputs["input_ids"] = input_ids.cpu().detach().numpy() if use_torch else input_ids

            if "attention_mask" in self.inputs_names:
                inputs["attention_mask"] = attention_mask.cpu().detach().numpy() if use_torch else attention_mask

            if "labels" in self.inputs_names:
                inputs["labels"] = labels.cpu().detach().numpy() if use_torch else labels

            if "position_ids" in self.inputs_names:
                if position_ids is None:
                    raise ValueError("position_ids was not passed but is a required input for this ONNX model.")
                inputs["position_ids"] = position_ids.cpu().detach().numpy() if use_torch else position_ids

            # Add the past_key_values to the decoder inputs
            if past_key_values is not None:
                for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                    inputs[input_name] = past_key_value.cpu().detach().numpy() if use_torch else past_key_value

            if use_cache_branch is not None:
                inputs["use_cache_branch"] = use_cache_branch.cpu().detach().numpy() if use_torch else use_cache_branch

            # (2) input_ids==28705, attention_mask==[1, ...](13個), position_ids==[[12]] (O)
            # (3) input_ids==48740, attention_mask==[1, ...](14個), position_ids==[[13]] (O)
            outputs = self.model.run(None, inputs)

            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (
                #   2 for the self-attention)
                past_key_values = tuple(
                    torch.from_numpy(outputs[self.output_names[key]]).to(self.device)
                    for key in self.key_value_output_names
                )

            logits = torch.from_numpy(outputs[self.output_names["logits"]]).to(self.device)
            if "loss" in self.output_names:
                loss = torch.from_numpy(outputs[self.output_names["loss"]]).to(self.device)

        if self.use_cache and self.model_type != "gpt_bigcode":
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to the number of self-attention and
            # per decoder layer
            past_key_values = tuple(
                past_key_values[i: i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values)

    def prepare_past_key_values(
        self,
        input_ids: Union[None, torch.LongTensor, np.ndarray],
        past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]],
        use_torch: bool,
    ):
        sequence_length = input_ids.shape[1]

        constructor = torch if use_torch else np
        if self.use_merged:
            # Uses without/with branch of a merged decoder depending on whether real past key values are passed
            use_cache_branch = constructor.full((1,), past_key_values is not None)
        else:
            # Uses separate decoders
            use_cache_branch = None

        if use_torch and use_cache_branch is not None:
            use_cache_branch = use_cache_branch.to(self.device)

        pkv_output_shape = {}
        # Generate dummy past for the first forward if uses a merged decoder
        if past_key_values is None:
            batch_size = input_ids.shape[0]
            embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
            if self.model_type == "gemma":
                num_attention_heads = self.normalized_config.num_key_value_heads
                embed_size_per_head = self.normalized_config.head_dim
            elif self.model_type in {"mistral", "llama", "qwen2"}:
                num_attention_heads = self.normalized_config.num_key_value_heads
            else:
                num_attention_heads = self.normalized_config.num_attention_heads

            dtype = constructor.float16 if self.use_fp16 else constructor.float32

            # TODO: find a way to better handle this controlflow, this is EXTREMELY UGLY.
            # "1" is the dummy sequence length
            if self.model_type == "bloom":
                shape_value = (batch_size * num_attention_heads, 0, embed_size_per_head)
                shape_key = (batch_size * num_attention_heads, embed_size_per_head, 0)
                key = constructor.zeros(shape_key, dtype=dtype)
                value = constructor.zeros(shape_value, dtype=dtype)

                if use_torch:
                    key = key.to(self.device)
                    value = value.to(self.device)

                past_key_values = tuple(
                    key_or_value for _ in range(len(self.key_value_input_names) // 2) for key_or_value in [key, value]
                )

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    index = 1 if "value" in name else 2

                    shape[index] += sequence_length
                    pkv_output_shape[name] = shape
            elif self.model_type == "gpt_bigcode":
                # GPT BigCode uses muti-query attention, and has the specificity of putting both key and value in the 
                # same cache tensor.
                shape_key_and_value = (batch_size, 0, embed_size_per_head * 2)
                key_and_value = constructor.zeros(shape_key_and_value, dtype=dtype)

                if use_torch:
                    key_and_value = key_and_value.to(self.device)

                past_key_values = tuple(key_and_value for _ in range(len(self.key_value_input_names)))

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    shape[1] += sequence_length
                    pkv_output_shape[name] = shape
            else:
                num_key_value_heads = self.num_key_value_heads if self.model_type == "falcon" else num_attention_heads

                shape = (batch_size, num_key_value_heads, 0, embed_size_per_head)
                key_or_value = constructor.zeros(shape, dtype=dtype)

                if use_torch:
                    key_or_value = key_or_value.to(self.device)

                past_key_values = tuple(key_or_value for _ in range(len(self.key_value_input_names)))

                for name, value in zip(self.key_value_output_names, past_key_values):
                    shape = [*value.shape]
                    shape[2] += sequence_length
                    pkv_output_shape[name] = shape

        return use_cache_branch, past_key_values, pkv_output_shape


class CustomORTModelRunner:
    MAX_NEW_TOKENS = 2048

    def __init__(self, tokenizer, sess, config, use_cache=True, use_io_binding=False, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.end_token_id = tokenizer.eos_token_id
        self.ort_model = BreezeORTModelForCausalLM(sess, config, use_cache=use_cache, use_io_binding=use_io_binding)
        self.return_tensors = return_tensors

    @staticmethod
    def create_position_ids(input_ids):
        sequence_length = input_ids.size(1)
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids

    @staticmethod
    def sample_top_p(logits, p=0.95, temperature=0.8):
        # top-p sample: top-k采样是从前k个概率最大的token中采样，top-p采样是从token中构造一个最小候选集，其概率和大于p，然后从该候选集中采样
        # 当T(temperature)大的时候，概率分布趋向平均，随机性增大；当T小的时候，概率密度趋向于集中，即强者愈强，随机性降低
        probs = torch.softmax(logits[0] / temperature, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    @staticmethod
    def greedy_sampling(logits):
        return torch.argmax(logits, dim=-1)

    @staticmethod
    def random_sampling(logits):
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @staticmethod
    def temperature_sampling(logits, temperature=1.0):
        """
        When T(temperature) is large, the probability distribution tends towards uniformity, increasing randomness;
          when T is small, the probability density tends to concentrate, meaning the stronger tokens become even
            stronger, and randomness decreases.
        """
        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)

    @staticmethod
    def top_k_sampling(logits, k=5):
        """
        Top-k sampling selects from the top k tokens with the highest probabilities.
        """
        values, indices = torch.topk(logits, k=k, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return torch.gather(indices, dim=-1, index=sampled_index)

    @staticmethod
    def top_p_sampling(logits, p=0.9):
        """
        Top-p sampling constructs a minimal candidate set of tokens where the cumulative probability exceeds p,
          and then samples from this candidate set.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs > p
        sorted_indices[mask] = -1
        sampled_index = torch.multinomial((sorted_indices != -1).float(), num_samples=1)
        return torch.gather(sorted_indices, dim=-1, index=sampled_index)

    @staticmethod
    def postprocess(logits, sample_method_name='greedy_sampling'):
        sampling_method = getattr(CustomORTModelRunner, sample_method_name)
        next_token = sampling_method(logits)
        next_token = next_token.reshape(-1).tolist()[-1:]
        return next_token

    def inference(self, inputs, past_key_values=None):
        # (O) input_ids = [tensor([[    1, 28705, 44639, 42168, 42750, 28914, 29480, 28971, 28804,    13, 4792, 28747]])
        # (O) attention_mask = tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        # (O) position_ids = tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]]) ... 67
        # () known_output_shapes = {present.0.key, ...,  present.31.key: [1, 8, 12, 128]}
        outputs = self.ort_model.forward(
            inputs["input_ids"],
            inputs.get("attention_mask"),
            inputs.get("position_ids", CustomORTModelRunner.create_position_ids(inputs["input_ids"])),
            past_key_values=past_key_values,
            labels=None,
            use_cache_branch=None
        )
        next_token_id = CustomORTModelRunner.postprocess(outputs.logits)
        # output_text = self.tokenizer.decode(outputs.logits.argmax(2)[0], skip_special_tokens=True)[
        #     len(inputs["input_ids"][0]):]

        # InferenceSession
        # print('starting inference by InferenceSession')
        # outputs = sess.run(None, {
        #     "input_ids": inputs["input_ids"].cpu().numpy(),
        #     "attention_mask": inputs["attention_mask"].cpu().numpy() if "attention_mask" in inputs else None,
        #     "position_ids": inputs["position_ids"].cpu().numpy() if "position_ids" in inputs else None,
        #     "past_key_values": None,  # Update this if past_key_values are used
        #     "use_cache_branch": None  # Update this if use_cache_branch is used
        # })
        # (O) 1. argmax > array([  415, 28750, 42424, 28914, 43802, 29480, 28971, 48740,    13, 13, 28747, 28705])
        # () 2. 28705 > 1, 1, ... > 12
        # print(tokenizer.decode(outputs[0][0], skip_special_tokens=True))
        return next_token_id, outputs

    def run(self, input_text):
        # Encode
        encode_inputs = self.tokenizer(input_text, return_tensors=self.return_tensors)

        # Inference
        next_token_id, outputs = self.inference(encode_inputs)
        token_ids = encode_inputs["input_ids"].reshape(-1).tolist() + next_token_id
        # print(self.tokenizer._convert_id_to_token(token_ids[-1]))

        # Generate the next token iteratively using llama_with_past model
        gene_count = 2
        while token_ids[-1] != self.end_token_id and gene_count <= CustomORTModelRunner.MAX_NEW_TOKENS:
            next_token_id, outputs = self.inference(
                inputs={
                    "input_ids": torch.tensor([next_token_id]),
                    "attention_mask": torch.tensor([[1] * len(token_ids)]),
                    "position_ids": torch.tensor([[len(token_ids) - 1]]),
                },
                past_key_values=outputs.past_key_values
            )
            token_ids += next_token_id
            # print(self.tokenizer._convert_id_to_token(token_ids[-1]))

        # Decode
        output_text = self.tokenizer.decode(token_ids)
        return output_text
