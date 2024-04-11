# LLM-TPU

source: https://github.com/sophgo/LLM-TPU/tree/main

```sh
# install
pip install transformers==4.36.0

# pre-setup
cp modeling_mistral.py {YOUR_ENV_LIBS_PATH}/transformers/models/mistral/modeling_mistral.py 

# export
python export_mistral.py

# run & check onnx
python mistral_with_past_only_onnx_flow.py
```