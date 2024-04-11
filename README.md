# Porting with Optimum

```sh
# install
pip install .

# export
optimum-cli export onnx -m {HUGGINGFACE_MODEL_PATH}--batch_size 1 --sequence_length SEQUENCE_LENGTH {YOUR_OUTPUT_FOLDER}

# run & check onnx
python ortrunner/runner.py
```