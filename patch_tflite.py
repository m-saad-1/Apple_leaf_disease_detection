import os

filepath = r"D:\WEB DEVELOPMENT\Apple_leaf_detection\venv\Lib\site-packages\tensorflow\lite\python\interpreter.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# We need to find the `CreateWrapperFromFile` and `CreateWrapperFromBuffer` calls inside `__init__` and patch them.
# The `__init__` method has:
# self._interpreter = _interpreter_wrapper.CreateWrapperFromFile(...)
# Let's replace it.

old_code_file = """      self._interpreter = _interpreter_wrapper.CreateWrapperFromFile(
          os.fspath(model_path),
          op_resolver_id,
          custom_op_registerers_by_name,
          custom_op_registerers_by_func,
          experimental_preserve_all_tensors,
          experimental_disable_delegate_clustering,
          int(num_threads or 1),
          experimental_default_delegate_latest_features,
      )"""

new_code_file = """      try:
        self._interpreter = _interpreter_wrapper.CreateWrapperFromFile(
            os.fspath(model_path),
            op_resolver_id,
            custom_op_registerers_by_name,
            custom_op_registerers_by_func,
            experimental_preserve_all_tensors,
            experimental_disable_delegate_clustering,
            int(num_threads or 1),
            experimental_default_delegate_latest_features,
        )
      except TypeError:
        # Fallback to 6 args for older C++ wrapper
        self._interpreter = _interpreter_wrapper.CreateWrapperFromFile(
            os.fspath(model_path),
            op_resolver_id,
            custom_op_registerers_by_name,
            custom_op_registerers_by_func,
            experimental_preserve_all_tensors,
            experimental_disable_delegate_clustering,
        )"""

old_code_buffer = """      self._interpreter = _interpreter_wrapper.CreateWrapperFromBuffer(
          model_content,
          op_resolver_id,
          custom_op_registerers_by_name,
          custom_op_registerers_by_func,
          experimental_preserve_all_tensors,
          experimental_disable_delegate_clustering,
          int(num_threads or 1),
          experimental_default_delegate_latest_features,
      )"""

new_code_buffer = """      try:
        self._interpreter = _interpreter_wrapper.CreateWrapperFromBuffer(
            model_content,
            op_resolver_id,
            custom_op_registerers_by_name,
            custom_op_registerers_by_func,
            experimental_preserve_all_tensors,
            experimental_disable_delegate_clustering,
            int(num_threads or 1),
            experimental_default_delegate_latest_features,
        )
      except TypeError:
        # Fallback to 6 args
        self._interpreter = _interpreter_wrapper.CreateWrapperFromBuffer(
            model_content,
            op_resolver_id,
            custom_op_registerers_by_name,
            custom_op_registerers_by_func,
            experimental_preserve_all_tensors,
            experimental_disable_delegate_clustering,
        )"""

content = content.replace(old_code_file, new_code_file)
content = content.replace(old_code_buffer, new_code_buffer)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Interpreter.py monkeypatched!")
