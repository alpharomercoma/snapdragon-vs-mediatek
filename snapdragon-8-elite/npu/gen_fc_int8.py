# gen_fc_int8.py — build the NPU matmul benchmark model: 8 chained
# fully-connected layers, int8 QDQ quantized, [32,2048]x[2048,2048] each.
# Mirrors npu_matmul_bench.c (MediaTek MDLA) from the POCO benchmark:
#   M=32, K=N=2048, LAYERS=8, int8 weights, quantized activations.
# The QDQ pattern lets ONNX Runtime's QNN EP lower every layer to the
# Hexagon HTP as a quantized MatMul.
import numpy as np
import onnx
from onnx import TensorProto, helper

M, K, N, LAYERS = 32, 2048, 2048, 8

nodes = []
inits = []

# quant params: activations u8, weights s8, per-tensor
act_scale = np.float32(0.02)
w_scale = np.float32(0.05)

inits.append(helper.make_tensor("act_scale", TensorProto.FLOAT, [], [act_scale]))
inits.append(helper.make_tensor("act_zp", TensorProto.UINT8, [], [128]))
inits.append(helper.make_tensor("w_scale", TensorProto.FLOAT, [], [w_scale]))
inits.append(helper.make_tensor("w_zp", TensorProto.INT8, [], [0]))

rng = np.random.default_rng(1234)
prev = "input_q"
for l in range(LAYERS):
    w = rng.integers(-4, 4, size=(K, N), dtype=np.int8)
    inits.append(helper.make_tensor(f"w{l}_q", TensorProto.INT8, [K, N], w.tobytes(), raw=True))
    # DQ activation, DQ weight -> MatMul (fp32) -> Q back to u8
    nodes.append(helper.make_node("DequantizeLinear", [prev, "act_scale", "act_zp"], [f"a{l}_f"]))
    nodes.append(helper.make_node("DequantizeLinear", [f"w{l}_q", "w_scale", "w_zp"], [f"w{l}_f"]))
    nodes.append(helper.make_node("MatMul", [f"a{l}_f", f"w{l}_f"], [f"m{l}_f"]))
    nodes.append(helper.make_node("QuantizeLinear", [f"m{l}_f", "act_scale", "act_zp"], [f"o{l}_q"]))
    prev = f"o{l}_q"

graph = helper.make_graph(
    nodes, "fc8_int8_bench",
    [helper.make_tensor_value_info("input_q", TensorProto.UINT8, [M, K])],
    [helper.make_tensor_value_info(prev, TensorProto.UINT8, [M, N])],
    inits)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)],
                          producer_name="snapdragon-npu-bench")
model.ir_version = 10
onnx.checker.check_model(model)
onnx.save(model, "fc8_int8.onnx")
print("saved fc8_int8.onnx:", LAYERS, "layers of", (M, K), "x", (K, N))

# fp16 single-layer FC — mirrors npu_fp16_test.c (fp16 capability proof)
w = rng.standard_normal((K, N)).astype(np.float16) * np.float16(0.02)
nodes_fp16 = [helper.make_node("MatMul", ["input_h", "w_h"], ["out_h"])]
inits_fp16 = [helper.make_tensor("w_h", TensorProto.FLOAT16, [K, N], w.tobytes(), raw=True)]
graph_fp16 = helper.make_graph(
    nodes_fp16, "fc_fp16_test",
    [helper.make_tensor_value_info("input_h", TensorProto.FLOAT16, [M, K])],
    [helper.make_tensor_value_info("out_h", TensorProto.FLOAT16, [M, N])],
    inits_fp16)
model_fp16 = helper.make_model(graph_fp16, opset_imports=[helper.make_opsetid("", 21)],
                               producer_name="snapdragon-npu-bench")
model_fp16.ir_version = 10
onnx.checker.check_model(model_fp16)
onnx.save(model_fp16, "fc_fp16.onnx")
print("saved fc_fp16.onnx")
