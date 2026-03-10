from rknn.api import RKNN

ONNX_MODEL = 'best.onnx'
RKNN_MODEL = 'best.rknn'
DATASET = 'dataset.txt'

rknn = RKNN(verbose=True)

print('--> Config model')
rknn.config(
    mean_values=[[0,0,0]],
    std_values=[[255,255,255]],
    target_platform='rk3588'
)
print('done')

print('--> Loading ONNX model')
ret = rknn.load_onnx(model=ONNX_MODEL)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

print('--> Building RKNN model')
ret = rknn.build(
    do_quantization=True,
    dataset=DATASET
)
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

print('--> Export RKNN model')
ret = rknn.export_rknn(RKNN_MODEL)
if ret != 0:
    print('Export rknn failed!')
    exit(ret)

print('RKNN model generated:', RKNN_MODEL)

rknn.release()