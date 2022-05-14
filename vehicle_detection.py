import cv2
from colorama import Fore

# import inference adapter and helper for runtime setup
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
# import model wrapper class
from openvino.model_zoo.model_api.models import SSD
# from openvino.model_zoo.model_api.models.utils import Detection


# define the path to mobilenet-ssd model in IR format
model_path = "vehicle-detection-adas-0002.xml"

# create adapter for OpenVINO™ runtime, pass the model path
model_adapter = OpenvinoAdapter(create_core(), model_path, device="CPU")

# create model API wrapper for SSD architecture
# preload=True loads the model on CPU inside the adapter
configuration = {
    'resize_type': None,
    'mean_values': None,
    'scale_values': None,
    'reverse_input_channels': False,
    'path_to_labels': None,
    'confidence_threshold': 0.14,
    'input_size': (600, 600), # The CTPN specific
    'num_classes': None, # The NanoDet and NanoDetPlus specific
}
ssd_model = SSD(model_adapter, configuration=configuration, preload=True)

input_data = cv2.imread("three_car.jpg")

# apply input preprocessing, sync inference, model output postprocessing
results = ssd_model(input_data)
# open_model_zoo/demos/object_detection_demo/python/object_detection_demo.py
print(Fore.RED)
print(results)

print(Fore.GREEN,'*'*18)
for result in results[0]:
    print(result.score)
# open_model_zoo/demos/object_detection_demo/python/object_detection_demo.py