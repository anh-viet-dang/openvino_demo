python open_model_zoo/demos/object_detection_demo/python/object_detection_demo.py \
    --model vehicle-detection-adas-0002.xml \
    --architecture_type ssd \
    --device CPU --prob_threshold 0.145 \
    --num_threads 8 \
    --input three_car.jpg \
    --output vehicle.jpg