# remove-image-background
remove background form image and video

example
https://google.github.io/mediapipe/images/selfie_segmentation_web.mp4

MediaPipe Selfie Segmentation segments the prominent humans in the scene. It can run in real-time on both smartphones and laptops. The intended use cases include selfie effects and video conferencing, where the person is close (< 2m) to the camera.


Models
In this solution, we provide two models: general and landscape. Both models are based on MobileNetV3, with modifications to make them more efficient. The general model operates on a 256x256x3 (HWC) tensor, and outputs a 256x256x1 tensor representing the segmentation mask. The landscape model is similar to the general model, but operates on a 144x256x3 (HWC) tensor. It has fewer FLOPs than the general model, and therefore, runs faster. Note that MediaPipe Selfie Segmentation automatically resizes the input image to the desired tensor dimension before feeding it into the ML models.

The general model is also powering ML Kit, and a variant of the landscape model is powering Google Meet. Please find more detail about the models in the model card.

ML Pipeline
The pipeline is implemented as a MediaPipe graph that uses a selfie segmentation subgraph from the selfie segmentation module.

Note: To visualize a graph, copy the graph and paste it into MediaPipe Visualizer. For more information on how to visualize its associated subgraphs, please see visualizer documentation.
