# Create mask for the guy in front of camera
## Video demo
<video controls src="video/test video.mp4" title="Title"></video>

Firstly, i have trained yolov5 model to get the bounding box of human face in image
Then, i have finetuned ResNet18 model of torchvision to get 68 landmarks of human face
Finally, the delaunay algorithm and affine transformn are used to convert human face to mask

### You can download the model from model folder
