# Forestrack Models

### Algorithm

The pseudo-code for the system used to detect the deforestration statistics of Sri Lanka is available here.

[pseudo_code](DOCS/images/psuedo%20code.PNG)

### Input

We use the sentinelHUB API to download satelite imagery to create a dataset as well and we use a cloud removal technique by stacking satelite imagery during different time periods of time.

[input](DOCS/images/forestrack4.png)

### Model Architecture.

We use a base UNET model trained on a NASA dataset as the segmentation model, this can be improved but due to time contraints was unable to move forward

### Model Results

[input](DOCS/images/forestrack.png)
[input](DOCS/images/forestrack2.png)
[input](DOCS/images/forestrack3.png)