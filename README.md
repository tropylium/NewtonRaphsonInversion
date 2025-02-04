See [original GNRI repository](https://github.com/dvirsamuel/NewtonRaphsonInversion). 

# Editing Images using Text
Observe the similar pose of the foreground animal and background of two images. 

|"a lion is sitting in the grass at sunset"|"a dog with a funny red hat is sitting in the grass at sunset"|"a raccoon sitting in a field of cherry blossoms, photorealistic"|
|---|---|---|
|![](sample_images/lion.png)|![](sample_images/dog.png)|![](sample_images/raccoon.png)|

# My Changes
* Fixed various errors to get `RealTimeEditingNotebook.ipynb` working (import, device errors)
* Updated GNRI algorithm to latest version of paper
* Fixed CFG error, allowing editing with guidance scale > 1.0
* Fixed inversion not actually inverting to the final timestep 
  * (this may be an SDXL-specific thing, not sure if this is actually a bug)
  * Appears subjectively to give better results on a small number of test runs
* Additionally separated inversion/ inference to two GPU's because mine aren't large enough to fit SDXL on one GPU
* Implemented GNRI with DDIM Scheduler in addition to Euler Scheduler
