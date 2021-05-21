# Installation
## Linux
https://github.com/isering/WoodPixel/blob/master/README.md
## Windows
Install packages:

 - Boost C++ Libraries (version 1.64+)
 - OpenCV (version 4.3+)
 - CMake (version 3.1+)
 
Configure/generate project using CMake, open the created solution in Visual Studio and build the project. Make sure to build Boost as well as the WoodPixel project in release mode.

# The Pipeline
![alt](https://github.com/philippmikus/WoodPixel/flowchart.PNG?raw=true)

The Wood Pixel pipeline consists of different tools which can be run in succesion via the command line using certain input parameters. Usually the tools output a JSON file and folders containing data that can be used as an input in the pipeline's following steps.

## 1. morph_grid
The morph_grid tool is used to segment an input image into patches to be used in the following pipeline steps.

**Inputs:**

*Basic (always needed) :*

 - -i "[path]": input image to be segmented
 - -f "[path]": filtered input image (can be identical to input image)
 - -o "[path]": output path

*Additionally for standard morph grid:*
 - -s [integer]: width/height of patches in pixels
 - --load_full_state "[path]": load saved state from JSON file (optional)
 - --load_partial_state "[path]": load saved partial state (edges, masks) from JSON file (optional)

*Additionally for adaptive Voronoi tessellation:*
 - --patch_count [integer]: amount of output patches
 - --max_iter [integer]: number of iterations
 - --threshold [integer]: minimum pixel amount per patch
 - -p [float]: segmentation parameter controlling color/spatial difference weighting

**Outputs:**

 - test.json: contains output patch data
 - patches: folder containing patch masks
 - ez_grid: folder containing patch masks + filtered images

**For standard morph grid:**
When running the standard morph grid version. A window ("EZgrid") will open, which allows to interactively edit the image's segementation. The following key bindings can be used to change the apperance of the image and edit the segmentation grid:

 - "1": Show input image
 - "2": Switch to bilateral filter image.
 - "3": Switch to rolling guidance filtered image.
 - "4": Switch to rolling guidance filter mask.
 - "5": Switch to bilateral filter mask.
 - "6": Switch to distance map.
 - "7": Switch to distance gradient.
 - "8": Switch to grid density.
 - "q": Don't show edge pixels.
 - "w": Show computed edge pixels.
 - "e": Show edge visualization.
 - "r": Show rolling guidance filtered edge image.
 - "t": Show bilateral filter edge image.
 - "y": Show drawn edge image.
 - "u": Show computed patches.
 - "i": Toggle draw grid.
 - "o": Toggle high visibility mode on/off
 - "a": Disable editing.
 - "s": Enter grid edit mode.
 - "d": Enter edge select mode.
 - "f": Enter filtered edge edit mode.
 - "g": Enter bilateral edge edit mode.
 - "h": Enter manual edge draw mode.
 - "z": Locally increase (Shift: decrease) grid density.
 - "x": Locally restore original grid cell size.
 - "c": Locally decrease (Shift: increase) grid density.

**For adaptive Voronoi tessellation:**
The selected patch count usually closely resembles the amount of output patches. However, depending on the parameter and threshold choice this amount can be smaller because some of the computed patches were not sufficiently large.
&ensp;In most cases, about 10 iterations are enough and further iterations only slightly impact the output.
&ensp;The threshold is the total minimum amount of pixels the output patches need to have, so the image size, patch count and parameter choice should be regarded.
&ensp;The parameter p is crucial to the quality of the output segmentation. In general, the desired parameter lies between 0 and 1. Smaller parameters emphasize more evenly sized and distributed patches, while bigger parameters will try to better take the image content into account. Thus, the patches will adhere to edges in the image better and will be cluttered in content-dense regions of the image. A safe parameter choice for most images lies at about 0.15, which can then be refined further.
&ensp; After running the Voronoi tessellation the computed segmentation will be depicted. It can be saved to the output directory by pressing space bar.


## 2. fit_patches
The fit_patches tool uses the morph_grid output to fit wood material to the given patches.

**Inputs:**

 - -i "[path]": input JSON file
 - -o "[path]": output directory
 - --vis/-v: turn on visualisation (off by default)
 - --steps/-s "[path]": intermediate output directory (optional)
 - --patches/-p "[path]" "[path]" ... : old patches for visualisation (optional)

**Outputs:**

 - result.json: resulting cut patterns used for export
 - render.png: output image
 - folders for target/texture responses, patches etc.

The computed patches from the morph_grid step will be fitted successively. Fitting a single Patch typically takes 0.5s to 3s, depending on pixel amount in the patches and the amount of provided wood samples. After completion the results are saved to the output directory.

*[EXTENSION]*
If the fit_patches extension is available, this process may be interrupted by pressing 'p'. After that individual fitting steps can be reverted one by one by pressing 'k'. It is then possible to load additional source textures by pressing 'l' and entering the path to the JSON file describing the additional textures in the console. This file is similar to fit_patches input JSON file, but descriptions regarding the target can be omitted. After that the process can be resumed by pressing 'p' again. 
&ensp; This extension is useful if the visualisation shows that after some time the initially provided wood samples are exhausted and fitted patches do no longer match the target well enough.

## 3. render_target

This tool is used to visualize the results of the fit_patches step in a nicer way. It does not provide intermediary outputs for following steps in the pipeline. The fit_patches tool already creates an output image, but that one does not necessarily use the full texture resolution which render_target does. Additionally the scale of the output image can be chosen using the "s" parameter. The second output image helps to visualize the burnt cut borders of the patches that arise during fabrication with a laser cutter.

**Inputs:**

 - -i "[path]": result.json (fit_patches step)
 - -o "[path]": output directory
 - -s [float]: output image scaling

**Outputs:**

 - image.png: rendered output image
 - image_boundary.png: rendered output image with additional visualization of the cut lines 


## 4. export_for_fab

**Inputs:**

 - -t "[path]": path to table.json
 - -v/--verify_markers: Visually verify marker positions in source textures
 - -o "[path]": output directory
 - --patches/-p "[path]" "[path]" ... : old patches for visualisation (optional)
 - -c "[path]" ("[path]" ...): path(s) to result.json file(s) (fit_patches step output)

**Outputs:**

 - result.json: resulting cut patterns used for export
 - render.png: unscaled output image
 - folders for target/texture responses, patches etc.
