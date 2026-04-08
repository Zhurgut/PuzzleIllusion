# Puzzle Illusions

In this repo, I'm sharing code to create jigsaw puzzles that have two distinct solutions, where the two solutions display different images. 

<div style="display: flex; gap: 10px;">
  <img src="assets/donutmug7.png" style="width: 49%; height: auto;" />
  <img src="assets/donutmug8.png" style="width: 49%; height: auto;" />
</div> 


## Inspiration and Background

I came across this problem in Matt Parker's video about ["How can a jigsaw have two distinct solutions?"](https://youtu.be/b5nElEbbnfU?si=KaLDxtgXktCinHvK). His approach to generating a jigsaw puzzle with multiple solutions left some room for improvement. Essentially, his program  "brute-forces a combinatorics problem by exploring millions of permutations for guess-and-check at every iteration", as one commenter put it. With a more sophisticated approach, we can generate bigger and nicer jigsaw puzzles with exactly two solutions. 

Once the puzzle pieces and solutions have been found, an image needs to be created so that both solutions look sensible as well. Ryan Burgert, who was featured in Matt's video, explained how to generate such images using diffusion models. A lot of really cool "diffusion illusions" are presented on their website: [Diffusion Illusions](https://diffusionillusions.com/). ([paper](https://arxiv.org/abs/2312.03817)). 

Since then, diffusion models have improved significantly. Daniel Geng et al. made similar images available on their website: [Visual Anagrams](https://dangeng.github.io/visual_anagrams/). ([paper](https://arxiv.org/abs/2311.17919)). They used the [DeepFloyd IF](https://github.com/deep-floyd/IF) pixel-based diffusion model to produce their amazing results. I highly recommend you check them out!

While using pixel based diffusion makes a lot of sense, I was excited about recent *latent* diffusion models. So I set out on the journey of making puzzles with two distinct solutions and generating images for them using a [Stable Diffusion 3.5](https://github.com/Stability-AI/sd3.5) model.



## Gallery

<div style="display: flex; gap: 10px;">
  <img src="assets/beardlady1.png" style="width: 49%; height: auto;" />
  <img src="assets/beardlady2.png" style="width: 49%; height: auto;" />
</div>
<br>


<div style="display: flex; gap: 10px;">
  <img src="assets/marylinplants1.png" style="width: 49%; height: auto;" />
  <img src="assets/marylinplants2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/picassoduckbunny1.png" style="width: 49%; height: auto;" /> 
  <img src="assets/picassoduckbunny2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/waterduck.png" style="width: 49%; height: auto;" />
  <img src="assets/waterduckbunny.png" style="width: 49%; height: auto;" />
</div>
<br>
 
<div style="display: flex; gap: 10px;">
  <img src="assets/kitchendoe1.png" style="width: 49%; height: auto;" />
  <img src="assets/kitchendoe2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/kitchendeer2.png" style="width: 49%; height: auto;" />
  <img src="assets/kitchendeer1.png" style="width: 49%; height: auto;" />
</div>
<br>

<!-- <img src="assets/castlefish1.png" style="width: 99%; height: auto;" />
<img src="assets/castlefish2.png" style="width: 99%; height: auto;" />
<br> 
<br>  -->

<div style="display: flex; gap: 10px;">
  <img src="assets/castlefish1.png" style="width: 49%; height: auto;" />
  <img src="assets/castlefish2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/fishcastle1.png" style="width: 49%; height: auto;" />
  <img src="assets/fishcastle2.png" style="width: 49%; height: auto;" />
</div>
<br>


<br>
Puzzles can also be generated using a target image, so that the second solution may reveal, for example, your favourite YouTuber.

<div style="display: flex; gap: 10px;">
  <img src="assets/mattrock1.png" style="width: 49%; height: auto;" />
  <img src="assets/mattrock2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/steverocks1.png" style="width: 49%; height: auto;" />
  <img src="assets/steverocks2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/guitarpark1.png" style="width: 49%; height: auto;" />
  <img src="assets/guitarpark2.png" style="width: 49%; height: auto;" />
</div>
<br>

## How it works

### Puzzle Generation

The goal is to find a jigsaw puzzle with exactly two solutions. Rather than creating entirely random jigsaw puzzles, I create randomized jigsaw puzzles which have at least two solutions by construction. 

I first assign a unique number to each puzzle piece knob/hole, before shuffling the pieces around randomly. 

<p align="center">
  <img src="assets/sol1.png" height="300" />
  <span style="font-size: 40px; margin: 0 15px;">→</span>
  <img src="assets/sol2.png" height="300" />
</p>


This yields a series of constraints that dictate which outies have to fit into which innies. E.g. connectors (1) and (-1) need to fit together. Then the second puzzle solution gives that (-1) has to fit into (-8), and (-8) in turn connects to (8), etc. By following these constraints, we can assign connector types to the different puzzle pieces, such that the resulting puzzle has the two solutions as above:

<img src="assets/sol3.png" height="300" />

Here a positive number designates a male connector, and the corresponding negative number the fitting female connector. 
So 1 → (-1) → (-8) → 8 → 5 → (-5) → (-12) → 12 becomes
1 → (-1) → 1 → (-1) → 1 → (-1) → 1 → (-1)

The resulting set of puzzle pieces has at least two distinct solutions. However it might have significantly more solutions. Rather than solving the puzzle completely to check how many solutions there are, which can be quite costly for larger puzzle sizes, we can filter out a large portion of these candidate solutions based on some easy-to-check conditions. For example, when we construct a puzzle in this way, if any of the puzzle pieces are rotationally symmetric, or if the puzzle has duplicate pieces, we know for sure that there are more than just two solutions.

An additional desirable quality that we want is "no repeated matches", i.e. no two pieces are connected in the same way in both solutions. This constraint is incorporated into the shuffling process. Care is taken so that in the second solution no connection from the first solution exists (unlike in the above example, where 7 connects to (-7) in both solutions). 

This approach facilitates iterating over many candidate solutions quickly. We search for a puzzle that has as many different connectors as possible, since puzzles with more different connectors are less likely to have any extra solutions. The largest puzzle I was able to find this way with exactly two solutions has size 9 by 9. For sizes larger than that, solving the puzzle to verify that there are no other solutions becomes too computationally expensive. 


### Generating the images

Generating such visual anagrams using latent diffusion models is challenging, because the transformation which maps one image view to the other is defined in pixel space, while the denoising process happens in latent space. This is an issue because, for example, rotating the latent image representation by 90° does not directly correspond to a rotated version of the pixel image:


| | |
|---|---|
| <img src="assets/donut.png" width="400"> | <img src="assets/donut90.png" width="400"> |
| image | image → encode → rotate(90) → decode → rotate(-90) |

Consequently, the transformation between the two image views has to be performed in pixel space. 

Since the VAE can only decode and encode images losslessly when they are clean (i.e. they do not contain noise), we have to denoise images first, before we can apply the transformation between the two image views. 

It becomes necessary to optimize two latent images, $x_t$ and $y_t$, simultaneously, one for each image view. The following image illustrates how the velocity vector for one denoising step for the image $x_t$ is calculated:

<img src="assets/imagegen.png" width="500">

At every step, we denoise and transform the image $y_t$ and treat the resulting image as the target clean image of $x_t$. Finally, we average the velocity vector which moves $x_t$ towards this target with the model's standard noise prediction, to get the direction for the current denoising step. The velocity vector for $y_t$ is computed analogously. 

This method is computationally quite intensive, but I found it necessary in order to get nice textures in the output.

## Installation

...

## Future Work
 
performance improvements

parallelize puzzle finding and puzzle solving

translating "solving the puzzle" to SAT, where satisfiable implies there is a third solution. pass to optimized sat solver, might be able to handle larger sizes...

Generating puzzles with no duplicate pieces, and even better no rotationally symmetric pieces as well, by construction. at size e.g. 16x16, it is almost impossible to find a puzzle with all distinct pieces, need a more sophisticated approach, if it is even possible. 



