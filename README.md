# Puzzle Illusions

In this repo I'm sharing code to create jigsaw puzzles that have two distinct solutions, where the two solutions display different images. 

<div style="display: flex; gap: 10px;">
  <img src="assets/donutmug7.png" style="width: 49%; height: auto;" />
  <img src="assets/donutmug8.png" style="width: 49%; height: auto;" />
</div> 


## Inspiration and Background

I came across this problem in Matt Parker's video about ["How can a jigsaw have two distinct solutions?"](https://youtu.be/b5nElEbbnfU?si=KaLDxtgXktCinHvK). His approach to generate a jigsaw puzzle with multiple solutions left some room for improvement. Essentially, his program  "brute-forces a combinatorics problem by exploring millions of permutations for guess-and-check at every iteration", as one commenter put it. With a more sophisticated approach, we can generate bigger and nicer jigsaw puzzles with exactly two solutions. 

Once the puzzle pieces and solutions have been found, an image needs to be created so that both solutions look sensible as well. Ryan Burgert was featured in Matt's video, who explained how to generate such images using diffusion models. A lot of really cool "diffusion illusions" are presented on their website: [Diffusion Illusions](https://diffusionillusions.com/). ([paper](https://arxiv.org/abs/2312.03817)). 

Since then, diffusion models have improved significantly. Daniel Geng et al. made similar images available on their website: [Visual Anagrams](https://dangeng.github.io/visual_anagrams/). ([paper](https://arxiv.org/abs/2311.17919)). They used the [DeepFloyd IF](https://github.com/deep-floyd/IF) pixel-based diffusion model to produce their amazing results. I highly recommend you check them out!

While using pixel based diffusion makes a lot sense, I was excited about recent *latent* diffusion models. So I set out on the journey of making puzzles with two distinct solutions and generating images for them using a [Stable Diffusion 3.5](https://github.com/Stability-AI/sd3.5) model.



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
  <img src="assets/waterduck.png" style="width: 49%; height: auto;" />
  <img src="assets/waterduckbunny.png" style="width: 49%; height: auto;" />
</div>
<br>



Puzzles can also be generated using a target image, so that the second solution may reveal, for example, your favourite youtuber.

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



<p align="center">
  <img src="assets/sol1.png" height="300" />
  <span style="font-size: 40px; margin: 0 15px;">→</span>
  <img src="assets/sol2.png" height="300" />
</p>


I first assign a number to each puzzle piece knob/hole, before shuffling the pieces around randomly. This yields a series of constraints, that dictate which outies have to fit into which innies. E.g. connectors (1) and (-1) need to fit together. Then the second puzzle solution gives that (-1) has to fit into (-8), and (-8) in turn connects to (8), which connects to (5), from there to (-5) to (-12) to (12), and from (12) back to (1). By solving for these constraints, we can assign piece connectors to the different puzzle pieces, such that the resulting puzzle has the two solutions as above:

<img src="assets/sol3.png" height="300" />

Here a positive number designates a male connector, and the corresponding negative number the fitting female connector. 
So 
1 → (-1) → (-8) → 8 → 5 → (-5) → (-12) → 12 becomes
1 → (-1) → 1 → (-1) → 1 → (-1) → 1 → (-1)

This yields a set of puzzle pieces which has at least two solutions. However it might have significantly more solutions. Rather than checking how many solutions there are, which can be quite costly for larger puzzle sizes, we can filter out a large portion of candidate solutions based on some easy-to-check conditions. For example, when we construct a puzzle in this way, if any of the puzzle pieces are rotationally symmetric, or if the puzzle has duplicate pieces, we know for sure that there are more than just two solutions.

An additional desirable quality that we want is "no repeated matches", i.e., no two pieces are connected in the same way in both solutions. This constraint is incorporated into the process of shuffling the pieces around. Care is taken so that in the second solution no connection from the first solution exists (unlike in the above example, where 7 connects to (-7) in both solutions). 

This approach facilitates iterating over many candidate solutions quickly. We search for a puzzle that has as many different connectors as possible, since puzzles with more different connectors are less likely to have any extra solutions. The largest puzzle I was able to find this way with exactly two solutions has size 9 by 9. For sizes larger than that, solving the puzzle to verify that there are no other solutions becomes quite hard and takes a long time. 


### Generating the images

To generate the images I follow conceptually a similar approach as described in the [Visual Anagrams](https://arxiv.org/abs/2311.17919) paper. Two noise predictions are gathered, one for the base and one for the transformed image. These are then averaged and one denoising step is taken. 
Generating such visual anagrams using latent diffusion models is challenging, because the transformation which maps one image view to the other has to be performed in pixel space. For example, rotating the latent image representation by 90° does not directly correspond to a rotated version of the pixel image:


| | |
|---|---|
| <img src="assets/donut.png" width="400"> | <img src="assets/donut90.png" width="400"> |
| image | image → encode → rotate(90) → decode → rotate(-90) |

In order to perform the image transformation in pixel space, we have to first decode the latent representation, then transform the pixels, and then reencode the resulting image again, so we can query the model for the second noise prediction. However, the VAE has been exclusively trained on clean images, whereas the latent image representations are noisy throughout the image generation process. The roundtrip of decode → transform → encode removes all the noise from the latent image. 

So in order to get everything to work, two latent images are optimized simultaneously. To compute the final noise prediction for image 1, two noise prediction are averaged. One noise prediction consists of the default noise prediction by the model. The second one is computed by first optimizing image 2 to the end to get a clean image. This image is then decoded, transformed and encoded to yield a target image for image 1. This target image is then used to analytically compute a second noise prediction, which pushes the latent image 1 towards it. The noise prediction for image 2 is calculated analagously. By following this process, the two seperate latent images converge in on each other during the denoising process. The two finished latent representations are numerically quite different, but they represent the same pixel data. 

This approach is quite computationally intensive, but I found it necessary to get clean textures in the output.

## Future Work

performance improvements

parallelize puzzle finding and puzzle solving

translating "solving the puzzle" to SAT, where satisfiable implies there is a third solution. pass to optimized sat solver, might be able to handle larger sizes...

Generating puzzles with no duplicate pieces, and even better no rotationally symmetric pieces as well, by construction. at size e.g. 16x16, it is almost impossible to find a puzzle with all distinct pieces, need a more sophisticated approach, if it is even possible. 



