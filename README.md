# Puzzle Illusions

In this repo, I'm sharing code to create jigsaw puzzles that have two distinct solutions, where the two solutions display different images. 

<div style="display: flex; gap: 10px;">
  <img src="assets/donutmug7.png" style="width: 49%; height: auto;" />
  <img src="assets/donutmug8.png" style="width: 49%; height: auto;" />
</div> 


## Inspiration and Background

I came across this problem in Matt Parker's video about ["How can a jigsaw have two distinct solutions?"](https://youtu.be/b5nElEbbnfU?si=KaLDxtgXktCinHvK). His approach to generating a jigsaw puzzle with multiple solutions left some room for improvement. Essentially, his program  "brute-forces a combinatorics problem by exploring millions of permutations for guess-and-check at every iteration", as one commenter put it. With a more sophisticated approach, we can generate bigger and nicer jigsaw puzzles with exactly two solutions. 

Once the puzzle pieces and solutions have been found, an image needs to be created so that both solutions look sensible as well. Ryan Burgert, who was featured in Matt's video, explained how to generate such images using diffusion models. A lot of really cool "diffusion illusions" are presented on their website: [Diffusion Illusions](https://diffusionillusions.com/). ([paper](https://arxiv.org/abs/2312.03817)). 

Since then, diffusion models have improved significantly. Daniel Geng et al. made similar images available on their website: [Visual Anagrams](https://dangeng.github.io/visual_anagrams/). ([paper](https://arxiv.org/abs/2311.17919)). They used the [DeepFloyd IF](https://github.com/deep-floyd/IF) pixel-based diffusion model to produce their amazing results.

While using pixel based diffusion makes a lot of sense, I was excited about recent *latent* diffusion models. So I set out on the journey of making puzzles with two distinct solutions and generating images for them using the [Stable Diffusion 3.5 Medium](https://github.com/Stability-AI/sd3.5) model. I adapted parts of the approach described in the [LookingGlass](https://arxiv.org/abs/2504.08902) paper.
 


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
  <img src="assets/waterduck1.png" style="width: 49%; height: auto;" />
  <img src="assets/waterduck2.png" style="width: 49%; height: auto;" />
</div>
<br> 
 
<div style="display: flex; gap: 10px;">
  <img src="assets/realduck.png" style="width: 49%; height: auto;" />
  <img src="assets/realbunny.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/robotcake1.png" style="width: 49%; height: auto;" />
  <img src="assets/robotcake2.png" style="width: 49%; height: auto;" />
</div>
<br>
 
<div style="display: flex; gap: 10px;">
  <img src="assets/fruitdeer1.png" style="width: 49%; height: auto;" />
  <img src="assets/fruitdeer2.png" style="width: 49%; height: auto;" />
</div>
<br>


<img src="assets/fishcastle1.png" style="width: 99%; height: auto;" />
<img src="assets/fishcastle2.png" style="width: 99%; height: auto;" />
<br> 
<br> 
<!-- 
<div style="display: flex; gap: 10px;">
  <img src="assets/castlefish1.png" style="width: 49%; height: auto;" />
  <img src="assets/castlefish2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/fishcastle1.png" style="width: 49%; height: auto;" />
  <img src="assets/fishcastle2.png" style="width: 49%; height: auto;" />
</div>
<br> -->

<div style="display: flex; gap: 10px;">
  <img src="assets/wine1.png" style="width: 49%; height: auto;" />
  <img src="assets/wine2.png" style="width: 49%; height: auto;" />
</div>
<br>

<div style="display: flex; gap: 10px;">
  <img src="assets/citycar2.png" style="width: 49%; height: auto;" />
  <img src="assets/citycar1.png" style="width: 49%; height: auto;" />
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

The goal is to find a jigsaw puzzle with exactly two solutions. My algorithms repeatedly constructs randomized jigsaw puzzles which have at least two solutions. Many such candidate jigsaw puzzles are generated, while candidates that must have more than two solutions are filtered out. The goal is to find a jigsaw puzzle that is as likely as possible to only have two solutions. Since verifying that indeed no additional solutions exist is the most expensive operation, this step is only performed at the end on the most promising candidate. 


#### In Detail

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

The resulting set of puzzle pieces has at least two distinct solutions. However it might have significantly more solutions. Rather than solving the puzzle completely to check how many solutions there are, which can be quite costly for larger puzzle sizes, we can filter out a large portion of these candidate puzzles based on some easy-to-check conditions. For example, when we construct a puzzle in this way, if any of the puzzle pieces are rotationally symmetric, or if the puzzle has duplicate pieces, we know for sure that there are more than just two solutions.

An additional quality that we want is "no repeated matches", i.e. no two pieces are connected in the same way in both solutions. This constraint is incorporated into the shuffling process. Care is taken so that in the second solution no connection from the first solution exists (unlike in the above example, where 7 connects to (-7) in both solutions). 

This approach facilitates iterating over many candidate solutions quickly. We search for a puzzle where the pieces are as different from each other as possible, as that makes it easier for the puzzle solver to verify the absence of more than two solutions. The largest puzzle I was able to find this way with exactly two solutions has size 10x10. For sizes larger than that, solving the puzzle to verify that there are no other solutions becomes very computationally expensive, due to the exponential blow up in complexity. 

## How to run the code

### Generating Images

#### Requirements
 
The code was tested using a CUDA capable GPU with 16GB of VRAM. If your GPU is smaller, or from a different vendor, you may need to modify the source code and choose a different model quantization strategy. (Running on CPU is very slow).  
Furthermore, you will need to share your contact information on [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) in order to get access to the Stable Diffusion model. 

The Python environment can be set up on Linux/WSL with the following commands:

```bash
git clone https://github.com/Zhurgut/PuzzleIllusion.git
cd PuzzleIllusion
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are new to HuggingFace, follow the instructions in the first cell of [src/main.ipynb](/src/main.ipynb) to login. 
Once installation is complete, you can open the project in VS Code and use [src/main.ipynb](/src/main.ipynb) to generate images for the jigsaw puzzles in the `/puzzles` folder. 

### Generating Jigsaw Puzzles

For better performance, the jigsaw generation is implemented in julia. The fastest way to install julia is via [juliaup](https://julialang.org/downloads/).

On Linux, that is: 

```bash
curl -fsSL https://install.julialang.org | sh
```

Jigsaw puzzles can be generated using the `generate_puzzle` function in [src/main.ipynb](/src/main.ipynb). The results are saved in the `/puzzles` folder. Required julia packages will be installed automatically on first use. 

## Future Work
 
At larger sizes, e.g. 16x16, it becomes virtually impossible to find jigsaw puzzles without duplicate pieces, is there a more sophisticated/mathematical construction method?




