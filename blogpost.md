# Introduction:
Our research project aims to investigate the paper "Unifying Diffusion Models’ Latent Space, With Applications To Cyclediffusion And Guidance"  ([Wu & De la Torre, 2022](https://arxiv.org/abs/2210.05559)). This paper proposes an alternative approach to the formulation of the latent space in diffusion models, which are instrumental in generative modeling, particularly in text-to-image models. Rather than using a sequence of gradually denoised images as the latent code, the authors propose using a Gaussian formulation, similar to the latent space used in Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and normalizing flows. They also propose a DPM-Encoder, which can map images to this latent space.

The paper presents two main findings from this new formulation. First, they observe that a common latent space appears when two diffusion models are independently trained on related domains. This led to the proposal of CycleDiffusion, a method for unpaired image-to-image translation. In experiments, CycleDiffusion outperformed previous methods based on GANs or diffusion models. When applied to large-scale text-to-image diffusion models, CycleDiffusion enabled the creation of zero-shot image-to-image editors.

Second, the authors found that the new formulation allows for the guidance of pre-trained diffusion models and GANs using a unified, plug-and-play approach based on energy-based models. In demonstrations, it was shown that diffusion models have better coverage of low-density sub-populations and individuals than GANs when guided by the CLIP model and a face recognition model.

## Related work:
Generative models, specifically Generative Adversarial Networks (GANs) and CycleGANs, have been the foundation for many advancements in image synthesis and translation tasks. GANs, as proposed by Goodfellow et al., offer a way to generate artificial data that is similar to some given real data. They have been used extensively in image synthesis, style transfer, and super-resolution among other applications. However, GANs often suffer from training instability and mode collapse.

CycleGANs, introduced by [Zhu et al., 2017](https://arxiv.org/abs/1703.10593), provide an effective solution for unpaired image-to-image translation by learning to translate images from one domain to another and vice versa. Despite their success, CycleGANs might produce artifacts in the translated images due to the unsupervised nature of the learning process.

Diffusion models, on the other hand, propose a different approach to generative modeling. Instead of learning a direct mapping from a latent space to a data distribution, diffusion models learn to gradually transform a simple noise distribution into a complex data distribution. However, their use in image-to-image translation tasks has been limited.

The work by [Khrulkov & Oseledets, 2022](https://arxiv.org/abs/2202.07477) and [Su et al., 2022](https://arxiv.org/abs/2203.08382) bring deterministic diffusion probabilistic models (DPMs) into the spotlight by introducing an optimal transport framework. However, these works focus on deterministic DPMs and do not provide insights into the latent space of stochastic DPMs, which this paper investigates. This paper extends the capabilities of diffusion models and brings them in line with the versatility of GANs and CycleGANs, while addressing the limitations of previous methods.

## Methods: 
The paper presents a novel methodological approach by reformulating the latent space of diffusion probabilistic models (DPMs). The central premise behind this reformulation is to provide a unified perspective on pre-trained generative models, thereby potentially enhancing their application and efficacy.
The research introduces CycleDiffusion, an innovative technique that uses diffusion models for image-to-image translation tasks, similar to CycleGANs. The CycleDiffusion method is based on training two independent diffusion models on two distinct domains. The reformulation of the latent space allows these models to achieve impressive performance on unpaired image-to-image translation tasks.

For zero-shot image-to-image translation, text-to-image diffusion models are used. This provides the advantage of translating images without the need for paired data in the source and target domain, which is a significant step forward for generative models in scenarios where obtaining paired data can be challenging.

Moreover, the researchers have expanded the possibilities of diffusion models by defining a new type of latent code that allows these models to be guided similarly to GANs. This is achieved through a plug-and-play mechanism that does not require fine-tuning on noisy images, which is a typical requirement for GANs.

## Experiments and results:
In the paper, the authors conducted several experiments to validate the effectiveness of their proposed method, CycleDiffusion, particularly in the context of unpaired and zero-shot image-to-image translation tasks. The models were trained on two separate domains independently, emulating the typical structure of CycleGANs but utilizing the reformulated diffusion model concept.

The CycleDiffusion approach was shown to outperform other state-of-the-art models in the same tasks. The authors also demonstrated the versatility of their model by applying it to text-to-image translation tasks, where it again showed excellent performance.

In addition to these translation tasks, the authors also explored the capability of diffusion models to be guided in the same way as GANs without requiring finetuning on noisy images, a process they refer to as "plug-and-play." The results were again encouraging, suggesting that diffusion models could have broader coverage of sub-populations and individuals than GANs, opening new opportunities for more flexible and inclusive generative models.

# Analyzing Weaknesses, Strengths, and Potential: Our Response
The paper presents a compelling approach to pre-trained generative models, introducing a reformulation of the latent space of diffusion probabilistic models (DPMs) that enhances their application and efficacy. This novel methodology, referred to as CycleDiffusion, demonstrated significant performance in unpaired image-to-image translation tasks and zero-shot image-to-image translation tasks. These are clear strengths of the paper, as they offer potential solutions to challenges inherent in these types of tasks, particularly where obtaining paired data can be difficult.

The proposed method's adaptability is another substantial strength. The defined latent code allows for diffusion models to be guided in a similar manner to GANs, introducing a plug-and-play mechanism that does not necessitate fine-tuning on noisy images. This could potentially open up new avenues for using generative models in various applications.

However, the paper also presents certain weaknesses, which our group has identified and aimed to address. One area of concern is the semantic variability of the encoded seed. The paper does not provide detailed insights into the nature of the components being encoded—are they primarily high-frequency components, or is there a mix of frequencies? Furthermore, it is unclear how steering at different levels of noise impacts the semantics. In the context of generative models, "steering" refers to guiding or manipulating the model's output by adjusting its input or internal representations. It is a crucial aspect to understand for effective implementation and fine-tuning of these models.

Additionally, the limitations of the method have not been thoroughly explored, specifically the influence of the input prompt on the diversity and success rate of the generated outputs. Understanding these limitations is vital for refining the model and expanding its applicability.

## Research questions:
Given the aforementioned weaknesses of the paper, the research questions that we will answer in this project are the following:

1. Explore semantic variability of the encoded seed.
   - a. Are only high-frequency components being encoded?
   - b. Does steering at different levels of noise result in different changes in the semantics?

2. Explore limits of the method, and the influence of the prompt: examine the feasibility of obtaining diverse outputs using similar prompts and investigate the factors affecting the success rate of generating non-similar images. We will analyze the limitations of the current methodology and evaluate the role of the input prompt in achieving the desired outcomes.

# Novel contributions:
The further research that we have done in this project has been answering the previously mentioned research questions. Additionally, the reproduction of their original code has also been performed to increase the quality of the research, but omitting the training of the models since that would require too much computing power.

## Methodology:

### Encoding of high-frequency components
The point of this research question is understanding which aspects of an image are the most important for the encoder. Our hypothesis is that high-frequency components are the most important (corners and edges). In order to test this, our method is to take an image and get two different versions of it: in one we apply a high pass filter (so it will only have high-frequency components), and in the other a low pass filter. Then we take the three images (the two edited images and the original), and encode them. If the encoded high-frequency image is more similar to the original encoded image than the encoded low-frequency image, it will mean that the high-frequency components of an image are the most relevant for the encoder. 

The images are transformed to the frequency domain with the Fast Fourier Transform (FFT), the frequency filter is applied, and then the images are converted back to pixel domain with the Inverse Fast Fourier Transform (IFFT). This will ensure that the encoder is given the same type of image it is used to. The frequency filter is a circle with a given radius (the value of which can be modified to change the aggressiveness of the filter), and some gaussian blur added to it, in order to make the filtered images more smooth. 

This function is called [pass_filter](demos/RQ1_1.ipynb), and the best values for its radius have been found to be 10 for the high-pass filter and 5 for the low-pass filter. This means that all the frequencies between the radius 5 to 10 are neither on the high-pass nor the low-pass images, but this should not necessarily be a problem since we want to compare the extremes of the frequency domain. 

The format of the encoded latent space was found to be hard to compare: it is a long vector, but cosine similarity and other metrics were not useful in this case since we want to compare human visual perception. Thus, another way of comparing the latent spaces has been used: a decoder is used to decode the latent spaces of all three images, and the resulting generated images are compared to see which generated images (high and low frequency) are more similar to the original generated image. The decoder is deterministic, so the comparison of the generated images can be assumed to be equal to the comparison of the latent spaces. 

The comparison is done with a metric, since visual comparison can be subjective and unreliable. The used metric is SSIM, the same metric that is used in the original paper, since it is much better correlated with human visual perception ([Wang et al., 2004](https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf)) than other numeric metrics such as mean squared error (MSE) or peak signal-to-noise ratio (PSNR). It takes into account both the structural information and the pixel values of the images, incorporating three main components: luminance, contrast, and structure.


### Steering the noise

Our contribution consists of a detailed analysis of the capabilities of the DPM-Encoder used in the task of unpaired image to image translation. The aim of this task is to generate a new image $\hat{x}$ from an input image $x$ that is very similar to the input image, but has a different domain.  The CycleDiffusion framework involves encoding and decoding images through a shared latent space. The latent space, denoted as z in the CycleDiffusion model, can be defined as follows:
$z \sim \text{DPMEnc}(z|x, G_1),\hat{x} = G_2(z)$

With the DPM-encoder defined as:
$x_1, ..., x_{T-1}, x_T \sim q(x_{1:T}|x_0), \epsilon_t = (x_{t-1} - \mu_T(x_t, t))/\sigma_t, t=T, ..., 1,$

$z := (x_t \oplus \epsilon_T \oplus ... \oplus \epsilon_2 \oplus \epsilon_1)$


We investigate the influence of the latent code $z$ on the generation process of $\hat{x}$. In particular, by modifying the latent code $z$ by adding Gaussian noise to it. The DPM-encoder is characterized by its dependence on the time step t ∈ T, which determines the level of noise added. As we increase the number of time steps, the sampled image becomes more saturated with noise. By manipulating the number of time steps in the DPM-encoder, also known as the diffusion process, we can control the amount of noise injected into the latent space. The resulting perturbed $\tilde{z}$ is then used as a starting point by the second pre-trained diffusion model to generate a new image $\hat{x}$. This allows us to evaluate the influence / capabilities of the latent code the DPM-Encoder provides, as we would expect that the more influence $z$ has on the reconstruction process of $\hat{x}$, the more divergence from an optimal domain-transferred generation is visible.

We provide a quantitative analysis of the results by also measuring realism (FID) and faithfulness (SSIM) of the generated images and also provide a qualitative analysis focussing on whether local details such as the background, lighting, pose, and overall color of the animal are preserved.

Through this investigation, we aim to gain insights into the behavior of the latent space under different noise conditions and understand how it influences the quality and fidelity of the generated images. We hypothesize that the same control over the generation can also be obtained by further improving the text prompts. We evaluate the trade-off between a noisy latent space $z$ and a quality improved textual prompt that provides a clearer guidance for the generation $\hat{x}$.
 
 
**Weakness** It is very interesting to see that this method seems to succeed in this task. This raises the question of how much control / influence the DPM-Encoder's representation indeed has on the generation process of $\hat{x}$.


# Results: 

## Encoding of high-frequency components
The experiments have been carried out following the previously explained methodology. In total, a batch of 50 cat pictures from the database used in the original paper have been studied. As formerly said, for each of these images two new versions have been rendered:

![](blogpost_results/output1.png)

These three versions have been encoded, and then decoded with a decoder trained on dogs.

The resulting generated images have the following form:

![](blogpost_results/output2.png)
![](blogpost_results/output3.png)
![](blogpost_results/output4.png)

As one can observe, each image has been modified to adjust the decoder image “into” the input image. Nevertheless, it is clear that this has more or less realistic results depending on the input: the decoded high frequency images keep having most of their characteristics (odd color combinations and sharp edges), whereas the low frequency images seem to lead to the most realistic results. 

The SSIM metric has been used to compare these images with each other (in pairs). The most relevant comparison is that of the generated dog (from the source image) with the generated high or low frequency images: we are comparing the difference between generating an image from the original latent space, and generating one from a (indirectly) modified latent space. The resulting statistics for the comparisons in all generated images for the original batch of 50 images are the following:

|     SSIM Score    | Mean   | Standard Deviation | Min                | Max                |
|------------------------------------------|--------|--------------------|--------------------|--------------------|
|  Generated vs High Pass      | 0.2347 | 0.0744             | 0.1162  | 0.4197 |
| Generated vs Low Pass       | 0.5364 | 0.0958             | 0.3215 | 0.7048 |

This can be more clearly visualized with the following violin plots:

![](blogpost_results/output5.png)

There is a clear difference in similarity between the generated image with high pass frequency filtering and low pass frequency filtering. Furthermore, it is also worth mentioning that the standard deviations are in similar ranges for both comparisons. 

The nature of the decoder is to merge its image (a dog in this case) with the input (the original and edited versions of cats). The result, as seen from the statistics, is that images generated from low frequency cats are more similar to the normal generated images. This means that it is easier for the decoder to merge its image with a low frequency image than a high frequency image. 

The reason behind this is that the low frequency image is more similar to noise, so there is not much information to begin with, and the decoder can easily merge it with its image. On the other hand, the high frequency images have more information, so there are more constraints and the decoder struggles to merge the images. 

This can be further proven by comparing the similarity between the original cat and the low frequency generated dog, and the original cat and the normal generated dog. The statistics are the following:
| SSIM Score     | Mean   | Standard Deviation | Min                | Max                |
|----------------------|--------|--------------------|--------------------|--------------------|
| Original cats vs generated low pass filtered dog | 0.3317 | 0.0883             | 0.1233| 0.5212 |
| Original cats vs generated dog                  | 0.5172 | 0.0509             | 0.4295| 0.6251  |

These can be visualized with a violin plot:

![](blogpost_results/output6.png)

The original cat is more similar to the normal generated dog than to the low frequency generated dog. Since the low frequency cat image has less information than both the original cat and the high frequency cat, it is easier for the decoder to put its dog image on top of it, and thus the resulting image is more similar to a dog than the original cat. 

From this explanation for the difference in similarities, it can be deduced that the higher frequency components of an image are more important, and therefore more predominant in the latent space of an image. 

## Steering the noise

# Conclusion:

# Student contributions:
Alon Shilo
- Worked on the first research question, together with Philip and Quim. 
- Helped set up the dependency notebook with Francesco.
- Implemented low/high pass filter for our experiments.
- Worked on the pipeline of our notebook, which consists of 3 different experiments for comparing images with the SSIM metric.
- Structured the github repo and cleaned some of it.


Dionne Gantzert


Francesco Tinner
- Worked on the second research question, together with Dionne.
- Helped set up the dependency notebook with Alon.
- Set-up CycleDiffusion on Lisa to be able to run experiments on the GPU cluster.
- Implemented modifications to the source code in order to run our own experiments.
- Worked on RQ1_2 notebook.


Philip Schutte
- Worked on the first research question, together with Alon and Quim.
- Investigated the source code of the CycleDiffusion paper and implemented the necessary changes in order to run our own experiments.


Quim Serra Faber
- Worked on the first research question, together with Philip and Alon. 
- Implemented low/high pass filter for our experiments.
- Wrote the general parts of the blogpost
- Studied and reported the results of the first research question in the blogpost
