-   _Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well._
-   _Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response._
-   _Describe your novel contribution._
-   _Results of your work (link that part with the code in the jupyter notebook)_
-   _Conclude_
-   _Close the notebook with a description of each student's contribution._

# Introduction:
Our research project aims to investigate the paper "Unifying Diffusion Models’ Latent Space, With Applications To Cyclediffusion And Guidance", written by Wu & De la Torre (2022). This paper proposes an alternative approach to the formulation of the latent space in diffusion models, which are instrumental in generative modeling, particularly in text-to-image models. Rather than using a sequence of gradually denoised images as the latent code, the authors propose using a Gaussian formulation, similar to the latent space used in Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and normalizing flows. They also propose a DPM-Encoder, which can map images to this latent space.

The paper presents two main findings from this new formulation. First, they observe that a common latent space appears when two diffusion models are independently trained on related domains. This led to the proposal of CycleDiffusion, a method for unpaired image-to-image translation. In experiments, CycleDiffusion outperformed previous methods based on GANs or diffusion models. When applied to large-scale text-to-image diffusion models, CycleDiffusion enabled the creation of zero-shot image-to-image editors.

Second, the authors found that the new formulation allows for the guidance of pre-trained diffusion models and GANs using a unified, plug-and-play approach based on energy-based models. In demonstrations, it was shown that diffusion models have better coverage of low-density sub-populations and individuals than GANs when guided by the CLIP model and a face recognition model.

## Related work:
Generative models, specifically Generative Adversarial Networks (GANs) and CycleGANs, have been the foundation for many advancements in image synthesis and translation tasks. GANs, as proposed by Goodfellow et al., offer a way to generate artificial data that is similar to some given real data. They have been used extensively in image synthesis, style transfer, and super-resolution among other applications. However, GANs often suffer from training instability and mode collapse.

CycleGANs, introduced by Zhu et al., provide an effective solution for unpaired image-to-image translation by learning to translate images from one domain to another and vice versa. Despite their success, CycleGANs might produce artifacts in the translated images due to the unsupervised nature of the learning process.

Diffusion models, on the other hand, propose a different approach to generative modeling. Instead of learning a direct mapping from a latent space to a data distribution, diffusion models learn to gradually transform a simple noise distribution into a complex data distribution. However, their use in image-to-image translation tasks has been limited.

The work by Khrulkov & Oseledets and Su et al. bring deterministic diffusion probabilistic models (DPMs) into the spotlight by introducing an optimal transport framework. However, these works focus on deterministic DPMs and do not provide insights into the latent space of stochastic DPMs, which this paper investigates. This paper extends the capabilities of diffusion models and brings them in line with the versatility of GANs and CycleGANs, while addressing the limitations of previous methods.

ADD SOME ADDITIONAL PAPERS THAT WE USE OURSELVES

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

# Novel contributions:

## Research questions:

## Methodology:

# Results: 

# Conclusion:

# Student contributions:
