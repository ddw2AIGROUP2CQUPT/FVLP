# Vision-Language Joint Representation Learning for Sketch Less Facial Image Retrieval
## 🌟 Pipeline

## 💾 Dataset

We use two standard datasets as follows:

**The [FS2K-SDE](https://github.com/ddw2AIGROUP2CQUPT/FS2K-SDE) dataset** This dataset comprises two subsets: the FS2K-SDE1 dataset with 75,530 sketches (1,079 images for training, with the remainder being allocated for testing) and the FS2K-SDE2 dataset with 23,380 sketches (334 images for training, with the remaining sketches and images being used for testing).

## 📔 Citation
Dawei Dai, Shiyu Fu, Yingge Liu, Guoyin Wang,
Vision-language joint representation learning for sketch less facial image retrieval,
Information Fusion,
2024,
102535,
ISSN 1566-2535,
https://doi.org/10.1016/j.inffus.2024.102535.
(https://www.sciencedirect.com/science/article/pii/S1566253524003130)
Abstract: The traditional sketch-based facial image retrieval (SBFIR) framework assumes that a high-quality facial sketch has been prepared prior to the retrieval task. However, drawing such a sketch requires considerable skills and is time consuming, resulting in limited applicability. Sketch less facial image retrieval (SLFIR) framework aims to break these barriers through human–computer interaction during the sketching process. The primary challenges for the SLFIR problem can be noted that initial sketches (at early sketching) contain only local details and exhibit significant differences among users, resulting in poor performance at early stages and weak generalization abilities in practical testing. In this study, we developed a vision–language pretraining model to align the representation of facial images and their associated semantics. Based on this framework, we proposed a method for learning joint representations by fusing sketches with prior semantics, thereby enriching the information of initial sketches. Specifically, (1) we developed a series of well-designed operations to improve the quality of facial image–text pairs in the LAION-Face dataset; we trained a facial vision–language pretraining (FVLP) model to align the facial image and its semantics at the feature level. (2) subsequently, using FVLP as the backbone, we designed a convolutional attention module to fuse the multiscale features extracted from image encoder. This facilitated the learning of a multimodal representation crucial for the final retrieval process. In experiments, our proposed method achieved state-of-the-art performance at early stages on two public datasets; moreover, it exhibited good generalization capabilities during practical testing. Thus, our method significantly outperforms other baselines in terms of early retrieval performance. Codes are available at: https://github.com/ddw2AIGROUP2CQUPT/FVLP.
Keywords: SBIR; SLFIR; Multimodal learning
