## SoK: All You Need to Know About On-Device ML Model Extraction - The Gap Between Research and Practice

This repo contains existing paper works on ML Model Extraction categorized into four distinct types of threat models, from the attacker’s and defender’s perspectives, respectively.

## Abstract

On-device ML is increasingly used in different applications. It brings convenience to offline tasks and avoids sending user-private data through the network. On-device ML models are valuable and may suffer from model extraction attacks from different categories. Existing studies lack a deep understanding of on-device ML model security, which creates a gap between research and practice. Our paper and this repository provide a systematization approach to classify existing model extraction attacks and defenses based on different threat models. We evaluated well-known research projects from existing work with real-world ML models and discussed their reproducibility, computation complexity, and power consumption. We identified the challenges for research projects in wide adoption in practice. We also provided directions for future research in ML model extraction security. We further evaluated well-known research projects from existing work with real-world ML models and discussed their reproducibility, computation complexity, and power consumption (more in `evaluate.md`)

## Threat Models for Model Extraction

The scope of this survey focuses only on ML model extraction or stealing and is mainly defined by the following aspects. 

**Attacker’s perspective**  

1.  **App-based attacks:** attackers assume they can gain access to the application files and they perform application de-packaging or decompiling, and extract the model files.
2.  **Device-based attacks:** attackers assume they can access the IoT devices and gain access to the memory and force a vulnerable application to launch and load ML models into memory or consistently scan the memory to wait for models to be loaded.
3.  **Communication-based attacks:** attackers can intercept communication between various memory regions and hardware architectures on an IoT device. These data can help to recover partial or complete details of ML models.
4.  **Model-based attacks:**  attackers assume to be able to send (selective) input queries, and receive inference results to assess the functionality of models and fine-tune the data to send in subsequent steps and then go through the process back and forth to train substitute models.

**Defender’s perspective**

1.  **App-based defense:** defenders apply techniques, including encryption, obfuscation, or customized protection to model files in an app package.
2.  **Device-based defense:** defenders apply techniques, such as secure hardware, to prevent arbitrary memory access. Defenders can also customize hardware to support computation on encrypted data so that memory extraction will not reveal plaintext models.
3.  **Communication-based defense:** defenders apply data transformation, encryption, and randomization techniques to prevent side-channel information leakage and enable further calculation based on the transformed data in the memory components.
4.  **Model-based defense:** defenders apply weight obfuscation, misinformation, and differential privacy to increase the effort of attackers in training equivalent student models.

 ![Alt text](AC_DC_Definition.png?raw=true&sanitize=true "Optional Title")

## Citing our paper

```plaintext
@inproceedings {nayan2024sok,
author = {Tushar Nayan and Qiming Gao and Mohammed Al Duniawi and Marcus Botacin and Selcuk Uluagac and Ruimin Sun},
title = {SoK: All You Need to Know About On-Device ML Model Extraction - The Gap Between Research and Practice},
booktitle = {33rd {USENIX} Security Symposium ({USENIX} Security 24)},
year = {2024},
url = {},
publisher = {{USENIX} Association},
month = aug,
}
```

## Table of Contents

**[Attacker’s perspective](#attacker’s-perspective)**

1.  **[App-based attacks](#app-based-attacks)**
2.  **[Device-based attack](#device-based-attacks)**
3.  **[Communication-based attacks](#communication-based-attacks)**
4.  **[Model-based attacks](#model-based-attacks)**

**[Defender’s perspective](#defender’s-perspective)**

1.  **[App-based defense](#app-based-defense)**
2.  **[Device-based defense](#device-based-defense)**
3.  **[Communication-based defense](#communication-based-defense)**
4.  **[Model-based defense](#model-based-defense)**

---

## **Overview**

### Attacker’s perspective <a name="attacker’s-perspective"></a>

*   **App-based attacks** <a name="app-based-attacks"></a>
    *   A First Look at Deep Learning Apps on Smartphones [\[Paper\]](%5Bhttps://arxiv.org/pdf/1812.05448%5D(https://arxiv.org/abs/1812.05448)) [\[Code\]](https://github.com/xumengwei/MobileDL)
    *   Smart App Attack: Hacking Deep Learning Models in Android Apps [\[Paper\]](https://arxiv.org/abs/2204.11075) [\[Code\]](https://github.com/Jinxhy/SmartAppAttack)
    *   Mind Your Weight(s): A Large-scale Study on Insufficient Machine Learning Model Protection in Mobile Apps [\[Paper\]](https://www.usenix.org/conference/usenixsecurity21/presentation/sun-zhichuang) [\[Code\]](https://github.com/RiS3-Lab/ModelXRay)
    *   Understanding Real-world Threats to Deep Learning Models in Android Apps [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3548606.3559388)
*   **Device-based attacks** <a name="device-based-attacks"></a>
    *   Mind Your Weight(s): A Large-scale Study on Insufficient Machine Learning Model Protection in Mobile Apps [\[Paper\]](https://www.usenix.org/conference/usenixsecurity21/presentation/sun-zhichuang) [\[Code\]](https://github.com/RiS3-Lab/ModelXRay)
    *   Understanding Real-world Threats to Deep Learning Models in Android Apps [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3548606.3559388)
*   **Communication-based attacks** <a name="communication-based-attacks"></a>
    *   Security Analysis of Deep Neural Networks Operating in the Presence of Cache Side-Channel Attacks [\[Paper\]](https://arxiv.org/abs/1810.03487) [\[Code\]](https://github.com/Sanghyun-Hong/DeepRecon)
    *   CSI NN: Reverse Engineering of Neural Network Architectures Through Electromagnetic Side Channel [\[Paper\]](https://www.usenix.org/conference/usenixsecurity19/presentation/batina)
    *   Cache Telepathy: Leveraging Shared Resource Attacks to Learn DNN Architectures [\[Paper\]](https://www.usenix.org/conference/usenixsecurity20/presentation/yan)
    *   Open DNN Box by Power Side-Channel Attack [\[Paper\]](https://ieeexplore.ieee.org/document/9000972)
    *   Reverse Engineering Convolutional Neural Networks Through Side-channel Information Leaks [\[Paper\]](https://www.csl.cornell.edu/~zhiruz/pdfs/rev-cnn-dac2018.pdf)
    *   GANRED: GAN-based Reverse Engineering of DNNs via Cache Side-Channel [\[Paper\]](https://dl.acm.org/doi/pdf/10.1145/3411495.3421356)
    *   DeepEM: Deep Neural Networks Model Recovery through EM Side-Channel Information Leakage [\[Paper\]](https://www.computer.org/csdl/proceedings-article/host/2020/09300274/1pQJ2QMAE00)
    *   Stealing Neural Networks via Timing Side Channels [\[Paper\]](https://arxiv.org/abs/1812.11720)
    *   HuffDuff: Stealing Pruned DNNs from Sparse Accelerators [\[Paper\]](https://dl.acm.org/doi/abs/10.1145/3575693.3575738)
    *   Hermes Attack: Steal DNN Models with Lossless Inference Accuracy [\[Paper\]](https://www.usenix.org/conference/usenixsecurity21/presentation/zhu)
    *   Leaky DNN: Stealing Deep-Learning Model Secret with GPU Context-Switching Side-Channel [\[Paper\]](https://ieeexplore.ieee.org/document/9153424)
    *   Stealing Neural Network Models through the Scan Chain: A New Threat for ML Hardware [\[Paper\]](https://eprint.iacr.org/2021/167)
    *   DeepSniffer: A DNN Model Extraction Framework Based on Learning Architectural Hints [\[Paper\]](https://dl.acm.org/doi/10.1145/3373376.3378460) [\[Code\]](https://github.com/xinghu7788/DeepSniffer)
    *   DeepSteal: rowhammer-based side channel for ML model weight stealing [\[Paper\]](https://arxiv.org/abs/2111.04625) [\[Code\]](https://github.com/casrl/DeepSteal-exploit/tree/master)
*   **Model-based attacks** <a name="model-based-attacks"></a>
    *   ML-Doctor: Holistic Risk Assessment of Inference Attacks Against Machine Learning Models [\[Paper\]](https://www.usenix.org/conference/usenixsecurity22/presentation/liu-yugeng) [\[Code\]](https://github.com/liuyugeng/ML-Doctor)
    *   Stealing Hyperparameters in Machine Learning [\[Paper\]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418595)
    *   Towards Reverse-Engineering Black-Box Neural Networks [\[Paper\]](https://arxiv.org/abs/1711.01768) [\[Code\]](https://github.com/coallaoh/WhitenBlackBox)
    *   ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data [\[Paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/5432) [\[Code\]](https://bitbucket.org/iiscseal/activethief/src/master/)
    *   ML-Stealer: Stealing Prediction Functionality of Machine Learning Models with Mere Black-Box Access [\[Paper\]](https://ieeexplore.ieee.org/abstract/document/9724507?casa_token=fe5g_XutmagAAAAA:hu_7RwHu07rDIa_DUSJUaQ18gr1az3Qlw9sq-8KABxBOfEZLdeoL-ORuI8KhHZufoPpL4HIilg)
    *   Knockoff Nets: Stealing Functionality of Black-Box Models [\[Paper\]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.pdf) [\[Code\]](https://github.com/tribhuvanesh/knockoffnets) 
    *   Simulating Unknown Target Models for Query-Efficient Black-box Attacks [\[Paper\]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.pdf) [\[Code\]](https://github.com/machanic/SimulatorAttack)

### **Defender’s perspective** <a name="defender’s-perspective"></a>

*   **App-based defense** <a name="app-based-defense"></a>
    *   TFSecured - Small library for TensorFlow proto model's (\*.pb) encryption/decryption [\[Code\]](https://github.com/dneprDroid/tfsecured)
    *   MindSpore [\[Code\]](https://github.com/mindspore-ai)
    *   Knox [\[Code\]](https://github.com/pinterest/knox)
    *   MACE [\[Code\]](https://github.com/XiaoMi/mace/blob/master/tools/python/encrypt.py)
    *   m2cgen [\[Code\]](https://github.com/BayesWitnesses/m2cgen#supported-models)
    *   MindDB [\[Code\]](https://github.com/mindsdb/mindsdb)
    *   MMGuard [\[Code\]](https://github.com/MMGuard123/MMGuard)
*   **Device-based defense** <a name="device-based-defense"></a>
    *   MyTEE: Own the Trusted Execution Environment on Embedded Devices [\[Paper\]](https://www.ndss-symposium.org/ndss-paper/mytee-own-the-trusted-execution-environment-on-embedded-devices/) [\[Code\]](https://github.com/sssecret2019/mytee)
    *   SANCTUARY: ARMing TrustZone with User-space Enclaves [\[Paper\]](https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_01A-1_Brasser_paper.pdf) [\[Code\]](https://github.com/sanctuary-js/sanctuary)
    *   DarkneTZ: Towards Model Privacy at the Edge using Trusted Execution Environments [\[Paper\]](https://arxiv.org/abs/2004.05703) [\[Code\]](https://github.com/mofanv/darknetz) 
    *   Graviton: Trusted Execution Environments on GPUs [\[Paper\]](https://www.usenix.org/conference/osdi18/presentation/volos)
    *   ShadowNet: A Secure and Efficient On-device Model Inference System for Convolutional Neural Networks [\[Paper\]](https://arxiv.org/pdf/2011.05905)
*   **Communication-based defense**  <a name="communication-based-defense"></a>
    *   ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach [\[Paper\]](https://arxiv.org/abs/2208.08569) [\[Code\]](https://github.com/Tongzhou0101/ObfuNAS)
    *   ShadowNet: A Secure and Efficient On-device Model Inference System for Convolutional Neural Networks [\[Paper\]](https://arxiv.org/pdf/2011.05905)
    *   Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware [\[Paper\]](https://arxiv.org/abs/1806.03287) [\[Code\]](https://github.com/ftramer/slalom) 
    *   Secure Outsourced Matrix Computation and Application to Neural Networks [\[Paper\]](https://eprint.iacr.org/2018/1041.pdf)
    *   NPUFort: a secure architecture of DNN accelerator against model inversion attack [\[Paper\]](https://www.semanticscholar.org/paper/NPUFort%3A-a-secure-architecture-of-DNN-accelerator-Wang-Hou/35d02c98d78abbd2b7239a44cdc920634af6926f)
    *   NeurObfuscator: A Full-stack Obfuscation Tool to Mitigate Neural Architecture Stealing [\[Paper\]](https://arxiv.org/abs/2107.09789) 
    *   NNReArch: A Tensor Program Scheduling Framework Against Neural Network Architecture Reverse Engineering [\[Paper\]](https://arxiv.org/abs/2203.12046)
*   **Model-based defense** <a name="model-based-defense"></a>
    *   MindSpore [\[Code\]](https://github.com/mindspore-ai)
    *   Defending Against Model Stealing Attacks with Adaptive Misinformation [\[Paper\]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kariyappa_Defending_Against_Model_Stealing_Attacks_With_Adaptive_Misinformation_CVPR_2020_paper.pdf) [\[Code\]](https://github.com/sanjaykariyappa/adaptive_misinformation)
    *   Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks [\[Paper\]](https://arxiv.org/abs/1906.10908) [\[Code\]](https://github.com/tribhuvanesh/prediction-poisoning)
    *   PRADA: Protecting against DNN Model Stealing Attacks [\[Paper\]](https://arxiv.org/abs/1805.02628v4) [\[Code\]](https://github.com/SSGAalto/prada-protecting-against-dnn-model-stealing-attacks)
    *   SteerAdversary [\[Paper\]](https://proceedings.mlr.press/v162/mazeika22a.html)
    *   Latent Dirichlet Allocation Model Training with Differential Privacy [\[Paper\]](https://arxiv.org/pdf/2010.04391)

---

## Contact

In case of feedback, suggestions, or issues, please contact the [Authors](https://github.com/tusharnayan10).
