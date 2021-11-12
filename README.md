Solving ill-posed inverse problems using iterative deep neural networks
=======================================================================

This repository will contain the code for the article "Solving ill-posed inverse problems using
iterative deep neural networks" [published on arXiv](https://arxiv.org/abs/1704.04058).

Contents
--------
The code contains the following

* Training using ellipse phantoms(训练，按某种分布随机生成椭圆)
* Evaluation on ellipse phantoms(评估泛化能力，使用Shepp-Logan phantom)
* Training using anthropomorphic data. (Data not included for legal/privacy reasons)
* (head切面数据采用实际临床数据，CT的参数和噪声为人为构造的。)
* Evaluation on example head slice
* Reference reconstructions of the above using [ODL](https://github.com/odlgroup/odl).

Dependencies
------------
The code is currently based on an experimental version of [ODL](https://github.com/odlgroup/odl/pull/972), so that specific branch needs to be used for the code to work.

Contact
-------
[Jonas Adler](https://www.kth.se/profile/jonasadl), PhD student  
KTH, Royal Institute of Technology  
Elekta Instrument AB  
jonasadl@kth.se

[Ozan Öktem](https://www.kth.se/profile/ozan), Associate Professor  
KTH, Royal Institute of Technology  
ozan@kth.se

Funding
-------
Development is financially supported by the Swedish Foundation for Strategic Research as part of the project "Low complexity image reconstruction in medical imaging" and "3D reconstruction with simulated forward models".

Development has also been financed by [Elekta](https://www.elekta.com/).

Haibo Li
--------
Solving ill-posed problems based on ODL(2021.8.27)