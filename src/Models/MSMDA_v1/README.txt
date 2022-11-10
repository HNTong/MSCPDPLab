% %
Created by: Zhiqiang Li

The code for the paper of
Zhiqiang Li, Xiao-Yuan Jing, Xiaoke Zhu, Hongyu Zhang, Baowen Xu and Shi Ying. 
"On the Multiple Sources and Privacy Preservation Issues for Heterogeneous Defect Prediction" 
which is submitted to the Joural of IEEE Transactions on Software Engineering (Under Review).

%%%
an example of MSMDA to predict CM1 project by using other heterogeneous source projects

Running demo_MSMDA 

MATLAB R2014a, 64bit operating system (the liblinear just only provides ".mexw64" files, you can remake these files according to the "\liblinear\README" file)
%%%


data.mat contains the data, name and the randmarks of 28 projects.


SRDO.m: Sparse Representation based Double Obfuscation algorithm, which requires l1_ls technique to solve l1-regularized optimization problem (or others l1-regularized optimization techniques).
The l1_ls algorithm can be downloaded from the URL of https://web.stanford.edu/~boyd/l1_ls/.
@article{kim2007an,
	title={An Interior-Point Method for Large-Scale l1-Regularized Least Squares},
	author={Kim, Seung Jean and Koh, K. and Lustig, M. and Boyd, S.},
	journal={IEEE Journal of Selected Topics in Signal Processing},
	volume={1},
	number={4},
	pages={606-617},
	year={2007}
}


Note we use LR classifier from LIBLINEAR, which is a library for
large-scale regularized linear classification and regression
(http://www.csie.ntu.edu.tw/~cjlin/liblinear). It is very easy to use
as the usage and the way of specifying parameters are the same as that
of LIBLINEAR.
@article{fan2008liblinear,
	title={LIBLINEAR: A library for large linear classification},
	author={Fan, Rong-En and Chang, Kai-Wei and Hsieh, Cho-Jui and Wang, Xiang-Rui and Lin, Chih-Jen},
	journal={The Journal of Machine Learning Research},
	volume={9},
	pages={1871--1874},
	year={2008},
	publisher={JMLR. org}
}


%%% NOTE %%%
The software is free for academic use only, and shall not be used, rewritten, or adapted as the basis of a commercial product without first obtaining permission from the authors. The authors make no representations about the suitability of this software for any purpose. 
It is provided "as is" without express or implied warranty.

