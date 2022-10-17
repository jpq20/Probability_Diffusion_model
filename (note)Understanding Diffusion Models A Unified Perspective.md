# (notes)Understanding Diffusion Models: A Unified Perspective

## Introduction

* generative model:给定采样x，学习建模它的真实数据分布，之后就可以随意生成新样本
* 当前：GANs、likelihood-based、energy-based、**Score-based** 

## Background: ELBO, VAE, and Hierarchical VAE

* 可以认为观察到的数据**x**由一个相关的潜在变量**z**表示或产生
* 生成模型中倾向于学习更低维的latent：徒劳&压缩

### Evidence Lower Bound

* 联合分布p(x,z)

* idea:"likelihood-based"对观察到的全部x最大化p(x)的似然度

* 得到p(x):

  1. $$\int{p(x,z)dz}$$
  2. $$p(x) = p(x,z)/{p(z|x)}$$

* *困难：直接计算和最大化似然p(x)：积分、p(z|x)*

  *solution:推导ELBO，对它进行优化*

* $$ELBO = E_{q_{\phi}(z|x)}[log{\frac {p(x,z)}{q_\phi(z|x)}}] <= log {p(x)}$$ 

* $q_{\phi}$：一个灵活的近似可变分布，$\phi$参数化，用于估计x的潜在变量分布，试图近似真后验：$p(z|x)$

---

*为什么ELBO可作为最大化的目标？*

* 对E1使用Jensen不等式，不直观

* 对E2进行推导：

  !(images\image-20221012213721782.png)

  

* KL散度（$D_{KL}(P||Q)$）表示当用概率分布Q拟合P时产生的信息损耗：

  $$D_{KL}(P||Q) = \sum_{i\in X}P(i)*[log(\frac{P(i)}{Q(i)})]$$

* 可见log p(x) = ELBO + 近似后验q和真实后验p(z|x)之间的DL散度

* *ELBO为什么是下界？*

  *Solution:KL散度非负*

* *为什么最大化ELBO?*

  *Solution:我们希望优化近似后验q的参数来精确匹配真正的后验分布p(z|x)，则还是通过最小化KL散度来实现的*

* *困难：难以直接最小化KL散度，因为p(z|x)未知*

  *solution:log p(x)不随$\phi$变化 -> ELBO + KL = constant ->ELBO关于$\phi$最大化等效于KL散度的最小化*

* ELBO可用于观察、生成数据的似然性

### Variational Autoencoders

* VAE的默认公式中直接对ELBO进行最大化
* variational:关于$\phi$的优化是可变的过程
* antuencoders:联想传统的自动编码器模型：input经过中间步骤处理后被用于预测自身

---

分析ELBO

![image-20221012215329383](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012215329383.png)

* 编码器：$q_\phi(z|x)$，中间步骤，输入->latent
* 解码器：$p_\theta(x|z)$，latent -> 观测x
* reconstruction term:衡量解码器的重构似然程度，确保学习正在建模有效的潜在变量
* prior matching term:衡量学习到的分布和对潜在变量持有的先验观点之间的相似性，最小化该term鼓励编码器实际学习分布而非收敛为Dirac delta函数

---

*问题：如何对$\phi$、$\theta联合优化$*

solution：

* VAE的编码器被选为具有对角协方差的多元高斯模型，先验通常选为标准多元高斯：

  ![image-20221012221835490](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012221835490.png)

* 基于以上公式，我们可以解析计算ELBO的KL散度项，用蒙特卡洛方法逼近重建项：

  ![image-20221012221936407](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012221936407.png)

* latents$\{z^{(l)}\}_{l=1}^L$：对于数据集中每一个观察值x，由$q_\phi(z|x)$抽样得到

* 先插入一个新的问题

*问题：loss计算基于每一个抽样得到的$\{z^{(l)}\}_{l=1}^L$之上，导致其不可微*

*solution：当$q_\phi$被设计为某些分布（包括多元高斯）时，可以使用重参数化技巧解决*

* 重参数化：将随机变量写作关于噪声变量的确定性函数 -> 可优化非随机项

* $$x = \mu + \sigma*\epsilon ,with \ \epsilon~N(\epsilon;0,I)$$

* 用高斯分布样化噪声：任意高斯分布可通过从标准高斯采样进行标准差缩放、平均值移动得到

* VAE中：

  ![image-20221012222927507](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012222927507.png)

* 可以通过计算关于$\phi$的梯度，优化$\mu_\phi(x)$和$\sigma_\phi(x)$ 

* 可以通过重参数化和蒙特卡洛方法进行联合优化

* 训练一个VAE之后，可以通过直接从latent空间p(z)中采样来生成新的数据，然后将其通过解码器运行

---

## Hierarchical Variational Autoencoders

* 扩展到latent的多个层次：潜在变量被解释为由更高层次、更抽象的latent变量生成

**关注一种特殊情况：MHAVE：马尔可夫HAVE**

* 生成过程是马尔可夫的：生成过程是一个马尔科夫链

  $$p(x_{t+1}|x_t,...,x_1)=p(x_{t+1}|x_t)$$

* 解码每个$z_t$只需要在前一个$z_{t+1}$上进行

* 整个模型可视作VAE的堆叠、递归

  ![image-20221012224128961](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012224128961.png)

* ELBO:

  ![image-20221012224147927](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012224147927.png)

* 代入：

  ![image-20221012224215075](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012224215075.png)

## Variational Diffusion Models（VDM）

* MHVAE with 3  key restrictions：

  1. latent维数 = data维数
  2. 每个编码器的结构$(q(x_t|x_{t-1}))$在每个timestep上不学习，被预定义为线性高斯模型：是以前一个timestep的输出为中心的高斯分布
  3. latent编码器的高斯参数随时间变化，使得latent在最后一个timestep上是标准高斯分布

* 分析：

  1. ![image-20221012224749131](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012224749131.png)

  2. **与MHVAE不同的是：**latent编码器的结构在每个timestep上不学习，被固定为线性高斯模型，均值方差均可设置为超参数或可以学习的参数，我们如此设置其参数：![image-20221012225003445](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012225003445.png)，系数形式的选择可以使latent变量的方差保持在相似的尺度上。（允许不同的参数化、$\alpha_t$是可学习的，随t变化)：

     ![image-20221012225227918](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012225227918.png)

  3. $\alpha_t$随时间演化按一个固定的或可学习的时间结构进行，使最终的latent变量分布为标准高斯,可如下改写解码器：

     ![image-20221012225515656](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012225515656.png)

     现在编码器q不再依赖于参数，只需要关注$p_\theta$，来模拟新数据，采样只需要从p(xT中采样高斯噪声->运行$p_\theta$得到新的x0)

     ---

     可以通过最大化ELBO优化：

     ![image-20221012225754465](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012225754465.png)

     1. reconstruction term:给定第一步latent，预测原始数据的对数相似度，可以以与VAE相似的方法优化；

     2. prior matching term:达到最小值when最终潜在分布与高斯先验相匹配，无需优化，large enough T -> 0

     3. consistency term:使xt处的分布一致，无论是前向还是后向过程，denoising 和 noising应相互匹配 ，最小化when$p_\theta(x_t|x_{t+1})$匹配q所代表的高斯分布:

        ![image-20221012230234678](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012230234678.png)

---

上述优化式中所有项都以期望的形式存在，因此可用蒙特卡罗方法解决

*问题：直接使用蒙特卡洛是次优的方法，consistency term每一步对两个随机变量计算期望导致了high variance*

*solution:推导ELBO的另一种形式，可以将编码器重写为q(xt|xt−1)= q(xt|xt−1,x0) --基于马尔可夫性质*

* 由贝叶斯：

  ![image-20221012230622570](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012230622570.png)

* 可推导出新的ELBO：

  ![image-20221012230641321](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012230641321.png)

* 低方差

* 一个优雅的解释（每一项）：

  1. 可用蒙塔卡罗估计；
  2. 最终结果与标准高斯的KL散度；
  3. 学习$p_\theta$来逼近易处理、真实的denoising:q

---

*困难：同时学习编码器的复杂性问题，对任意后验、任意复杂的MHVAE，每个KL散度项都很难最小化*

*solution：利用高斯变换假设*

* 由贝叶斯：

  ![image-20221012231427106](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012231427106.png)

* 已知：![image-20221012231453753](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012231453753.png)

* 剩余问题：贝叶斯导出项中其余两项的形式

  * 利用编码器transition是线性高斯模型，在重参数化技巧下，![image-20221012231730472](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012231730472.png)可以写作：

    ![image-20221012231757041](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012231757041.png)

  * 设：![image-20221012232152806](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012232152806.png)

  * 于是就可以反复应用上述公式递归推导结果：

    ![image-20221012231900719](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012231900719.png)

* 于是就可以利用上述结果推导贝叶斯导出式：

  ![image-20221012232038848](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012232038848.png)

* 为使$p_\theta$尽可能接近q，我们将p也建模为高斯分布

* $\alpha$在每个时间步上固定：可以构造近似的去噪步骤：取相同的方差

* 将均值参数化为一个xt、t的函数：$\mu_\theta(x_t,t)$

* 优化问题简化为最小化两个分布的均值差：

  ![image-20221012232639639](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012232639639.png)

* $\mu_\theta$可建模如下：

  ![image-20221012232746539](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012232746539.png)

* 优化问题可简化为：

  ![image-20221012232849749](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012232849749.png)

* 优化问题可以归结为学习一个神经网络，从任意噪声版本中预测最原始的ground truth图像

* 在所有时间步上最小化期望来最小化ELBO求和项：![image-20221012233035672](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012233035672.png)

---

## Learning Diffusion Noise Parameters

*如何共同学习VDM的噪声参数？*

*solution:推导一种学习扩散噪声参数的替代方法*

* 将之前固定的方差项代入上一小节得到的单步优化式中可得：

  原式![image-20221012233448047](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012233448047.png)

* 引入了信噪比：$SNR = \frac{\mu^2}{\sigma^2}$

* 已知：q(xt|x0)有如下高斯形式：![image-20221012233922443](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012233922443.png)则有：$SNR(t) = \frac{\overline{\alpha^t}}{1-{\overline{\alpha^t}}}$

* ![image-20221012233827877](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012233827877.png)

* 扩散模型中要求信噪比随t增大而单调减小

* 用神经网络参数化每个时间步的信噪比，并与扩散模型联合学习：

  ![image-20221012234110180](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012234110180.png)

* 可进一步推导：(这些项对计算是必要的：如使用重参数化技巧从输入创建任意噪声时)

  ![image-20221012234219860](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012234219860.png)

## Three Equivalent Interpretations

1. 之前讨论的 xt -> x0

2. * rearrange: q(xt|x0)得到：![image-20221012234527629](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012234527629.png)

   * 代入之前推导:![image-20221012232746539](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012232746539.png)

     得到：

     ![image-20221012234701379](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012234701379.png)

   * 可以如下设置$\mu_\theta$如下：

     ![image-20221012234746694](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012234746694.png)

   * 优化变为：

     ![image-20221012234817796](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012234817796.png)

   * **综上，通过预测原始图像x0学习一个VDM 等效于 学习预测noise**

3. * Tweedie's Formula：对高斯分布z~N(z;uz,sigmaz)有：

     ![image-20221012235023420](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012235023420.png)

   * 已知：![image-20221012235044146](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012235044146.png)

   * 有：

     ![image-20221012235102856](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012235102856.png)

   * 因为：对产生xt真正的mean的最佳估计是：![image-20221012235534349](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012235534349.png),所以有;

     ![image-20221012235554817](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012235554817.png)

   * 重新代入$\mu_q$有：![image-20221012235629326](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221012235629326.png)

   * 如此一来，就可以将优化均值设置为：
   
     ![image-20221015161420688](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015161420688.png)
   
   * 优化问题转化为：
   
     ![image-20221015161509760](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015161509760.png)
   
   * 此处$s_\theta$是一个神经网络，学习$\bigtriangledown_{x_t}log p(x_t)$,该式表示x_t在数据空间中的梯度
   
   * $\bigtriangledown_{x_t}log p(x_t) = -\frac{1}{\sqrt{1-{\overline{\alpha_t}}}}*\epsilon_0$ 
   
   * 上述推导表明score function衡量如何在一个概率空间中移动以最大化log概率 -> 直观来说，源噪声被添加来破坏原始图像，那么向相反方向改变会“去噪”，这是增加后续log概率的最佳更新，也即学习对score function建模 = 对源噪声负值建模

## Score-based Generative Models

$$
s_\theta(x_t,t) = \bigtriangledown_{x_t}log p(x_t)
$$

* 上述推导使用tweedie's formula，并足以说明score function是什么，或者提供它值得建模、优化的直觉
* 我们可以研究另一类基于分数的生成模型，来发现这种直觉
* 可以证明之前推导的VDM公式有一个等价的基于分数的生成模型，建模方程允许我们在这两种解释中灵活切换



* 任意灵活的概率分布可写作：![image-20221015162942370](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015162942370.png)

* $f_\theta$任意、灵活，参数化的方程称为能量函数，通常由神经网络建模

* $z_\theta$则是归一化常数

* 一种学习这种分布的方法是最大似然：

  *问题：对于复杂能量函数，归一化系数并不容易处理*

  *solution：一种避免建模归一化常数的方法是——用神经网络$s_\theta$学习分布p(x)的score function*

* 分析：![image-20221015163417727](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015163417727.png)

* 神经网络$s_\theta$可以自由地表示为一个神经网络而不涉及归一化常数

* 优化：最小化

  ![image-20221015164219939](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015164219939.png)

* score function本质上描述了为进一步增加似然程度，在数据空间中应移动的方向

### Langevin dynamics

* score function在x的整个数据空间定义了一个指向modes的向量场。

* ![image-20221015163823224](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015163823224.png)

* 通过学习真实数据分布的score function，我们从同一空间的任意一点开始，沿着评分迭代生成样本，直到到达一个mode——这个过程被称为郎之万动力学，数学上描述为：

  ![image-20221015163952598](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015163952598.png)

* x0：从一个先验分布中任意采样得到

* $\epsilon $ ~ $\N(\epsilon;0,I)$：噪声项，保证样本不总是收敛到一个Mode,避免确定性轨迹

---

*问题：上述优化问题依赖于ground truth score function，对于复杂分布是不能得到的*

*solution:分数匹配*

---

Score-based Generative：学习将一个分布表示为score function并使用它、通过马尔可夫蒙特卡洛方法（如郎之万动力学）来生成样本

---

* 三个问题：
  1. score function在x处于高维空间的低维流形时是ill-defined(对数)
  2. 低密度区域
  3. may not mix，混合系数会丢失
* 解决：在数据中加入多级高斯噪声
* ![image-20221015165054568](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015165054568.png)
* 退火方法：初始化是从某个固定的先验中随机选取，后续的每个采样点都从前面模拟的最终样本开始。
* 噪声水平随着t不断下降，随着时间步长减小，样本最终收敛到真模式

---

* 扩展到无限时间步：灵活建模

## Guidance

* 之前的推导专注于建模p(x)
* 此处开始关注p(x|y)：这使我们能够显式地控制通过条件信息y生成数据
* ![image-20221015165621208](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015165621208.png)
* y可以是一个文本编码或一个待重建的低像素图片，这个过程可以用如之前讨论的VDM一样训练:
* ![image-20221015165813050](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015165813050.png)

*问题：以这种方法训练的条件扩散模型可能会学会忽略或淡化任何给定的条件信息*

*solution:Guidance——更明确地控制模型给予条件信息地权重*

### Classifier Guidance

* 基于 score-based扩散模型地公式，通过贝叶斯进行推导：

  ![image-20221015170230671](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015170230671.png)

* 最终的结果可以解释为学习一个无条件的score function以及分类器p(y|xt)的对抗性梯度相结合

* 即按照之前推导的方法学习无条件扩散模型的分数，同时学习一个接受任意噪声xt并试图预测条件信息y的分类器。

* 然后，在采样过程中，用于退火Langevin动力学的总体条件分数函数被计算为无条件分数函数和噪声分类器的对抗梯度的总和。



* 细粒度控制来鼓励或阻止模型考虑条件信息：

  ![image-20221015170523205](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015170523205.png)

### Classifier-Free Guidance

*上述模型依赖于一个单独学习的分类器*

* 这种方法放弃了单独分类器模型的训练，采用无条件扩散模型和条件扩散模型

* 对上一节公式重排代入可得：

  ![image-20221015170709419](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20221015170709419.png)

* $\gamma>1$时，进一步明确使用条件，减少多样性，精确匹配

* 由于学习两个模型的代价很高，可以将条件扩散模型和无条件扩散模型作为一个singular条件模型一起学习

* 无条件扩散模型可以通过将条件信息替换为固定的常数值(如零)来查询

