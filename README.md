# Neural Network from Scratch

Given a training data $x,y$ where $x \in \mathbb{R}^{n_{x}}$ is an input matrix and $y \in \{0, 1 \}$ contains labels and $m$ is total training examples, we can establish a linear relation ship as

$$
\hat{y} = w^{T} \cdot x + b
$$

where $w \in \mathbb{R}^{n_{x}}, b \in \mathbb{R}$ are parameters most commonly known as **weights** and **biases** of a neural network. We can introduce non-linearity by using a "activation" function $\sigma$ which can be either *sigmoid*, Rectified Linear Unit *ReLU*, *tanh*. The function above becomes

$$
\hat{y} = \sigma \left(w^{T} \cdot x + b \right)
$$

We will use log-likelihood as loss (error) function.

$$
\mathcal{L}(\hat{y} - y) = - \left[ y log \hat{y} + (1 - y) log (1 - \hat{y}) \right]
$$

Here, if $y = 1 \rightarrow \mathcal{L}(\hat{y} - y) = -log \hat{y}$, we want $log \hat{y}$ to be large which imply $\hat{y}$ should be large and by sigmoid function, $\hat{y} \approx 1$. Also, if $y = 0 \rightarrow \mathcal{L}(\hat{y} - y) = -log (1- \hat{y})$, we want $(1- log \hat{y})$ to be large which imply $\hat{y}$ should be small and by sigmoid function, $\hat{y} \approx 0$. This is another reason why we use log-likelihood instead of SSD. Similarly, we have cost function as 

$$
\mathcal{J}(w,b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y_{i} log \hat{y}_{i} + (1 - y_{i}) log (1 - \hat{y}_{i}) \right]
$$

We want to find $w, b$ such that $\mathcal{J}(w,b)$ is minimized. So we use gradient descent algorithm with learning rate $\alpha$ to update $w,b$ iteratively.

$$
w := w - \alpha \frac{\partial \mathcal{J}(w,b)}{\partial w}
$$

$$
b := b - \alpha \frac{\partial \mathcal{J}(w,b)}{\partial b}
$$

Up to this we can see this as a logistic regression problem. But a neural network can have a lot of hidden layers which have these logistic regression like architecture per hidden layer.

Suppose we have $m$ $X, Y$ training examples, $L$-hidden layers. For a $l^{th}$ hidden layer, we can generalize forward propagation as follow:

$$
Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}
$$

$$
A^{[l]} = \sigma^{[l]}(Z^{[l]})
$$

where $W$ are $(n^{[l]}, n^{[l-1]})$ weight matrices ($n^{[l]}$ is number of hidden units in $l^{th}$-layer), $b, A, Z$ are $(n^{[l]}, 1)$ bias, activation and linear vectors and $A^{[0]} = X$, $A^{[L]} = \hat{Y}$. For backward propagation, we can use similar generalization.

$$
dZ^{[l]} = dA^{[l]} * {\sigma'}^{[l]} (Z^{[l]})
$$

$$
dW^{[l]} = \frac{1}{m} dZ^{[l]} \cdot {A^{[l-1]}}^{T}
$$

$$
db = \frac{1}{m} \sum dZ^{[l]}
$$

$$
dA^{[l-1]} = {W^{[l]}}^{T} \cdot dZ^{[l]}
$$