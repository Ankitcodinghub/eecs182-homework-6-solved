# eecs182-homework-6-solved
**TO GET THIS SOLUTION VISIT:** [EECS182 Homework 6 Solved](https://www.ankitcodinghub.com/product/eecs182-solved-3/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;116347&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EECS182 Homework 6 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
1. Backprop through a Simple RNN

Consider the following 1D RNN with no nonlinearities, a 1D hidden state, and 1D inputs ut at each timestep. (Note: There is only a single parameter w, no bias). This RNN expresses unrolling the following recurrence relation, with hidden state ht at unrolling step t given by:

ht = w · (ut + ht−1) (1)

The computational graph of unrolling the RNN for three timesteps is shown below:

Figure 1: Illustrating the weight-sharing and intermediate results in the RNN.

where w is the learnable weight, u1, u2, and u3 are sequential inputs, and p, q, r, s, and t are intermediate values.

(a) Fill in the blanks for the intermediate values during the forward pass, in terms of w and the ui’s:

p = u1 q = w · u1 r = u2 + q = u2 + w · u1

s = w · r = w · u2 + w2 · u1

t =

y =

(b) Using the expression for y from the previous subpart, compute .

(d) Calculate the partial derivatives along each of the three outgoing edges from the learnable w in Figure 1, replicated below. (e.g., the right-most edge has a relevant partial derivative of t in terms of how much the output y is effected by a small change in w as it influences y through this edge. You need to compute the partial derivatives for the other two edges yourself.)

You can write your answers in terms of the p,q,r,s,t and the partial derivatives of y with respect to them.

Use these three terms to find the total derivative .

(HINT: You can use your answer to part (b) to check your work.)

2. Beam Search

This problem will also be covered in discussion.

When making predictions with an autoregressive sequence model, it can be intractable to decode the true most likely sequence of the model, as doing so would require exhaustively searching the tree of all O(MT ) possible sequences, where M is the size of our vocabulary, and T is the max length of a sequence. We could decode our sequence by greedily decoding the most likely token each timestep, and this can work to some extent, but there are no guarantees that this sequence is the actual most likely sequence of our model.

Instead, we can use beam search to limit our search to only candidate sequences that are the most likely so far. In beam search, we keep track of the k most likely predictions of our model so far. At each timestep, we expand our predictions to all of the possible expansions of these sequences after one token, and then we keep only the top k of the most likely sequences out of these. In the end, we return the most likely sequence out of our final candidate sequences. This is also not guaranteed to be the true most likely sequence, but it is usually better than the result of just greedy decoding.

The beam search procedure can be written as the following pseudocode:

Algorithm 1 Beam Search

for each time step t do for each hypothesis y1:t−1,i that we are tracking do find the top k tokens yt,i,1,…,yt,i,k end for

sort the resulting k2 length t sequences by their total log-probability store the top k

Figure 2: The numbers shown are the decoder’s log probability prediction of the current token given previous tokens.

We are running the beam search to decode a sequence of length 3 using a beam search with k = 2. Consider predictions of a decoder in Figure 2, where each node in the tree represents the next token log probability prediction of one step of the decoder conditioned on previous tokens. The vocabulary consists of two words: “neural” and “network”.

(a) At timestep 1, which sequences is beam search storing?

(b) At timestep 2, which sequences is beam search storing? (c) At timestep 3, which sequences is beam search storing?

(d) Does beam search return the overall most-likely sequence in this example? Explain why or why not.

(e) What is the runtime complexity of generating a length-T sequence with beam size k with an RNN? Answer in terms of T and k and M. (Note: an earlier version of this question said to write it in terms of just T and k. This answer is also acceptable.)

3. Implementing RNNs (and optionally, LSTMs)

This problem involves filling out this notebook.

Note that implementing the LSTM portion of this question is optional and out-of-scope for the exam.

(a) Implement Section 1A in the notebook, which constructs a vanilla RNN layer. This layer implements the function

ht = σ(Whht−1 + Wxxt + b)

where Wh, Wx, and b are learned parameter matrices, x is the input sequence, and σ is a nonlinearity such as tanh. The RNN layer “unrolls” across a sequence, passing a hidden state between timesteps and returning an array of hidden states at all timesteps.

Figure 3: Source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

Copy the outputs of the “Test Cases” code cell and paste it into your submission of the written assignment.

(b) Implement Section 1.B of the notebook, in which you’ll use this RNN layer in a regression model by adding a final linear layer on top of the RNN outputs.

yˆt = Wfht + bf

We’ll compute one prediction for each timestep.

Copy the outputs of the “Tests” code cell and paste it into your submission of the written assignment.

(c) RNNs can be used for many kinds of prediction problems, as shown below. In this notebook we will look at many-to-one prediction and aligned many-to-many prediction.

We will use a simple averaging task. The input X consists of a sequence of numbers, and the label y is a running average of all numbers seen so far.

We will consider two tasks with this dataset:

• Task 1: predict the running average at all timesteps

• Task 2: predict the average at the last timestep only

Implement Section 1.C in the notebook, in which you’ll look at the synthetic dataset shown and implement a loss function for the two problem variants.

Copy the outputs of the “Tests” code cell and paste it into your submission of the written assignment.

Figure 4: Image source: https://calvinfeng.gitbook.io/machine-learning-notebook/ supervised-learning/recurrent-neural-network/recurrent_neural_networks

(d) Consider an RNN which outputs a single prediction at timestep T. As shown in Figure 4, each weight matrix W influences the loss by multiple paths. As a result, the gradient is also summed over multiple paths:

(2)

When you backpropagate a loss through many timesteps, the later terms in this sum often end up with either very small or very large magnitude – called vanishing or exploding gradients respectively. Either problem can make learning with long sequences difficult.

Implement Notebook Section 1.D, which plots the magnitude at each timestep of . Play around with this visualization tool and try to generate exploding and vanishing gradients. Include a screenshot of your visualization in the written assignment submission.

(e) If the network has no nonlinearities, under what conditions would you expect the exploding or vanishing gradients with for long sequences? Why? (Hint: it might be helpful to write out the formula for and analyze how this changes with different t). Do you see this pattern empirically using the visualization tool in Section 1.D in the notebook with last_step_only=True?

(f) Compare the magnitude of hidden states and gradients when using ReLU and tanh nonlinearities in Section 1.D in the notebook. Which activation results in more vanishing and exploding gradients? Why? (This does not have to be a rigorous mathematical explanation.)

(g) What happens if you set last_target_only = False in Section 1.D in the notebook? Explain why this change affects vanishing gradients. Does it help the network’s ability to learn dependencies across long sequences? (The explanation can be intuitive, not mathematically rigorous.)

(h) (Optional) Implement Section 1.8 of the notebook in which you implement a LSTM layer. LSTMs pass a cell state between timesteps as well as a hidden state. Explore gradient magnitudes using the visualization tool you implemented earlier and report on the results.

Figure 5: Image source: https://colah.github.io/posts/2015-08-Understanding-LSTMs/ The LSTM forward pass is shown below:

ft = σ(xtUf + ht−1Wf + bf) it = σ(xtUi + ht−1Wi + bi) ot = σ(xtUo + ht−1Wo + bo)

C˜t = tanh(xtUg + ht−1Wg + bg)

Ct = ft ◦ Ct−1 + it ◦ C˜t ht = tanh(Ct) ◦ ot

where ◦ represents the Hadamard Product (elementwise multiplication) and σ is the sigmoid function.

(i) (Optional) When using an LSTM, you should still see vanishing gradients, but the gradients should vanish less quickly. Interpret why this might happen by considering gradients of the loss with respect to the cell state. (Hint: consider computing using the terms ∂L,∂CT ,∂CT−1,∂hT ,∂hT−1).

(j) (Optional)

Consider a ResNet with simple resblocks defined by ht+1 = σ(Wtht + bt) + ht. Draw a connection between the role of a ResNet’s skip connections and the LSTM’s cell state in facilitating gradient propagation through the network.

(k) (Optional) We can create multi-layer recurrent networks by stacking layers as shown in Figure 6. The hidden state outputs from one layer become the inputs to the layer above.

Figure 6: Image source: https://calvinfeng.gitbook.io/machine-learning-notebook/ supervised-learning/recurrent-neural-network/recurrent_neural_networks

Implement notebook Section 1.K and run the last cell to train your network. You should be able to reach training loss &lt; 0.001 for the 2-layer networks, and &lt;.01 for the 1-layer networks.

4. RNNs for Last Name Classification

Please follow the instructions in this notebook. You will train a neural network to predict the probable language of origin for a given last name / family name in Latin alphabets. Once you finished with the notebook, download submission_log.json and submit it to “Homework 6 (Code)” in Gradescope.

(a) Although the neural network you have trained is intended to predict the language of origin for a given last name, it could potentially be misused. In what ways do you think this could be problematic in real-world applications?

5. Read a Blog Post: How to train your Resnet

In previous homeworks, we saw how memory and compute constraints on GPUs put limits on the architecture and the hyperparameters (e.g., batch size) we can use to train our models. To train better models, we could scale up by using multiple GPUs, but most distributed training techniques scale sub-linearly and often we simply don’t have as many GPU resources at our disposal. This raises a natural question – how can we make model training more efficient on a single GPU?

The blog series How to train your Resnet (https://myrtle.ai/learn/how-to-train-your-resnet/) explores how to train ResNet models efficiently on a single GPU. It covers a range of topics, including architecture, weight decay, batch normalization, and hyperparameter tuning. In doing so, it provides valuable insights into the training dynamics of neural networks and offers lessons that can be applied in other settings.

Read the blog series and answer the questions below.

(a) What is the baseline training time and accuracy the authors started with? What was the final training time and accuracy achieved by the authors?

(b) Comment on what you have learnt. (≈ 100 words)

(c) Which approach taken by the authors interested you the most? Why? (≈ 100 words)

6. Convolutional Networks

Note: Throughout this problem, we will use the convention of NOT flipping the filter before dragging it over the signal. This is the standard notation with neural networks (ie, we assume the filter given to us is already flipped)

(a) List two reasons we typically prefer convolutional layers instead of fully connected layers when working with image data.

(b) Consider the following 1D signal: [1,4,0,−2,3]. After convolution with a length-3 filter, no padding, stride=1, we get the following sequence: [−2,2,11]. What was the filter?

(Hint: Just to help you check your work, the first entry in the filter that you should find is 2. However, if you try to use this hint directly to solve for the answer, you will not get credit since this hint only exists to help you check your work.)

(c) Transpose convolution is an operation to help us upsample a signal (increase the resolution). For example, if our original signal were [a,b,c] and we perform transpose convolution with pad=0 and stride=2, with the filter [x,y,z], the output would be [ax,ay,az+bx,by,bz+cx,cy,cz]. Notice that the entries of the input are multipled by each of the entries of the filter. Overlaps are summed. Also notice how for a fixed filtersize and stride, the dimensions of the input and output are swapped compared to standard convolution. (For example, if we did standard convolution on a length-7 sequence with filtersize of 3 and stride=2, we would output a length-3 sequence).

If our 2D input is and the 2D filter is What is the output of transpose convo-

lution with pad=0 and stride=1?

7. Homework Process and Study Group

We also want to understand what resources you find helpful and how much time homework is taking, so we can change things in the future if possible.

(a) What sources (if any) did you use as you worked through the homework?

(b) If you worked with someone on this homework, who did you work with?

List names and student ID’s. (In case of homework party, you can also just describe the group.)

(c) Roughly how many total hours did you work on this homework? Write it down here where you’ll need to remember it for the self-grade form.

Contributors:
