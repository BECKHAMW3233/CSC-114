# Module 4 — Assess: Explain What You Built

**CSC-114 Artificial Intelligence I**  
**Student:** William Edward Beckham III  
**Option:** B — House Prices (Regression)

---

## Part 1 — Your Run

**Q1.** Which option did you build, and what one change did you make?

I built Option B — house prices regression. If I were to make a change it would be to bring in more datasets for the area as well as expand the housing area to increase the sample sizes for all the training and benchmarking data.

**Q2. 🖊** Looking at your curve, at which epoch does the validation line stop improving?

Around epoch 130 or so it became more of an average than a stabilization — it still jumped around at times but within a variation that was predictable. The image produced shows the actual data and gave an intersect line of where it was the best average.

**Q3. 🖊** What is the model doing wrong after that turnaround point?

Past that point the model starts overfitting — it seemed to have more variation in the benchmark points from training and introduced slight overfitting from what I could tell. It stops learning patterns that generalize and starts memorizing the training data, so validation performance stops improving or gets noisier while training loss keeps falling.

---

## Part 2 — Working With Your Agent

**Q4. 🖊** Describe one moment you corrected or pushed back on your agent.

My agent functioned as programmed and did not cause me to have to push back, but from the start I had to carefully phrase my prompts to allow for the agent to work with them using the available data in the book and from the proper chapters it was referencing. It would not create new code but could only use existing code from the chapters, so I had to work within those constraints from the beginning.

**Q5.** Name one thing your agent did well that saved you time.

The agent was straightforward but also restrained in what it could do, so I had to use Claude.ai to help create the dependencies py file to build the environment — that was out of scope for the book-based agent to answer. Using Claude.ai to create a py file that could build my environment for me saved so much time when attempting to run the code to train the model, instead of running it, seeing what was missing, and pip installing everything separately and hoping I got it right.

---

## Part 3 — Why Your Settings Are the Right Ones

**Q6. 🖊** Why does the last layer have no activation? Why MAE instead of accuracy? Why normalize using training stats only?

No activation on the last layer because you're predicting a price which can be any number — putting an activation on it would cap or restrict the output which doesn't work for regression.

MAE over accuracy because accuracy is just right or wrong and you're never going to hit an exact price so it's useless here. MAE tells you how far off you actually were on average which means something. I saw this directly in my EMNIST project — the original models hit 88% accuracy on the benchmark but completely fell apart on real handwriting because they had learned stroke patterns instead of actual character shapes. Accuracy said 88% and looked fine but the model was broken. It wasn't until I tested against real images that the failure showed up, and the gap between benchmark performance and real world performance is exactly what accuracy alone can't show you.

Training stats only for normalization because if you use all your data to calculate the mean and std you're letting the model see the validation and test sets before evaluation which makes your numbers look better than reality. I learned this the hard way — my EMNIST models were getting [0,1] normalized input but were trained expecting [-1,1]. Fixing that one normalization bug took me from 0% real world accuracy to 54% in a single change. Wrong normalization stats applied at the wrong stage breaks everything, so you calculate from training only and apply those same numbers everywhere else to keep evaluation honest.

**Q7. 🖊** Using your own training curve as evidence, explain why "more epochs = better" isn't true.

Past 130 epochs the validation MAE stopped dropping and just bounced around in the same range, so every epoch after that was the model spinning its wheels on the same 480 training samples without getting any better at predicting prices it hadn't seen. More epochs just meant more noise, not more accuracy. After 130 epochs it just started to jump around more and more — more epochs just means it learns the same data more and learns patterns instead of how to properly predict and understand the data.

**Q8. 🖊** How much do you genuinely understand versus trust your agent on?

Using what I've already learned from building my own EMNIST project and then finding that training performance is not the same as real world ready, this model was simpler by far but also used an extremely tiny dataset to train on so it really was basic in comparison. The math behind why MSE penalizes large errors harder than MAE I take on trust rather than working through myself — I understood what it does and why to use it, but if you asked me to derive it from scratch I'd be looking it up.

**Q9. 🖊** Explain your model to a classmate in three sentences.

This model takes eight features from the 1990 California housing census — including median income, population, total rooms, and number of households — and predicts the median home price for a district. It learns by running the data through two hidden layers of 64 neurons each, adjusting its weights each epoch to minimize the difference between predicted and actual prices. K-fold cross-validation is used to make sure the model is actually learning to generalize and not just memorizing the small 480-sample training set.

---

## Submission

- Apply PR: `CSC-114/module_4/Apply_Classification_&_Regression/`
- Training curve: `validation_mae_curve.png` (attached)
