# Inferring details from the paper
Some details that were not clear from the paper and these are my attempts at inferring them.

## Figuring out `max_iterations`
The paper does not explicitly state the number of training epochs (`max_iterations`) used across all experiments. 
Here's the analysis to determine the likely values:

### Evidence from Different Sources
1. **Original Repository**
   - Default value is set to 3000 iterations

2. **Humanoid Task**
   - Paper results (Appendix G.1 Example 2): Score of ~3-4 during iterations 2-3
   - Our tests: 
     - With 3000 iterations: Score of 7-8 so it makes this setting unlikely

   
3. **ShadowHand Task (Appendix G.1 Example 1)**
   - Paper results: Score of 9-10 on iterations 1-2
   - Our tests:
     - With 20k iterations: Score of ~10
     - With 3k iterations: Score of ~1

4. **Dexterity Results**
   - Comparing training curves in Figure 10 (3k iterations) vs Figure 5
   - Final scores don't align, suggesting default 6k epochs were used instead of 3k

### Conclusion
More likely to be default rather than fixed 3000.

## Averaging over runs
Seemingly confusing instructions:
```
"we run 5 independent PPO training runs and report the average
of the maximum task metric values achieved from 10 policy checkpoints sampled at fixed intervals.
In particular, the maximum is taken over the same number of checkpoints for each approach" 
```
vs

```
"The final EUREKA reward, like all other baseline reward
functions, are evaluated using 5 PPO runs with the average performance on the task fitness function
F as the reward performance."
```

From the code, 10 checkpoint averaging happens for training (`epoch_freq`) but eval takes max over entire timeseries.

## Normalizing outliers

```
Given that there are several significant outliers in human
normalized score when EUREKA is substantially better both Human and Sparse on a task, when
reporting the average normalized improvement in our abstract, we adjust the score so that the
normalized score must lie between [0, 3] per task before computing the average over all 29 tasks.
```
This is why FrankaCabinet's score of 12 is compressed in the figures of the paper