tf_efficientdet_d?_ap


```
Algorithm 1: Det-AdvProp

**Input:** Object detection dataset D

**Output:** Learned network parameter θ

**for each training epoch do:**

  1. Sample a random batch `{x_i, (y_i, b_i)}` from dataset D
  2. Generate adversarial example `x_i^{cls}` based on classification loss L_cls(x_i, y_i) using auxiliary batchnorm
  3. Generate adversarial example `x_i^{loc}` based on localization loss L_loc(x_i, b_i) using auxiliary batchnorm
  4. Select final adversarial example `x_i` based on Equation (5) (replace with actual equation if available)
  5. Compute detection loss L_det(x_i, (y_i, b_i)) with main batchnorm
  6. Compute detection loss L_det(x_i^{adv}, (y_i, b_i)) with auxiliary batchnorm
  7. Perform a step of gradient descent w.r.t. θ:
     min( L_det(x_i, y_i, b_i) + L_det(x_i^{adv}, y_i, b_i) )
  8. **end for**
  ```
