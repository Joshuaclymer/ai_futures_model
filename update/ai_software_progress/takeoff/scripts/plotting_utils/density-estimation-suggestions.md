The problem is just “density estimation with a *hard* lower boundary.” A vanilla Gaussian KDE assumes support on ℝ, so it leaks mass into the past and piles it up at the boundary. You want an estimator whose **support is [today, ∞)** by construction.

Here are the clean, principled options (in order of practicality + rigor):

# 1) Transform-to-ℝ KDE (recommended)

Map your arrival time (T) (in decimal years) to a variable with unbounded support, do KDE there, and transform back with the change-of-variables formula. This enforces zero density for (t< t_0).

* Let (t_0) be “today” in the same units as your samples.
* Define (Y=\log!\big((T-t_0)/\tau\big)) with some scale (\tau>0) (e.g., 0.25–1.0 years; it only affects bandwidth numerics, not support).
* Fit a standard Gaussian KDE to ({y_i}).
* Convert back:
  [
  f_T(t)=\frac{1}{t-t_0},f_Y!\left(\log!\frac{t-t_0}{\tau}\right),\quad t>t_0.
  ]
  (The (\tau) cancels out up to bandwidth choice; keeping (\tau) ≈ “1 year” is numerically stable.)

**Bandwidth:** choose it by leave-one-out cross-validated log likelihood (or likelihood CV on a grid). That’s statistically principled and usually fixes the over/under-smoothing that makes the default KDE look wrong.

**What about a few past samples?** They violate the known support, so treat them as *invalid under the data-generating process*. Drop them (or clip them to (t_0+\varepsilon) with (\varepsilon) ≈ 1 day) *before* the transform. That matches the constraint you stated: the true distribution has zero mass in the past.

**Python sketch (drop-in):**

```python
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# inputs
t0 = 2025.86  # "today" as decimal year
T = np.array(times)  # your samples
eps_days = 1.0/365.25
T = T[T >= t0]  # drop logically impossible past draws
tau = 1.0      # 1 year scale for numerical stability

Y = np.log((T - t0)/tau)

# CV bandwidth selection
grid = {'bandwidth': np.logspace(-1.5, 0.8, 40)}  # tune if needed
kde_cv = GridSearchCV(KernelDensity(kernel='gaussian'), grid,
                      scoring='neg_log_loss', cv=min(10, len(Y)))
kde_cv.fit(Y[:, None])
kde = kde_cv.best_estimator_

# Evaluate pdf on a future grid
t_grid = np.linspace(t0 + 1e-6, 2200, 2000)
y_grid = np.log((t_grid - t0)/tau)
logfy = kde.score_samples(y_grid[:, None])        # log f_Y
f_t = np.exp(logfy) / (t_grid - t0)               # change-of-variables

# (Optional) normalize numerically to kill any tiny integration error:
f_t /= np.trapz(f_t, t_grid)
```

This will **never** place mass before (t_0) and typically produces a much better-behaved shape than reflection tricks.

# 2) Boundary-aware kernels (Gamma/Beta kernels)

There’s a fully nonparametric alternative that keeps support on ([t_0,\infty)) without transforming: use **Gamma kernels** (for nonnegative support) applied to (X=T-t_0). The Chen (2000) gamma-kernel KDE adapts kernel shape near the boundary so density doesn’t “leak” left. In R it’s available off-the-shelf; in Python you’d implement or pull from niche repos. If you want the most textbook-clean estimator *on the original scale* and you’re happy to code it, this is excellent.

# 3) Parametric (or semi-parametric) survival fit with left boundary

Model (X=T-t_0) using a positive-support family (log-normal, Weibull, log-logistic, Gamma, or a **mixture of log-normals** for more flexibility). Fit by MLE; pick the model via out-of-sample log likelihood (CV). Because your tail runs to ~2200, a log-normal or log-logistic (or a 2-component log-normal mixture) often matches both the early mass and the long right tail better than a single Weibull. This is the most *interpretable* route (you get closed-form CDFs/hazards and extrapolation), but you are imposing a family.

# Why the transform-KDE is “most principled” here

* It encodes the *true support constraint* exactly.
* It retains full nonparametric flexibility.
* It uses a **likelihood-based** bandwidth criterion (CV), not ad-hoc tuning.
* It is easy to implement, robust with (n\approx 10^3), and visually stable even with long tails.

# Practical tips for your plot

* Plot the density of (X=T-t_0) on a **log x-axis** alongside the transformed-KDE pdf; the visual match to your histogram will be much clearer for long right tails.
* Overlay your P10, median, P90 as vertical lines to sanity-check the fit.
* If a handful of samples extend far right (e.g., 2200), do the CV on a **trimmed** set (e.g., remove the top 0.5–1%) and then renormalize the fitted pdf over the full grid—this reduces bandwidth inflation from a few extreme points without violating support.

# If you want an even stricter “principled” variant

Do the transform-KDE but **select bandwidth by maximizing leave-one-out *conditional* likelihood under the support**. In practice with the log transform this is the same as standard LOO CV on (Y) (because the support constraint is already built in), which is why the simple CV above is acceptable.

---

