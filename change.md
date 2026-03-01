- **Sandwich option (front-run reserves sentence):**
  - Before:
        \paragraph{Front-run swap (X$\to$Y).}After the front-run swap of size $s$, they become $x_0 + s,y_0-\Delta y_{fr}$, where
  - After:
        \paragraph{Front-run swap (X$\to$Y).}After the front-run swap of size $s$, they become $x_0 + r s,\,y_0-\Delta y_{fr}$, where

- **Gas/bribe-ceiling block (made $\phi'$ usage consistent and included gas terms):**
  - Before:
           &= S f - G_{\text{JIT}} 
           = x_{0} \bigl(\sigma f - \phi' \bigr), \\
           &= x_{0} \pi^*.
  - After:
           &= S f - G_{\text{JIT}}
           = x_0\sigma f - G_{\text{JIT}}, \\
           &= x_0 \pi^* - G_{\text{sand}}.
        \]
        and therefore
        \[
        m_{\max}^{\text{JIT}} \ge m_{\max}^{\text{sand}}
        \quad\Longleftrightarrow\quad
        \sigma f - \pi^* - \phi' \ge 0,
        \qquad
        \phi'=\frac{G_{\text{JIT}}-G_{\text{sand}}}{x_0}.

- **Mixed (Sandwich + JIT) burn amounts (fixed dropped dependence + fixed minted-$Y$ baseline):**
  - Before:
            \Delta x_{burn} = \frac{\alpha}{\alpha + 1} x_3 = \frac{x_0\alpha}{\alpha + 1}
        \end{equation}
        $\alpha x_0(1+r\varepsilon) + \frac{\alpha}{\alpha +1}r \sigma x_0$ tokens X and $\alpha y_0 - \frac{\alpha}{\alpha +1}\Delta y_v$ of tokens Y.
  - After:
            \Delta x_{\mathrm{burn}}
            = \frac{\alpha}{1+\alpha}\,x_3
            = \alpha x_0(1+r\varepsilon) + \frac{\alpha}{1+\alpha}\,r\sigma x_0.
        \end{equation}
        $\alpha x_0(1+r\varepsilon) + \frac{\alpha}{1+\alpha}r \sigma x_0$ tokens $X$ and $\frac{\alpha y_0}{1+r\varepsilon} - \frac{\alpha}{1+\alpha}\Delta y_v$ tokens $Y$.

- **Mixed (Sandwich + JIT) “profit is therefore” equation (removed dimensionally-inconsistent term and rewrote profit in $X$ units):**
  - Before:
            \Pi_{sand+jit}(\varepsilon, \alpha, \sigma) = \underbrace{\bigl(\theta_{self}y_0 - \varepsilon x_0 \bigr)}_{\text{Sandwich Profit}} + \underbrace{\frac{\alpha\sigma x_0f}{\alpha + 1}}_{\text{LP fees}} + \underbrace{ \frac{x_0\alpha r \sigma}{\alpha+1}}_{\text{LP inventory change in X}}
  - After:
            \Pi_{\mathrm{sand+jit}}(\varepsilon, \alpha, \sigma)
            = \Delta x_{\mathrm{br}}(\theta_{\mathrm{self}}) - \varepsilon x_0 + \frac{\alpha}{1+\alpha}\,\sigma x_0