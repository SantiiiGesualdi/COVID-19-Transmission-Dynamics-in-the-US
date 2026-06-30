
# COVID-19 Transmission Dynamics in the US

![WIP](https://img.shields.io/badge/state-in_progress-red?style=for-the-badge) 
![Julia](https://img.shields.io/badge/julia-v1.12.6-9558B2?logo=julia&style=for-the-badge)


## 📝 Summary
Analyzing the transmission dynamics of COVID-19 at the intra-state level in the U.S. using Neural Networks to estimate key parameters in epidemiological differential equation models.

## 🎯 Objectives
* **Test** a SEIRVD (Susceptible-Exposed-Infectious-Recovered-Vaccinated-Deceased) model approximation for SARS-CoV-2.
* **Validate** the approach and solve it using a Universal Differential Equations (UDE) implementation.

---

## 🧮 Parameters and Definitions

Below is the dictionary of the state variables and parameters used in the model:

| Variable / Parameter | Definition |
| :--- | :--- |
| **$S$** | Susceptible population |
| **$E$** | Exposed population |
| **$I$** | Infectious population |
| **$R$** | Recovered population |
| **$V$** | Vaccinated population |
| **$D$** | Deceased population |
| **$C$** | Cumulative cases |
| **$\beta_i$**, **$\beta_e$** | Transmission rates from Infectious and Exposed individuals, respectively |
| **$\tau$** | Time delay parameter |
| **$\rho$** | Rate of new exposures (force of infection) |
| **$v$** | Vaccination rate |
| **$\omega$** | Rate of waning immunity (Recovered to Susceptible) |
| **$\eta$** | Rate of waning vaccine efficacy (Vaccinated to Susceptible) |
| **$\phi_E$** | Proportion of exposed individuals moving directly to recovery/asymptomatic |
| **$\phi_R$** | Recovery rate from the infectious compartment |
| **$\phi_D$** | Disease-induced death rate |

---

## 🔢 Delay Differential Equations System

The force of infection and its delayed counterpart are defined as

$$
\begin{aligned}
    \rho &= S \cdot (\beta_i I + \beta_e E) \\
     \rho_\tau &= S_\tau \cdot (\beta_i^\tau I_\tau + \beta_e^\tau E_\tau)
\end{aligned}
$$

*Note: $`\tau`$ indicates the time delay, looking at the previous instant $`t - \tau`$.*

The full SEIRVD delay differential equation (DDE) system is defined as follow

$$
\begin{aligned}
    \frac{dS}{dt} = -\rho - v S + \omega R + \eta V &\quad \frac{dV}{dt} &= v S - \eta V \\
    \frac{dE}{dt} = \rho - \rho_\tau &\quad \frac{dD}{dt} &= \phi_D I \\
    \frac{dI}{dt} = (1 - \phi_E)\rho_\tau - (\phi_R + \phi_D)I &\quad \frac{dC}{dt} &= (1 - \phi_E)\rho_\tau \\
    \frac{dR}{dt} = \phi_E \rho_\tau + \phi_R I - \omega R
\end{aligned}
$$
