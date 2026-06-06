# Spiking Liquid State Machine (LSM) Architecture for Context-Sensitive Multiple-Counting ($a^n b^n c^n$)

## Executive Summary
Standard Liquid State Machines (LSMs) and homogeneous Recurrent Neural Networks (RNNs) suffer from exponential fading memory, governed by membrane and synaptic time constants ($h(t) \propto e^{-t / \tau_m}$). This constraint limits their applicability to regular languages ($a^* b^* c^*$) and prevents them from tracking long-range, context-sensitive grammar sequences such as $a^n b^n c^n$. 

This document details the theoretical architectural plan for a modular Spiking Neural Network (SNN) that resolves these limitations. By integrating **Temporal Excitation Partitioned Reservoir Ensembles (TEPRE)**, **Multi-Scale Time Constants**, **Predictive Coding Loops (RLSM)**, and **Adaptive Spiking Gating**, the architecture isolates sequential computational phases. Crucially, the fading memory barrier is bypassed via two complementary non-fading physical mechanisms: **Calcium-Activated Persistent Currents ($I_{\text{CAN}}$)** supporting **Neural Line Attractors (Odometer Chains)**, and **Tsodyks-Markram Short-Term Synaptic Plasticity (STSP)** serving as a silent, energy-efficient synaptic working memory. The network is optimized via a localized **Three-Factor Dopamine-Modulated Spike-Timing-Dependent Plasticity (D-STDP)** protocol, completely avoiding the need for Backpropagation Through Time (BPTT).

---

## 1. Advanced LSM Principles & Modular Architecture

### 1.1 Temporal Excitation Partitioned Reservoir Ensembles (TEPRE)
To prevent interference between counting and matching states, the single homogeneous reservoir of a traditional LSM is replaced by a segregated modular topology consisting of four distinct reservoirs, collectively termed the **Temporal Excitation Partitioned Reservoir Ensembles (TEPRE)**:

```
                    +------------------------------------+
                    |             Gating Hub             |
                    |            (Reservoir G)           |
                    +----+-------------+-------------+---+
                         |             |             |
                         v             v             v
      [Input a/b/c] --> Gate A        Gate B        Gate C
                         |             |             |
                         v             v             v
                    +-----------+  +-----------+  +-----------+
                    | Reservoir |  | Reservoir |  | Reservoir |
                    |     A     |  |     B     |  |     C     |
                    | (Count a) |  | (Count b) |  | (Match c) |
                    +-----+-----+  +-----+-----+  +-----+-----+
                          |              |              ^
                          |              v              |
                          +------------->+--------------+
                             Feedforward State Transfer
```

1. **Reservoir A ($\mathcal{R}_A$) - The Accumulator**:
   - **Role**: Integrates input token $a$ and maintains a stable state representation of the total count $n_a$.
   - **Connectivity**: Local recurrent connections with high excitation-to-inhibition ratio ($80\%$ excitatory, $20\%$ inhibitory).
2. **Reservoir B ($\mathcal{R}_B$) - The Comparator-Accumulator**:
   - **Role**: Integrates input token $b$ to count $n_b$, while receiving topographic feedforward projections from $\mathcal{R}_A$.
   - **Connectivity**: Receives slow feedforward NMDA connections from $\mathcal{R}_A$.
3. **Reservoir C ($\mathcal{R}_C$) - The Matcher-Predictor**:
   - **Role**: Activated during the arrival of token $c$. It decrements the sustained representations from $\mathcal{R}_A$ and $\mathcal{R}_B$ to verify the matching condition $n_c = n_b = n_a$.
4. **Reservoir G ($\mathcal{R}_G$) - The Gating Hub / Control Reservoir**:
   - **Role**: A sparse, high-speed control module that detects transition tokens (e.g., $a \to b$, $b \to c$) and coordinates information flow by actively suppressing or releasing reservoirs via shunting inhibition.

### 1.2 Multi-Scale Time Constants
We establish a temporal hierarchy across the TEPRE modules to balance rapid input detection with ultra-long state retention:

| Reservoir Partition | Primary Function | Membrane $\tau_m$ | AMPA / $\text{GABA}_A$ $\tau_s$ | NMDA / Slow $\tau_s$ |
|---|---|---|---|---|
| **Control ($\mathcal{R}_G$)** | High-speed transition detection | $5 - 10\text{ ms}$ | $2 - 4\text{ ms}$ | N/A (Fast response) |
| **Counting Cores ($\mathcal{R}_A, \mathcal{R}_B, \mathcal{R}_C$)** | Synaptic integration & tracking | $20 - 40\text{ ms}$ | $5 - 10\text{ ms}$ | $150 - 300\text{ ms}$ |
| **Memory Keepers ($\mathcal{R}_{\text{mem}}$)** | Long-term state preservation | $200 - 500\text{ ms}$ | $10\text{ ms}$ | $1000 - 2000\text{ ms}$ |

Slow NMDA channels in the feedback loops of $\mathcal{R}_{\text{mem}}$ maintain depolarizing currents that extend the network's intrinsic memory retention window by several orders of magnitude without requiring continuous sensory inputs.

### 1.3 Predictive Coding & Recurrent LSM (RLSM)
The architecture incorporates predictive closed-loops (RLSM). At each step, a prediction error population ($\mathcal{P}_E$) evaluates the mismatch between the predicted token $\hat{x}_{t|t-1}$ generated by the readout layer and the actual incoming token $x_t$:

$$E_{\text{pred}}(t) = x_t - \hat{x}_{t|t-1}$$

- **Predictive Matching**: While the sequence conforms to the grammar ($a^n$ continuing as $a^n$), $\mathcal{P}_E$ remains silent due to balanced excitatory prediction and inhibitory sensory cancellation.
- **State Transition Signaling**: When the first $b$ token arrives, the mismatch between the prediction $\hat{x} = a$ and the input $x = b$ causes $\mathcal{P}_E$ to generate a localized burst of error spikes. These spikes are projected directly to the control reservoir $\mathcal{R}_G$, triggering the immediate transition of the gating signals.

### 1.4 Adaptive Spiking Gating
Gating of information flow is achieved through **shunting inhibition**. To resolve the mutual exclusion conflict between shunting inhibition (gating) and persistent activity ($I_{\text{CAN}}$), shunting is applied selectively to the feedforward output projection synapses connecting the reservoirs and the sensory input pathways, rather than shunting the reservoirs' internal recurrent loops. This configuration enables the internal recurrent persistence to preserve and maintain the count state undisturbed, while preventing the gated-off reservoir from transmitting signals downstream or receiving external inputs.

Let $z_k(t)$ represent the activation state of the gating population for reservoir $k \in \{A, B, C\}$, where $z_k(t) \in [0, 1]$.
The gating current $I_{\text{gate}, i}^{(k)}(t)$ injected into a projection neuron $i$ within reservoir $k$ is given by:

$$I_{\text{gate}, i}^{(k)}(t) = - g_{\text{gate}} \cdot z_k(t) \cdot (V_i^{(k)}(t) - E_{\text{inh}})$$

where $E_{\text{inh}} \approx -75\text{ mV}$ is the chloride reversal potential.
- When $z_k(t) \to 1$, the gating conductance $g_{\text{gate}}$ dominates, clamping the membrane potential of the projection neurons near $E_{\text{inh}}$ and silencing the output of reservoir $k$.
- For internal recurrent neurons not involved in feedforward projection or input receipt, $g_{\text{gate}} = 0$, ensuring the persistent current $I_{\text{CAN}}$ can maintain sustained activity.
- The gating variables $z_k(t)$ are dynamically driven by the control reservoir $\mathcal{R}_G$. To correct the dimensional mismatch and enforce the $z_k(t) \in [0, 1]$ bound, a saturation term is added and the input spike train sum is scaled by $\tau_g$:

$$\tau_g \frac{dz_k(t)}{dt} = -z_k(t) + (1 - z_k(t)) \tau_g \sum_{p \in \mathcal{R}_G} W_{kp}^{\text{gate}} S_p(t)$$

where $S_p(t) = \sum_f \delta(t - t_p^f)$ is the spike train of control neuron $p$.

- **Propagation Delay Mitigation**: To address the propagation delay race condition during phase transitions (e.g., $a^n \to b^n$ or $b^n \to c^n$), sensory input pathways are routed through transient delay lines/buffers ($\tau_{\text{delay}} \approx 20-30\text{ ms}$). Alternatively, the gating hub $\mathcal{R}_G$ uses fast feedforward shunting pathways that bypass the slow integration time constants, ensuring the downstream gate is fully open before sensory spikes arrive.

---

## 2. Mathematical Mechanisms to Overcome Fading Memory

To track context-sensitive grammar states over arbitrary delay lengths, the network utilizes persistent neural activity and silent synaptic memory.

#### 2.1 Leaky Integrate-and-Fire with Calcium-Activated Current ($I_{\text{CAN}}$)
We model the membrane potential $V_i^{(k)}(t)$ of the neurons in reservoir $k$ using a Leaky Integrate-and-Fire (LIF) formulation augmented by a slow, calcium-activated non-selective cationic current ($I_{\text{CAN}}$) and the gating shunting current $I_{\text{gate}, i}^{(k)}(t)$:

$$C_m \frac{dV_i^{(k)}(t)}{dt} = -g_L (V_i^{(k)}(t) - E_L) - I_{\text{syn}, i}(t) + I_{\text{CAN}, i}(t) + I_{\text{gate}, i}^{(k)}(t)$$

where $C_m$ is the membrane capacitance, $g_L$ is the leak conductance, and $E_L$ is the leak reversal potential. The total synaptic current is:

$$I_{\text{syn}, i}(t) = g_{\text{AMPA}, i}(t)(V_i^{(k)}(t) - E_{\text{AMPA}}) + g_{\text{NMDA}, i}(t) \cdot H(V_i^{(k)}) \cdot (V_i^{(k)}(t) - E_{\text{NMDA}}) + g_{\text{GABA}, i}(t)(V_i^{(k)}(t) - E_{\text{GABA}})$$

where $H(V_i^{(k)}) = \frac{1}{1 + e^{-0.062 V_i^{(k)}(t)} / 3.57}$ represents the magnesium block voltage dependency of the NMDA channel.

To ensure that $I_{\text{CAN}, i}(t)$ acts as a depolarizing inward current, we define its driving force with the reversal potential $E_{\text{CAN}}$ minus the membrane potential $V_i^{(k)}(t)$:

$$I_{\text{CAN}, i}(t) = g_{\text{CAN}} \cdot m_i(t) \cdot (E_{\text{CAN}} - V_i^{(k)}(t))$$

where $E_{\text{CAN}} \approx 0\text{ mV}$ is the reversal potential of the cationic channel. Since the sub-threshold membrane potential $V_i^{(k)}(t)$ resides in negative ranges (e.g., $-70\text{ mV}$ to $-50\text{ mV}$), the term $(E_{\text{CAN}} - V_i^{(k)}(t))$ is strictly positive, thereby generating a depolarizing (inward) current that sustains persistent firing.

The gating variable $m_i(t)$ is regulated by intracellular calcium concentration $[\text{Ca}^{2+}]_i$:

$$\frac{dm_i(t)}{dt} = \frac{m_\infty([\text{Ca}^{2+}]_i) - m_i(t)}{\tau_{\text{CAN}}}$$

$$m_\infty([\text{Ca}^{2+}]_i) = \frac{[\text{Ca}^{2+}]_i}{K_d + [\text{Ca}^{2+}]_i}$$

The calcium dynamics are defined by influx per spike and slow decay:

$$\frac{d[\text{Ca}^{2+}]_i}{dt} = -\frac{[\text{Ca}^{2+}]_i}{\tau_{\text{Ca}}} + \alpha_{\text{Ca}} S_i(t)$$

With $\tau_{\text{Ca}} \approx 1000\text{ ms}$ and $\tau_{\text{CAN}} \approx 2000\text{ ms}$, the long decay constant of the gating variable allows groups of active neurons to maintain stable persistent firing rates corresponding to $n$ even after external inputs cease.

### 2.2 Short-Term Synaptic Plasticity (STSP)
To prevent energy depletion from continuous spiking, count states can be stored in the synaptic efficiency utilizing the Tsodyks-Markram model:

$$\frac{dx_j(t)}{dt} = \frac{1 - x_j(t)}{\tau_D} - u_j(t) x_j(t) S_j(t)$$

$$\frac{du_j(t)}{dt} = \frac{U - u_j(t)}{\tau_F} + U(1 - u_j(t)) S_j(t)$$

- $x_j(t) \in [0, 1]$ represents the fraction of available neurotransmitter resources (synaptic depression).
- $u_j(t) \in [0, 1]$ represents the utilization of resources (synaptic facilitation).
- $U$ is the baseline release probability, $\tau_D \approx 200\text{ ms}$, and $\tau_F \approx 5000\text{ ms}$.

The effective weight of the synapse between presynaptic neuron $j$ and postsynaptic neuron $i$ is:

$$W_{ij}^{\text{eff}}(t) = W_{ij} \cdot x_j(t) \cdot u_j(t)$$

**Memory Retention Logic**: During the $a^n$ phase, incoming spikes facilitate recurrent synapses in $\mathcal{R}_A$ ($u_j \to 1$). When the gating hub closes the projection gate for $\mathcal{R}_A$ ($z_A \to 1$) upon transitioning to $b$, the downstream propagation is blocked. The facilitated synaptic state is preserved in the variable $u_j(t)$ over the duration of $\tau_F$ (up to $5\text{ seconds}$). During matching phases, a low-frequency probe pulse reads out the count representation without causing cumulative drift.

#### 2.3 Neural Line Attractor Dynamics & Synfire Odometer Chains
A line attractor is formed by arranging neurons in sequential, recurrently inhibited clusters (an odometer chain):

$$\mathcal{P}_0 \to \mathcal{P}_1 \to \dots \to \mathcal{P}_N$$

Each cluster $\mathcal{P}_k$ is a closely coupled excitatory assembly that projects to the next cluster $\mathcal{P}_{k+1}$ via facilitated feedforward connections, and inhibits all other clusters via feedback interneurons. 

The dynamics of the line attractor are defined by:

$$\tau_a \frac{dr_k(t)}{dt} = -r_k(t) + \phi\left( W_{\text{rec}} r_k(t) + W_{\text{ff}} r_{k-1}(t) - \beta \sum_{j \neq k} r_j(t) + I_{\text{ext}}(t) + I_{\text{CAN}, k}(t) \right)$$

where $r_k(t)$ is the mean firing rate of assembly $\mathcal{P}_k$, $W_{\text{rec}}$ is the recurrent excitatory weight, $W_{\text{ff}}$ is the feedforward coupling weight from the preceding assembly $\mathcal{P}_{k-1}$ (with boundary condition $r_{-1}(t) = 0$), $\beta$ is the global inhibition factor, and $\phi(\cdot)$ is a non-linear threshold-linear transfer function. 
- The $I_{\text{CAN}}$ current provides a localized self-sustaining depolarizing drive.
- The combination of localized excitation, global lateral inhibition, and persistent cationic currents creates stable, discrete fixed points along a one-dimensional attractor manifold, preventing representation drift during the transition phases.

---

## 3. Sequence Routing and State Transition Flow

The sequential processing of $a^n b^n c^n$ is executed through three distinct phases controlled by the gating hub:

```
=== PHASE 1 (a^n) ===
Input 'a' -----> [Reservoir A (Count a)] (Gate A: Open)
                 [Reservoir B]           (Gate B: Closed)
                 [Reservoir C]           (Gate C: Closed)

=== PHASE 2 (b^n) ===
                 [Reservoir A]           (Gate A: Closed / State Frozen)
Input 'b' -----> [Reservoir B (Count b)] (Gate B: Open)
                 [Reservoir C]           (Gate C: Closed)
                 * Topographic projection A -> B verifies n_b <= n_a

=== PHASE 3 (c^n) ===
                 [Reservoir A]           (Gate A: Closed / State Frozen)
                 [Reservoir B]           (Gate B: Closed / State Frozen)
Input 'c' -----> [Reservoir C (Match c)] (Gate C: Open)
                 * Topographic projection B -> C decrements match state
```

### 3.1 Step-by-Step State Evolution

#### Phase 1: Counting $a$ ($a^n$)
- **Input Routing**: The input stream $I_a$ active. Gating states: $z_A = 0$, $z_B = 1$, $z_C = 1$.
- **Evolution**: Each spike representing token $a$ drives the odometer chain in Reservoir $\mathcal{R}_A$, shifting active state $\mathcal{P}_k \to \mathcal{P}_{k+1}$.
- **Storage**: When the $a^n$ sequence terminates, the active population index $k = n$ is locked in place via $I_{\text{CAN}}$ persistent dynamics and facilitated STSP weights.

#### Phase 2: Shifting to $b$ and Comparing ($b^n$)
- **Transition Trigger**: Arrival of the first $b$ token triggers a predictive mismatch error in $\mathcal{P}_E$. The control reservoir $\mathcal{R}_G$ fires, setting gating states to: $z_A = 1$, $z_B = 0$, $z_C = 1$. To prevent gating transition race conditions and off-by-one counting errors, the input $I_b$ is buffered by sensory delay lines, ensuring the gate $z_B(t)$ is open before input spikes arrive at $\mathcal{R}_B$.
- **Evolution**: Spikes from input $I_b$ drive the odometer chain in Reservoir $\mathcal{R}_B$ ($\mathcal{Q}_m \to \mathcal{Q}_{m+1}$).
- **Comparison**: Excitatory topographic pathways project from the frozen state of $\mathcal{R}_A$ to corresponding assemblies in $\mathcal{R}_B$ ($\mathcal{P}_k \to \mathcal{Q}_k$). If the count of $b$ tokens ($m$) exceeds the frozen count of $a$ tokens ($n$), the imbalance triggers an over-counting error population, terminating processing.

#### Phase 3: Shifting to $c$ and Matching ($c^n$)
- **Transition Trigger**: The arrival of the first $c$ token updates gating states to: $z_A = 1$, $z_B = 1$, $z_C = 0$. Gating transition race conditions are mitigated by sensory delay lines routing $I_c$ to $\mathcal{R}_C$.
- **Subtractive Matching**: Reservoir $\mathcal{R}_C$ acts as a comparator. It receives direct excitatory input from $I_c$ and inhibitory topography from the frozen state of $\mathcal{R}_B$ ($\mathcal{Q}_m \to \mathcal{R}_C$). 
- **Sequence Delimiter Prediction**: Each incoming $c$ token decrements the active subtraction state. When the subtraction matches exactly and reaches zero (represented by the activation of the baseline cluster $\mathcal{Q}_0$ in the comparator loop), the readout layer predicts the End-Of-File (EOF) token. If the sequence terminates before reaching zero, or if $c$ tokens continue to arrive after reaching zero, the mismatch triggers prediction error spikes, invalidating the sequence.

### 3.2 Propagation Delay Race Condition Resolution
To prevent the propagation delay race condition during phase transitions (e.g., $a^n \to b^n$), where the first token of a new block arrives at the counting reservoir before the gating hub has fully transitioned the gates, the network incorporates two complementary temporal mechanisms:

1. **Sensory Pathway Delay Lines (Buffers)**: The sensory inputs for tokens $a, b, c$ project to their respective reservoirs through axonal delay lines or transient feedforward neural buffers. These introduce a small propagation delay:
   $$\tau_{\text{delay}} \approx 20 - 30\text{ ms}$$
   This delay ensures that when a transition token arrives, the predictive error population $\mathcal{P}_E$ and the gating hub $\mathcal{R}_G$ have sufficient time to update the gating variables (opening the target reservoir and closing the previous one) before the sensory spike volley reaches the counting reservoirs.
2. **Fast Feedforward Shunting Pathways**: Direct bypass pathways from the sensory inputs project to the gating hub $\mathcal{R}_G$. These pathways bypass the slower recurrent loops of the counting reservoirs to initiate the gate state transition preemptively, ensuring that the gate is fully open when the main sensory inputs arrive.

---

## 4. Localized Training Protocol (Without BPTT)

To avoid Backpropagation Through Time (BPTT), we implement a **Three-Factor Dopamine-Modulated Spike-Timing-Dependent Plasticity (D-STDP)** rule. This rule restricts synaptic modifications to local temporal correlations, modulated by a global reinforcement signal representing sequence correctness.

### 4.1 Synaptic Eligibility Trace Dynamics
At each synapse between presynaptic neuron $j$ and postsynaptic neuron $i$, localized fast eligibility traces $E_{ij}^+(t)$ (for LTP) and $E_{ij}^-(t)$ (for LTD) are maintained to store the temporal correlation between pre- and post-synaptic firing. To prevent sign inversion under error conditions and allow independent modulation, we decouple the LTP and LTD eligibility traces as positive-valued quantities:

$$\frac{dE_{ij}^+(t)}{dt} = -\frac{E_{ij}^+(t)}{\tau_E} + S_{\text{post}, i}(t) P_j(t)$$

$$\frac{dE_{ij}^-(t)}{dt} = -\frac{E_{ij}^-(t)}{\tau_E} + S_{\text{pre}, j}(t) Q_i(t)$$

where $\tau_E \approx 150\text{ ms}$ is the fast eligibility decay time constant. The pre- and post-synaptic activity traces are defined by:

$$\frac{dP_j(t)}{dt} = -\frac{P_j(t)}{\tau_+} + A_+ S_{\text{pre}, j}(t)$$

$$\frac{dQ_i(t)}{dt} = -\frac{Q_i(t)}{\tau_-} + A_- S_{\text{post}, i}(t)$$

Here, $\tau_+, \tau_- \approx 20\text{ ms}$ govern the timing window, while $A_+$ and $A_-$ dictate the amplitude of potentiation and depression, respectively. Both activity traces and eligibility traces are defined as non-negative.

#### Slow Retrograde Chemical Traces (Temporal Credit Assignment)
Because sequence validation occurs only at the End-Of-File (EOF), which can be several seconds after early sequence transitions, the fast eligibility traces $E_{ij}^{\pm}(t)$ would decay to zero before reward delivery. To solve this temporal credit assignment problem, we introduce slow retrograde chemical traces $G_{ij}^+(t)$ and $G_{ij}^-(t)$ with $\tau_G \approx 10-20\text{ s}$ (representing CaMKII phosphorylation or nitric oxide retrograde signaling) that act as biochemical buffers:

$$\tau_G \frac{dG_{ij}^+(t)}{dt} = -G_{ij}^+(t) + E_{ij}^+(t)$$

$$\tau_G \frac{dG_{ij}^-(t)}{dt} = -G_{ij}^-(t) + E_{ij}^-(t)$$

These slow traces integrate the fast eligibility signals and decay slowly, preserving the memory of synaptic correlations over the duration of the entire multi-second sequence. Alternatively, intermediate checkpoint dopamine rewards can be delivered at phase transition boundaries (e.g., $a \to b$ and $b \to c$) using local prediction error silence to reinforce sub-phase transitions before fast traces decay.

### 4.2 Third-Factor Neuromodulation Update
The slow eligibility traces $G_{ij}^{\pm}(t)$ are converted into physical weight changes via modulation by the global dopamine fluctuation signal $DA(t)$. To correct the sign inversion flaw under error conditions ($DA(t) < 0$), we use a rectified dopaminergic update equation:

$$\frac{dW_{ij}(t)}{dt} = \eta \cdot \left( DA(t) \cdot G_{ij}^+(t) - \max(0, DA(t)) \cdot G_{ij}^-(t) \right)$$

where $\eta$ is the learning rate.

#### Dopamine Signal ($DA(t)$) Control Logic and Sign Inversion Resolution
1. **Reward State ($DA(t) > 0$)**: On successful sequence validation (matching $a^n b^n c^n$ at EOF), a positive dopamine burst is released. Since $DA(t) > 0$, the update simplifies to:
   $$\frac{dW_{ij}(t)}{dt} = \eta \cdot DA(t) \cdot (G_{ij}^+(t) - G_{ij}^-(t))$$
   This reinforces LTP while depressing LTD, reinforcing the correct pathways.
2. **Error State ($DA(t) < 0$)**: If an over-counting or under-counting error is detected, a negative dopamine dip is triggered. Since $\max(0, DA(t)) = 0$, the update becomes:
   $$\frac{dW_{ij}(t)}{dt} = \eta \cdot DA(t) \cdot G_{ij}^+(t)$$
   Because $DA(t) < 0$, this results in direct depression of recently active LTP pathways. Crucially, it prevents the sign inversion flaw where LTD pathways would otherwise be erroneously potentiated under a negative dopamine signal.
3. **Gated Updates**: Because the reservoirs are gated, only synapses within the active pathways maintain non-zero eligibility traces at any given moment. This localized updates mechanism prevents catastrophic forgetting in silent reservoirs.

### 4.3 Readout Layer Dynamics and Classification Mapping
The readout layer $\mathcal{R}_{\text{out}}$ translates the spiking activity of the reservoirs into token predictions and sequence validation. It consists of Leaky Integrate-and-Fire (LIF) classification neurons, with each neuron $r \in \{a, b, c, \text{EOF}, \text{Error}\}$ representing a target category.

#### Readout Neuron Dynamics
The membrane potential $V_r(t)$ of readout neuron $r$ is governed by:

$$C_m \frac{dV_r(t)}{dt} = -g_L (V_r(t) - E_L) - I_{\text{syn}, r}(t)$$

where the synaptic input current $I_{\text{syn}, r}(t)$ is driven by the spiking activity of all presynaptic reservoir neurons:

$$I_{\text{syn}, r}(t) = g_{\text{AMPA}, r}(t)(V_r(t) - E_{\text{AMPA}})$$

The AMPA conductance evolves according to:

$$\tau_{\text{out}} \frac{dg_{\text{AMPA}, r}(t)}{dt} = -g_{\text{AMPA}, r}(t) + \sum_{k \in \{A, B, C\}} \sum_{i \in \mathcal{R}_k} W_{ri}^{\text{out}} S_i^{(k)}(t)$$

where $W_{ri}^{\text{out}}$ is the synaptic weight from reservoir neuron $i$ in reservoir $k$ to readout neuron $r$, and $S_i^{(k)}(t) = \sum_f \delta(t - t_{i, f}^{(k)})$ is the presynaptic spike train.

#### Classification Mapping
The network's prediction at any time $\hat{x}_{t|t-1}$ is mapped via a temporal winner-take-all (WTA) mechanism over the readout population's firing rates or first-spike latencies. Let $S_r(t)$ be the output spike train of readout neuron $r$. The classification decision is determined by the maximum spike count within a sliding temporal integration window $\Delta t \approx 50\text{ ms}$:

$$\hat{x}_{t|t-1} = \operatorname{arg\,max}_{r \in \{a, b, c, \text{EOF}, \text{Error}\}} \int_{t-\Delta t}^{t} S_r(t') dt'$$

If the network reaches EOF and the "EOF" neuron fires, the sequence is successfully validated. If the "Error" neuron fires, or if a mismatch is detected, the sequence is rejected.

---

## 5. Comparative Structural Analysis

| Feature Metric | Standard Homogeneous LSM | Proposed TEPRE + Multi-Scale RLSM |
|---|---|---|
| **Memory Scaling** | Fades exponentially ($e^{-t/\tau_m}$) | Persistent ($I_{\text{CAN}}$ attractors & silent STSP) |
| **Information Routing** | Random recurrent scattering | Gated routing via control hub ($\mathcal{R}_G$) |
| **Language Capability** | Regular Languages ($a^* b^* c^*$) | Context-Sensitive ($a^n b^n c^n$) |
| **Training Algorithm** | Readout training only (unstable) | Localized 3-factor D-STDP (no gradients) |
| **Energy Footprint** | High (continuous spiking to maintain states) | Low (leverages silent synaptic memory states) |

---

## 6. Verification and Simulation Validation Protocol

To validate this architecture in a spiking simulator (such as NEST or Brian2):

1. **Dynamic Length Generalization Sweep**:
   - Stimulate the network with sequences $a^n b^n c^n$ for $n \in [1, 50]$.
   - Introduce variable delay intervals between token blocks (e.g., $100\text{ ms}$ to $5000\text{ ms}$).
   - Verify that classification accuracy of sequence validation from the readout layer remains at $100\%$.
2. **Attractor State Stability Test**:
   - Probe the membrane potential and firing rates of the odometer assemblies during the gated silent phases.
   - Confirm that the representation does not drift along the line attractor manifold during delay periods.
3. **Synaptic Strength Convergence Check**:
   - Track the evolution of topographic weights $W(\mathcal{P}_k \to \mathcal{Q}_k)$ under D-STDP training.
   - Verify that weights converge monotonically to a stable diagonal matrix without gradient explosion.
