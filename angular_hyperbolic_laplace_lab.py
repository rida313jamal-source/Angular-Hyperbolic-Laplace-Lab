import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle

try:
    from scipy.special import gamma as scipy_gamma
    from scipy.special import jv as scipy_jv
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    scipy_gamma = None
    scipy_jv = None


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Angular–Hyperbolic Laplace Laboratory",
    page_icon="∠",
    layout="wide",
)


# ============================================================
# Core mathematics
# ============================================================
@dataclass
class PolarData:
    t: float
    b: float
    r: float
    theta: float


@dataclass
class HyperbolicData:
    t: float
    b: float
    r_bar: float
    phi: float


class DomainError(ValueError):
    pass


def validate_t(t: float) -> None:
    if t <= 0:
        raise DomainError("The Laplace damping parameter t must satisfy t > 0.")


def validate_hyperbolic_domain(t: float, b: float) -> None:
    validate_t(t)
    if abs(b) >= t:
        raise DomainError("Hyperbolic formulas require |b| < t.")


def gamma_fn(x: float) -> float:
    if SCIPY_AVAILABLE:
        return float(scipy_gamma(x))
    return float(math.gamma(x))


def polar_data(t: float, b: float) -> PolarData:
    validate_t(t)
    r = math.sqrt(t * t + b * b)
    theta = math.atan2(b, t)
    return PolarData(t=t, b=b, r=r, theta=theta)


def hyperbolic_data(t: float, b: float) -> HyperbolicData:
    validate_hyperbolic_domain(t, b)
    r_bar = math.sqrt(t * t - b * b)
    phi = math.atanh(b / t)
    return HyperbolicData(t=t, b=b, r_bar=r_bar, phi=phi)


def dtheta_operator(t: float, b: float, f_t: float, f_b: float) -> float:
    """D_theta = -b d/dt + t d/db"""
    return -b * f_t + t * f_b


def dr_operator(t: float, b: float, f_t: float, f_b: float) -> float:
    """D_r = (t/r) d/dt + (b/r) d/db"""
    pd = polar_data(t, b)
    return (t / pd.r) * f_t + (b / pd.r) * f_b


def safe_jv(v: float, x: float) -> float:
    if SCIPY_AVAILABLE:
        return float(scipy_jv(v, x))
    # Fallback: a very rough truncated series around x=0 for demonstration only.
    # This keeps the app functional if SciPy is absent.
    total = 0.0
    for m in range(15):
        num = ((-1) ** m) * (x / 2.0) ** (2 * m + v)
        den = math.factorial(m) * gamma_fn(m + v + 1.0)
        total += num / den
    return float(total)


# ============================================================
# Unified formulas from the paper
# ============================================================
def L_exp_ibx(t: float, b: float) -> complex:
    pd = polar_data(t, b)
    return complex(math.cos(pd.theta), math.sin(pd.theta)) / pd.r


def L_cos(t: float, b: float) -> float:
    pd = polar_data(t, b)
    return math.cos(pd.theta) / pd.r


def L_sin(t: float, b: float) -> float:
    pd = polar_data(t, b)
    return math.sin(pd.theta) / pd.r


def L_xn_cos(t: float, b: float, n: int) -> float:
    pd = polar_data(t, b)
    return math.factorial(n) * math.cos((n + 1) * pd.theta) / (pd.r ** (n + 1))


def L_xn_sin(t: float, b: float, n: int) -> float:
    pd = polar_data(t, b)
    return math.factorial(n) * math.sin((n + 1) * pd.theta) / (pd.r ** (n + 1))


def L_xs_cos(t: float, b: float, s: float) -> float:
    if s <= -1:
        raise DomainError("Fractional formula requires s > -1.")
    pd = polar_data(t, b)
    return gamma_fn(s + 1.0) * math.cos((s + 1.0) * pd.theta) / (pd.r ** (s + 1.0))


def L_xs_sin(t: float, b: float, s: float) -> float:
    if s <= -1:
        raise DomainError("Fractional formula requires s > -1.")
    pd = polar_data(t, b)
    return gamma_fn(s + 1.0) * math.sin((s + 1.0) * pd.theta) / (pd.r ** (s + 1.0))


def L_phase_shifted_sin(t: float, b: float, omega: float) -> float:
    pd = polar_data(t, b)
    return math.sin(pd.theta + omega) / pd.r


def L_phase_shifted_cos(t: float, b: float, omega: float) -> float:
    pd = polar_data(t, b)
    return math.cos(pd.theta + omega) / pd.r


def L_exp_minus_bx(t: float, b: float) -> float:
    hd = hyperbolic_data(t, b)
    return math.exp(-hd.phi) / hd.r_bar


def L_exp_plus_bx(t: float, b: float) -> float:
    hd = hyperbolic_data(t, b)
    return math.exp(hd.phi) / hd.r_bar


def L_sinh(t: float, b: float) -> float:
    hd = hyperbolic_data(t, b)
    return math.sinh(hd.phi) / hd.r_bar


def L_cosh(t: float, b: float) -> float:
    hd = hyperbolic_data(t, b)
    return math.cosh(hd.phi) / hd.r_bar


def L_xn_sinh(t: float, b: float, n: int) -> float:
    hd = hyperbolic_data(t, b)
    return math.factorial(n) * math.sinh((n + 1) * hd.phi) / (hd.r_bar ** (n + 1))


def L_xn_cosh(t: float, b: float, n: int) -> float:
    hd = hyperbolic_data(t, b)
    return math.factorial(n) * math.cosh((n + 1) * hd.phi) / (hd.r_bar ** (n + 1))


def L_xn_exp_minus_bx(t: float, b: float, n: int) -> float:
    hd = hyperbolic_data(t, b)
    return math.factorial(n) * math.exp(-(n + 1) * hd.phi) / (hd.r_bar ** (n + 1))


def L_xn_exp_plus_bx(t: float, b: float, n: int) -> float:
    hd = hyperbolic_data(t, b)
    return math.factorial(n) * math.exp((n + 1) * hd.phi) / (hd.r_bar ** (n + 1))


def L_sinc(t: float, b: float) -> float:
    validate_t(t)
    if abs(b) < 1e-14:
        return 1.0 / t
    pd = polar_data(t, b)
    return pd.theta / b


def L_J0(t: float, b: float) -> float:
    pd = polar_data(t, b)
    return 1.0 / pd.r


def L_Jv(t: float, b: float, nu: float) -> float:
    pd = polar_data(t, b)
    if abs(b) < 1e-14:
        if abs(nu) < 1e-14:
            return 1.0 / pd.r
        return 0.0
    return (math.tan(pd.theta / 2.0) ** nu) / pd.r


def angular_shift_formula(t: float, b: float, a: float, trig: str = "sin") -> Tuple[float, float, float]:
    validate_t(t)
    r_a = math.sqrt((t - a) ** 2 + b * b)
    theta_a = math.atan2(b, t - a)
    if trig == "sin":
        value = math.sin(theta_a) / r_a
    else:
        value = math.cos(theta_a) / r_a
    return theta_a, r_a, value


# ============================================================
# Symbolic display helpers
# ============================================================
def fmt(x: float, digits: int = 8) -> str:
    return f"{x:.{digits}g}"


def formula_block(title: str, body: str) -> None:
    st.markdown(f"### {title}")
    st.latex(body)


# ============================================================
# Numerical integration helpers for verification
# ============================================================
def trapz_integral(f: Callable[[np.ndarray], np.ndarray], upper: float = 30.0, n: int = 12000) -> float:
    xs = np.linspace(0.0, upper, n)
    ys = f(xs)
    return float(np.trapz(ys, xs))


def verify_sin_integral(t: float, b: float, n: int) -> Tuple[float, float]:
    def f(x: np.ndarray) -> np.ndarray:
        return (x ** n) * np.exp(-t * x) * np.sin(b * x)
    numeric = trapz_integral(f)
    closed = L_xn_sin(t, b, n)
    return numeric, closed


def verify_cos_integral(t: float, b: float, n: int) -> Tuple[float, float]:
    def f(x: np.ndarray) -> np.ndarray:
        return (x ** n) * np.exp(-t * x) * np.cos(b * x)
    numeric = trapz_integral(f)
    closed = L_xn_cos(t, b, n)
    return numeric, closed


def verify_sinh_integral(t: float, b: float, n: int) -> Tuple[float, float]:
    validate_hyperbolic_domain(t, b)
    def f(x: np.ndarray) -> np.ndarray:
        return (x ** n) * np.exp(-t * x) * np.sinh(b * x)
    numeric = trapz_integral(f)
    closed = L_xn_sinh(t, b, n)
    return numeric, closed


def verify_cosh_integral(t: float, b: float, n: int) -> Tuple[float, float]:
    validate_hyperbolic_domain(t, b)
    def f(x: np.ndarray) -> np.ndarray:
        return (x ** n) * np.exp(-t * x) * np.cosh(b * x)
    numeric = trapz_integral(f)
    closed = L_xn_cosh(t, b, n)
    return numeric, closed


def verify_sinc_integral(t: float, b: float) -> Tuple[float, float]:
    validate_t(t)
    def sinc(z: np.ndarray) -> np.ndarray:
        out = np.ones_like(z)
        mask = np.abs(z) > 1e-12
        out[mask] = np.sin(z[mask]) / z[mask]
        return out
    def f(x: np.ndarray) -> np.ndarray:
        return np.exp(-t * x) * sinc(b * x)
    numeric = trapz_integral(f, upper=50.0, n=20000)
    closed = L_sinc(t, b)
    return numeric, closed


def verify_bessel_integral(t: float, b: float, nu: float = 0.0) -> Tuple[float, float]:
    validate_t(t)
    def f(x: np.ndarray) -> np.ndarray:
        vals = np.array([safe_jv(nu, b * xi) for xi in x])
        return np.exp(-t * x) * vals
    numeric = trapz_integral(f, upper=40.0, n=16000)
    closed = L_Jv(t, b, nu)
    return numeric, closed


# ============================================================
# Plotting utilities
# ============================================================
def setup_axes(ax, title: str):
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel("b")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def plot_polar_geometry(t: float, b: float):
    pd = polar_data(t, b)
    fig, ax = plt.subplots(figsize=(6, 6))
    setup_axes(ax, "Circular (Euclidean) Geometry")

    radius = max(1.25 * pd.r, 1.0)
    ax.set_xlim(-0.2 * radius, 1.2 * radius)
    ax.set_ylim(-1.1 * radius, 1.1 * radius)

    ax.plot([0, pd.t], [0, pd.b], linewidth=2)
    ax.plot([pd.t, pd.t], [0, pd.b], linestyle="--", linewidth=1)
    ax.plot([0, pd.t], [0, 0], linestyle="--", linewidth=1)
    ax.scatter([pd.t], [pd.b], s=60)

    circle = Circle((0, 0), pd.r, fill=False, linewidth=1.5)
    ax.add_patch(circle)
    arc = Arc((0, 0), 0.7 * pd.r, 0.7 * pd.r, theta1=0, theta2=np.degrees(pd.theta), linewidth=2)
    ax.add_patch(arc)

    ax.text(pd.t, pd.b, f"  z=(t,b)=({fmt(t,4)},{fmt(b,4)})")
    ax.text(pd.t / 2, pd.b / 2, f"r={fmt(pd.r,4)}")
    ax.text(0.3 * pd.r * math.cos(pd.theta / 2), 0.3 * pd.r * math.sin(pd.theta / 2), f"θ={fmt(pd.theta,4)}")
    return fig


def plot_hyperbolic_geometry(t: float, b: float):
    hd = hyperbolic_data(t, b)
    fig, ax = plt.subplots(figsize=(6, 6))
    setup_axes(ax, "Hyperbolic (Minkowskian-Type) Geometry")

    extent = max(1.2 * t, 1.0)
    ax.set_xlim(0, extent)
    ax.set_ylim(-extent, extent)

    ax.plot([0, t], [0, b], linewidth=2)
    ax.plot([t, t], [0, b], linestyle="--", linewidth=1)
    ax.plot([0, t], [0, 0], linestyle="--", linewidth=1)
    ax.scatter([t], [b], s=60)

    ys = np.linspace(-0.98 * extent, 0.98 * extent, 500)
    valid = ys * ys < hd.r_bar ** 2 + extent ** 2
    ys = ys[valid]
    xs = np.sqrt(hd.r_bar ** 2 + ys ** 2)
    ax.plot(xs, ys, linewidth=1.5)

    ax.text(t, b, f"  z=(t,b)=({fmt(t,4)},{fmt(b,4)})")
    ax.text(t / 2, b / 2, f"r̄={fmt(hd.r_bar,4)}")
    ax.text(max(0.1, 0.35 * t), 0.15 * np.sign(b if b != 0 else 1) * extent, f"φ={fmt(hd.phi,4)}")
    return fig


def plot_horizontal_translation(t: float, b: float, delta_t: float):
    validate_t(t)
    fig, ax = plt.subplots(figsize=(7, 4))
    setup_axes(ax, "Horizontal Differentiation in the (t,b)-Plane")
    t2 = t + delta_t
    ax.set_xlim(0, max(t2, t) * 1.4 + 0.5)
    ax.set_ylim(-abs(b) * 2 - 1, abs(b) * 2 + 1)

    ax.scatter([t, t2], [b, b], s=70)
    ax.plot([t, t2], [b, b], linewidth=2)
    ax.annotate("∂ₜ", xy=((t + t2) / 2, b + 0.1), ha="center")
    ax.text(t, b, "  z=t+ib")
    ax.text(t2, b, "  z+Δt")
    ax.text((t + t2) / 2, b - 0.4, "horizontal translation\n(no geometric rotation)", ha="center")
    return fig


def plot_kernel_classification_sketch():
    fig, ax = plt.subplots(figsize=(8, 5))
    setup_axes(ax, "Kernel Geometry Sketch")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)

    circ = Circle((0, 0), 1.0, fill=False, linewidth=1.5)
    ax.add_patch(circ)

    ang = np.pi / 6
    ax.plot([0, np.cos(ang)], [0, np.sin(ang)], linewidth=2)
    ax.plot([0, np.cos(ang + 0.25)], [0, np.sin(ang + 0.25)], linewidth=2)
    ax.text(0.8, 0.55, "sin, cos")
    ax.text(0.55, 0.82, "sinc sector")
    ax.text(-0.95, 0.95, "J0, Jν: full circle")
    return fig


def plot_transform_values_grid(t_values: np.ndarray, b: float, family: str, n: int = 0):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("t")
    ax.set_ylabel("Transform value")
    ax.set_title(f"Transform family: {family}")
    ax.grid(True, alpha=0.25)

    ys = []
    for t in t_values:
        try:
            if family == "sin":
                ys.append(L_xn_sin(float(t), b, n))
            elif family == "cos":
                ys.append(L_xn_cos(float(t), b, n))
            elif family == "sinh":
                ys.append(L_xn_sinh(float(t), b, n))
            elif family == "cosh":
                ys.append(L_xn_cosh(float(t), b, n))
            elif family == "sinc":
                ys.append(L_sinc(float(t), b))
            else:
                ys.append(L_J0(float(t), b))
        except Exception:
            ys.append(np.nan)
    ax.plot(t_values, ys, linewidth=2)
    return fig


# ============================================================
# Text sections based on the paper
# ============================================================
def section_introduction():
    st.header("1. Introduction")
    st.markdown(
        """
This app implements the research framework **A Unified Angular–Hyperbolic Representation of the Laplace Transform**.

The core idea is to regard the Laplace parameter pair $(t,b)$ as a geometric point and then classify kernels into two complementary regimes:

This paper introduces a unified angular–hyperbolic representation of theLaplace transform, aimed at providing a geometric interpretation of Laplace-induced kernels .  The proposed framework distinguishes between boundedoscillatory kernels and unbounded kernels like exponential or hyperbolic throughtheir geometric realization in the parameter space $(t,b)$ This approach illus-trates how the angular geometric representation simplifies the understandingof kernel behaviors and provides new insights into Laplace transform applica-tions.Remark:  In  this  paper,  the  term Laplace-induced kernels  refers  to  theinput functions within the Laplace integral.  We use this term to denote thosefunctions whose geometric behavior is analyzed within the framework of thetransform

- **Bounded oscillatory kernels** → circular / angular geometry.
- **Unbounded exponential kernels** → hyperbolic geometry.

The classical Laplace transform remains the same. The novelty lies in the **representation lens**.
        """
    )
    st.markdown("In the beginning the complex quantity $(z=t+ib)$ is  introduced  solely  as a  geometric parameterization  of the  Laplace  variablet and Scale Parameter b The Laplace transform remains defined by the real integral")

    formula_block(
        "Classical Laplace form used in the paper",
        r"\mathcal{L}\{f(bx)\}(t)=\int_0^{\infty} e^{-tx} f(bx)\,dx, \qquad t>0."
    )

    st.markdown(
        """
where the parameter t governs exponential decay and the parameter b modu-lates oscillatory or growth behavior inside the kernel his yields an analytically equivalent form to the classical.The  frequency  parameter b  is  introduced  by  placing  it  explicitly  inside  theoriginal function, rather than embedding it in the exponential variable.  Ac-cordingly, the function is written in the form.
        """
    )


def section_foundations():
    st.header("2. Geometric Foundations")

    st.subheader("2.1 Circular (bounded) regime")
    formula_block(
        "Polar coordinates",
        r"r=\sqrt{t^2+b^2}, \qquad \theta=\tan^{-1}\!\left(\frac{b}{t}\right), \qquad z=t+ib=r e^{i\theta}."
    )
    st.markdown(
        "In this regime, oscillatory kernels such as : $sin(bx)$, $cos(bx)$, and complex exponentials are represented by a Euclidean triangle in the parameter plane."
    )

    st.subheader("2.2 Hyperbolic (unbounded) regime")
    formula_block(
        "Hyperbolic coordinates",
        r"\bar r=\sqrt{t^2-b^2}, \qquad \phi=\tanh^{-1}\!\left(\frac{b}{t}\right), \qquad |b|<t."
    )
    st.markdown(
        "This regime governs exponential and hyperbolic kernels such as : $e^{bx}$, $e^{-bx}$, $sinh(bx)$, and $cosh(bx)$."
    )

    col1, col2 = st.columns(2)
    with col1:
        t = st.number_input("t (circular plot)", min_value=0.01, value=2.0, step=0.1, key="found_t1")
        b = st.number_input("b (circular plot)", value=1.0, step=0.1, key="found_b1")
        try:
            st.pyplot(plot_polar_geometry(t, b), use_container_width=True)
        except Exception as e:
            st.error(str(e))
    with col2:
        t2 = st.number_input("t (hyperbolic plot)", min_value=0.01, value=2.0, step=0.1, key="found_t2")
        b2 = st.number_input("b (hyperbolic plot)", value=0.75, step=0.1, key="found_b2")
        try:
            st.pyplot(plot_hyperbolic_geometry(t2, b2), use_container_width=True)
        except Exception as e:
            st.error(str(e))


def section_unified_theorem():
    st.header("3. Unified Representation Theorem")
    st.markdown(
        "The paper's main theorem states that Laplace images can be represented in either angular or hyperbolic form depending on kernel boundedness."
    )
    formula_block(
        "Unified theorem",
        r"\mathcal{L}\{f(bx)\}(t)=\begin{cases}\dfrac{F(\theta)}{r}, & \text{bounded kernels},\\[6pt]\dfrac{F(\phi)}{\bar r}, & \text{unbounded kernels}.\end{cases}"
    )

    st.markdown(
        "This is not a change in the Laplace transform itself, but a unified representation of its structural output."
    )

    st.subheader("Theorem checker")
    kernel = st.selectbox(
        "Choose a kernel family",
        [
            "sin(bx)",
            "cos(bx)",
            "e^{ibx}",
            "e^{-bx}",
            "e^{bx}",
            "sinh(bx)",
            "cosh(bx)",
        ],
        key="theorem_kernel",
    )
    t = st.number_input("t", min_value=0.01, value=2.0, step=0.1, key="theorem_t")
    b = st.number_input("b", value=1.0, step=0.1, key="theorem_b")

    try:
        if kernel == "sin(bx)":
            value = L_sin(t, b)
            st.latex(r"\mathcal{L}\{\sin(bx)\}(t)=\frac{\sin\theta}{r}")
        elif kernel == "cos(bx)":
            value = L_cos(t, b)
            st.latex(r"\mathcal{L}\{\cos(bx)\}(t)=\frac{\cos\theta}{r}")
        elif kernel == "e^{ibx}":
            value = L_exp_ibx(t, b)
            st.latex(r"\mathcal{L}\{e^{ibx}\}(t)=\frac{e^{i\theta}}{r}")
        elif kernel == "e^{-bx}":
            value = L_exp_minus_bx(t, b)
            st.latex(r"\mathcal{L}\{e^{-bx}\}(t)=\frac{e^{-\phi}}{\bar r}")
        elif kernel == "e^{bx}":
            value = L_exp_plus_bx(t, b)
            st.latex(r"\mathcal{L}\{e^{bx}\}(t)=\frac{e^{\phi}}{\bar r}")
        elif kernel == "sinh(bx)":
            value = L_sinh(t, b)
            st.latex(r"\mathcal{L}\{\sinh(bx)\}(t)=\frac{\sinh\phi}{\bar r}")
        else:
            value = L_cosh(t, b)
            st.latex(r"\mathcal{L}\{\cosh(bx)\}(t)=\frac{\cosh\phi}{\bar r}")
        st.success(f"Computed value: {value}")
    except Exception as e:
        st.error(str(e))


def section_proofs_and_examples():
    st.header("4. Elementary Proofs and Low-Order Examples")

    st.subheader("4.1 Oscillatory exponential proof skeleton")
    st.latex(r"\mathcal{L}\{e^{ibx}\}(t)=\int_0^{\infty} e^{-(t-ib)x}\,dx=\frac{1}{t-ib}=\frac{e^{i\theta}}{r}.")
    st.markdown(
        "Taking real and imaginary parts produces the cosine and sine formulas."
    )

    st.subheader("4.2 Sine family")
    st.latex(r"\mathcal{L}\{x^n\sin(bx)\}(t)=\frac{n!\sin((n+1)\theta)}{r^{n+1}}.")

    st.subheader("4.3 Cosine family")
    st.latex(r"\mathcal{L}\{x^n\cos(bx)\}(t)=\frac{n!\cos((n+1)\theta)}{r^{n+1}}.")

    c1, c2 = st.columns(2)
    with c1:
        t = st.number_input("t (verification)", min_value=0.05, value=1.5, step=0.1, key="proof_t")
        b = st.number_input("b (verification)", value=1.0, step=0.1, key="proof_b")
        n = st.slider("n", min_value=0, max_value=6, value=2, key="proof_n")
        family = st.radio("Family", ["sine", "cosine"], horizontal=True, key="proof_family")
        if st.button("Verify integral numerically", key="verify_button"):
            try:
                if family == "sine":
                    numeric, closed = verify_sin_integral(t, b, n)
                else:
                    numeric, closed = verify_cos_integral(t, b, n)
                st.write({
                    "numeric_integral": numeric,
                    "closed_form": closed,
                    "absolute_error": abs(numeric - closed),
                })
            except Exception as e:
                st.error(str(e))
    with c2:
        t_values = np.linspace(max(0.1, abs(b) + 0.05 if family == "sine" else 0.1), 6.0, 250)
        if family == "sine":
            st.pyplot(plot_transform_values_grid(t_values, b, "sin", n), use_container_width=True)
        else:
            st.pyplot(plot_transform_values_grid(t_values, b, "cos", n), use_container_width=True)


def section_fractional_lifting():
    st.header("5. Fractional Angular Lifting")
    st.markdown(
        "The paper extends angular lifting from integer powers to arbitrary real exponents s > -1 using the Gamma function."
    )
    formula_block(
        "Fractional sine formula",
        r"\mathcal{L}\{x^s\sin(bx)\}(t)=\frac{\Gamma(s+1)}{r^{s+1}}\sin((s+1)\theta),\qquad s>-1."
    )
    formula_block(
        "Fractional cosine formula",
        r"\mathcal{L}\{x^s\cos(bx)\}(t)=\frac{\Gamma(s+1)}{r^{s+1}}\cos((s+1)\theta),\qquad s>-1."
    )

    col1, col2 = st.columns(2)
    with col1:
        t = st.number_input("t", min_value=0.05, value=1.0, step=0.1, key="frac_t")
        b = st.number_input("b", value=1.0, step=0.1, key="frac_b")
        s = st.slider("s", min_value=-0.9, max_value=4.0, value=0.5, step=0.1, key="frac_s")
        family = st.radio("Fractional family", ["sin", "cos"], horizontal=True, key="frac_family")
        try:
            if family == "sin":
                value = L_xs_sin(t, b, s)
            else:
                value = L_xs_cos(t, b, s)
            st.metric("Fractional transform value", fmt(value, 10))
        except Exception as e:
            st.error(str(e))
    with col2:
        try:
            pd = polar_data(t, b)
            st.write({
                "r": pd.r,
                "theta": pd.theta,
                "Gamma(s+1)": gamma_fn(s + 1.0),
                "effective angle": (s + 1.0) * pd.theta,
            })
        except Exception as e:
            st.error(str(e))


def section_hyperbolic_regime():
    st.header("6. Hyperbolic Regime and Analytic Continuation")
    st.markdown(
        "By substituting b → ib, the angular representation continues to a hyperbolic one."
    )
    formula_block(
        "Key relations",
        r"\bar r=\sqrt{t^2-b^2},\qquad \phi=\tanh^{-1}\!\left(\frac{b}{t}\right),\qquad |b|<t."
    )
    formula_block(
        "Examples",
        r"\mathcal{L}\{e^{-bx}\}(t)=\frac{e^{-\phi}}{\bar r},\quad \mathcal{L}\{\sinh(bx)\}(t)=\frac{\sinh\phi}{\bar r},\quad \mathcal{L}\{\cosh(bx)\}(t)=\frac{\cosh\phi}{\bar r}."
    )

    t = st.number_input("t", min_value=0.05, value=2.0, step=0.1, key="hyp_t")
    b = st.number_input("b", value=0.5, step=0.1, key="hyp_b")
    n = st.slider("n (for x^n kernels)", min_value=0, max_value=6, value=2, key="hyp_n")
    family = st.selectbox("Family", ["e^{-bx}", "e^{bx}", "sinh(bx)", "cosh(bx)", "x^n sinh(bx)", "x^n cosh(bx)"], key="hyp_family")

    try:
        if family == "e^{-bx}":
            value = L_exp_minus_bx(t, b)
        elif family == "e^{bx}":
            value = L_exp_plus_bx(t, b)
        elif family == "sinh(bx)":
            value = L_sinh(t, b)
        elif family == "cosh(bx)":
            value = L_cosh(t, b)
        elif family == "x^n sinh(bx)":
            value = L_xn_sinh(t, b, n)
        else:
            value = L_xn_cosh(t, b, n)
        st.success(f"Computed value: {value}")
        st.pyplot(plot_hyperbolic_geometry(t, b), use_container_width=True)
    except Exception as e:
        st.error(str(e))


def section_angular_shifting():
    st.header("7. Angular Shifting Formula")
    st.latex(r"\theta_a=\tan^{-1}\!\left(\frac{b}{t-a}\right),\qquad r_a=\sqrt{(t-a)^2+b^2}.")
    st.latex(r"\mathcal{L}\{e^{ax}\sin(bx)\}(t)=\frac{\sin\theta_a}{r_a},\qquad \mathcal{L}\{e^{ax}\cos(bx)\}(t)=\frac{\cos\theta_a}{r_a}.")

    c1, c2 = st.columns(2)
    with c1:
        t = st.number_input("t", min_value=0.05, value=2.0, step=0.1, key="shift_t")
        b = st.number_input("b", value=1.0, step=0.1, key="shift_b")
        a = st.number_input("a", value=0.5, step=0.1, key="shift_a")
        trig = st.radio("Kernel", ["sin", "cos"], horizontal=True, key="shift_trig")
        try:
            
            theta_a, r_a, value = angular_shift_formula(t, b, a, trig)
            st.write({
                "theta_a": theta_a,
                "r_a": r_a,
                "value": value,
            })
        except Exception as e:
            st.error(str(e))
    with c2:
        delta_t = max(0.1, a)
        try:
            st.pyplot(plot_horizontal_translation(t - a if t - a > 0 else 0.1, b, delta_t), use_container_width=True)
        except Exception as e:
            st.error(str(e))


def section_horizontal_differentiation():
    st.header("8. Horizontal Differentiation in the (t,b)-Plane")
    st.markdown(
        "The paper interprets differentiation with respect to t as a horizontal translation at fixed b."
    )
    formula_block(
        "Translation law",
        r"z(t)=t+ib \quad \longmapsto \quad z(t+\Delta t)=(t+\Delta t)+ib,\qquad b=\text{const}."
    )
    formula_block(
        "Oscillatory differentiated family",
        r"\mathcal{L}\{x^n e^{ibx}\}(t)=(-1)^n \partial_t^n\!\left(\frac{1}{t-ib}\right)=\frac{n!}{r^{n+1}}e^{i(n+1)\theta}."
    )

    t = st.number_input("t", min_value=0.05, value=1.5, step=0.1, key="hd_t")
    b = st.number_input("b", value=0.8, step=0.1, key="hd_b")
    delta_t = st.slider("Δt", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="hd_dt")
    st.pyplot(plot_horizontal_translation(t, b, delta_t), use_container_width=True)


def section_composite_kernels():
    st.header("9. Composite Bounded Kernels: sinc and Bessel")
    st.markdown(
        "The paper treats sinc and Bessel kernels as saturated circular structures rather than freely liftable angular kernels."
    )

    st.subheader("9.1 sinc kernel as an angular sector average")
    st.latex(r"\operatorname{sinc}(bx)=\frac{\sin(bx)}{bx}=\int_0^1 \cos((Tb)x)\,dT.")
    st.latex(r"\mathcal{L}\{\operatorname{sinc}(bx)\}(t)=\frac{\theta}{b}."
    )

    st.subheader("9.2 Bessel kernel as full circular averaging")
    st.latex(r"J_0(bx)=\frac{1}{\pi}\int_0^{\pi}\cos(bx\sin\varphi)\,d\varphi.")
    st.latex(r"\mathcal{L}\{J_0(bx)\}(t)=\frac{1}{\sqrt{t^2+b^2}}=\frac{1}{r}." )
    st.latex(r"\mathcal{L}\{J_\nu(bx)\}(t)=\frac{1}{r}\left(\tan\frac{\theta}{2}\right)^\nu,\qquad t>0,\ \nu>-1." )

    c1, c2, c3 = st.columns(3)
    with c1:
        t = st.number_input("t", min_value=0.05, value=1.6, step=0.1, key="comp_t")
    with c2:
        b = st.number_input("b", value=1.0, step=0.1, key="comp_b")
    with c3:
        nu = st.slider("ν", min_value=-0.9, max_value=4.0, value=0.0, step=0.1, key="comp_nu")

    try:
        sinc_value = L_sinc(t, b)
        j0_value = L_J0(t, b)
        jv_value = L_Jv(t, b, nu)
        st.write({
            "L{sinc(bx)}": sinc_value,
            "L{J0(bx)}": j0_value,
            f"L{{J_{nu}(bx)}}": jv_value,
        })
        st.pyplot(plot_kernel_classification_sketch(), use_container_width=True)
    except Exception as e:
        st.error(str(e))


def section_negative_powers_and_inverse():
    st.header("10. Negative Powers and the Inverse Appearance of sinc")
    st.markdown(
        "The paper argues that negative powers break the forward angular lifting structure. In particular, sinc is not a forward lifted kernel, but arises naturally in the inverse picture."
    )
    formula_block(
        "Key inverse identity",
        r"\mathcal{L}^{-1}\!\left\{\frac{\theta}{b}\right\}(x)=\operatorname{sinc}(bx)."
    )

    st.info(
        "Interpretation: angular lifting requires a free angle. In composite kernels like sinc and J0, the angle is consumed internally."
    )


def section_phase_shift_theorem():
    st.header("11. Phase-Shifted Oscillations")
    formula_block(
        "Rigid angular translation",
        r"\mathcal{L}\{\sin(bx+\omega)\}(t)=\frac{\sin(\theta+\omega)}{r},\qquad \mathcal{L}\{\cos(bx+\omega)\}(t)=\frac{\cos(\theta+\omega)}{r}."
    )
    formula_block(
        "Compact complex form",
        r"\mathcal{L}\{e^{i(bx+\omega)}\}(t)=\frac{e^{i(\theta+\omega)}}{r}."
    )

    t = st.number_input("t", min_value=0.05, value=2.0, step=0.1, key="phase_t")
    b = st.number_input("b", value=1.0, step=0.1, key="phase_b")
    omega = st.slider("ω", min_value=-math.pi, max_value=math.pi, value=0.5, step=0.05, key="phase_omega")
    kernel = st.radio("Kernel", ["sin(bx+ω)", "cos(bx+ω)"], horizontal=True, key="phase_kernel")
    try:
        if kernel == "sin(bx+ω)":
            value = L_phase_shifted_sin(t, b, omega)
        else:
            value = L_phase_shifted_cos(t, b, omega)
        st.metric("Transform value", fmt(value, 10))
    except Exception as e:
        st.error(str(e))


def section_differential_structure():
    st.header("12. Differential Structure in Parameter Space")
    st.markdown(
        "The paper introduces natural differential operators adapted to the polar geometry of the (t,b)-plane."
    )
    formula_block(
        "Operators",
        r"D_{\theta}=-b\,\partial_t+t\,\partial_b,\qquad D_r=\frac{t}{r}\partial_t+\frac{b}{r}\partial_b."
    )
    formula_block(
        "Commutator relation",
        r"[D_{\theta},D_r]=D_{\theta}D_r-D_rD_{\theta}=\frac{1}{r}D_{\theta}."
    )

    st.subheader("Action on elementary kernels")
    st.latex(r"D_{\theta}\left(\frac{\sin\theta}{r}\right)=\frac{\cos\theta}{r},\qquad D_{\theta}\left(\frac{\cos\theta}{r}\right)=-\frac{\sin\theta}{r}." )
    st.latex(r"\frac{1}{n+1}D_{\theta}\,\mathcal{L}\{x^n\sin(bx)\}=\mathcal{L}\{x^n\cos(bx)\}." )
    st.latex(r"\frac{1}{n+1}D_{\theta}\,\mathcal{L}\{x^n\cos(bx)\}=-\mathcal{L}\{x^n\sin(bx)\}." )

    st.subheader("Operator calculator")
    t = st.number_input("t", min_value=0.05, value=2.0, step=0.1, key="op_t")
    b = st.number_input("b", value=1.0, step=0.1, key="op_b")
    n = st.slider("n", min_value=0, max_value=6, value=1, key="op_n")
    target = st.selectbox("Target transform", ["sin", "cos", "x^n sin", "x^n cos", "J0", "sinc"], key="op_target")

    try:
        pd = polar_data(t, b)
        if target == "sin":
            f = L_sin(t, b)
            f_t = -2 * t * b / ((t * t + b * b) ** 2)
            f_b = (t * t - b * b) / ((t * t + b * b) ** 2)
            dth = dtheta_operator(t, b, f_t, f_b)
            expected = L_cos(t, b)
        elif target == "cos":
            f = L_cos(t, b)
            f_t = (b * b - t * t) / ((t * t + b * b) ** 2)
            f_b = -2 * t * b / ((t * t + b * b) ** 2)
            dth = dtheta_operator(t, b, f_t, f_b)
            expected = -L_sin(t, b)
        elif target == "x^n sin":
            f = L_xn_sin(t, b, n)
            eps = 1e-5
            f_t = (L_xn_sin(t + eps, b, n) - L_xn_sin(t - eps, b, n)) / (2 * eps)
            f_b = (L_xn_sin(t, b + eps, n) - L_xn_sin(t, b - eps, n)) / (2 * eps)
            dth = dtheta_operator(t, b, f_t, f_b)
            expected = (n + 1) * L_xn_cos(t, b, n)
        elif target == "x^n cos":
            f = L_xn_cos(t, b, n)
            eps = 1e-5
            f_t = (L_xn_cos(t + eps, b, n) - L_xn_cos(t - eps, b, n)) / (2 * eps)
            f_b = (L_xn_cos(t, b + eps, n) - L_xn_cos(t, b - eps, n)) / (2 * eps)
            dth = dtheta_operator(t, b, f_t, f_b)
            expected = -(n + 1) * L_xn_sin(t, b, n)
        elif target == "J0":
            f = L_J0(t, b)
            f_t = -t / (pd.r ** 3)
            f_b = -b / (pd.r ** 3)
            dth = dtheta_operator(t, b, f_t, f_b)
            expected = 0.0
        else:
            f = L_sinc(t, b)
            eps = 1e-5
            f_t = (L_sinc(t + eps, b) - L_sinc(t - eps, b)) / (2 * eps)
            f_b = (L_sinc(t, b + eps) - L_sinc(t, b - eps)) / (2 * eps)
            dth = dtheta_operator(t, b, f_t, f_b)
            expected = 1.0 / (t * t + b * b) if abs(b) > 1e-12 else 0.0

        st.write({
            "F(t,b)": f,
            "D_theta F": dth,
            "expected_reference": expected,
            "difference": abs(dth - expected),
        })
    except Exception as e:
        st.error(str(e))


def section_saturation_criterion():
    st.header("13. Saturation Criterion")
    st.markdown(
        "Angular lifting exists if and only if the angular derivative is nonzero. Saturation occurs when the angular degree of freedom has been exhausted."
    )
    formula_block(
        "Criterion",
        r"\text{angular lifting exists} \iff D_{\theta}F(t,b)\neq 0."
    )
    formula_block(
        "Saturation",
        r"\text{kernel is angularly saturated} \iff D_{\theta}\mathcal{L}\{f(bx)\}=0."
    )

    st.subheader("Summary Table (Mathematical View)")

    st.markdown(r"""
| # | Kernel | Angular structure | Angular lifting? | Geometry |
|---|---|---|---|---|
| 0 | $ \sin(bx) $ | Single free angle | Yes | Circular |
| 1 | $ \cos(bx) $ | Single free angle | Yes | Circular |
| 2 | $ x^n \sin(bx) $ | Discrete lifted angle $(n+1)\theta$ | Yes | Circular |
| 3 | $ x^n \cos(bx) $ | Discrete lifted angle $(n+1)\theta$ | Yes | Circular |
| 4 | $ \mathrm{sinc}(bx) $ | Angular sector average | No | Circular saturated |
| 5 | $ J_0(bx) $ | Full circular average | No | Purely radial |
| 6 | $ J_\nu(bx) $ | Circle + analytic phase offset | No | Circular saturated |
| 7 | $ e^{bx} $ | Hyperbolic parameter $\phi$ | No | Hyperbolic |
| 8 | $ \sinh(bx) $ | Hyperbolic parameter $\phi$ | No | Hyperbolic |
| 9 | $ \cosh(bx) $ | Hyperbolic parameter $\phi$ | No | Hyperbolic |
""")

   


def section_master_calculator():
    st.header("14. Master Calculator")
    st.markdown("A single calculator covering the major formulas from the paper.")

    family = st.selectbox(
        "Family",
        [
            "sin(bx)",
            "cos(bx)",
            "x^n sin(bx)",
            "x^n cos(bx)",
            "x^s sin(bx)",
            "x^s cos(bx)",
            "e^{ibx}",
            "e^{-bx}",
            "e^{bx}",
            "x^n e^{-bx}",
            "x^n e^{bx}",
            "sinh(bx)",
            "cosh(bx)",
            "x^n sinh(bx)",
            "x^n cosh(bx)",
            "sinc(bx)",
            "J0(bx)",
            "Jν(bx)",
            "sin(bx+ω)",
            "cos(bx+ω)",
        ],
        key="master_family",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t = st.number_input("t", min_value=0.01, value=2.0, step=0.1, key="master_t")
    with col2:
        b = st.number_input("b", value=1.0, step=0.1, key="master_b")
    with col3:
        n = st.slider("n", min_value=0, max_value=8, value=2, key="master_n")
    with col4:
        s_or_nu = st.number_input("s / ν / ω", value=0.5, step=0.1, key="master_s_or_nu")

    try:
        output = None
        latex_formula = ""
        if family == "sin(bx)":
            output = L_sin(t, b)
            latex_formula = r"\frac{\sin\theta}{r}"
        elif family == "cos(bx)":
            output = L_cos(t, b)
            latex_formula = r"\frac{\cos\theta}{r}"
        elif family == "x^n sin(bx)":
            output = L_xn_sin(t, b, n)
            latex_formula = r"\frac{n!\sin((n+1)\theta)}{r^{n+1}}"
        elif family == "x^n cos(bx)":
            output = L_xn_cos(t, b, n)
            latex_formula = r"\frac{n!\cos((n+1)\theta)}{r^{n+1}}"
        elif family == "x^s sin(bx)":
            output = L_xs_sin(t, b, s_or_nu)
            latex_formula = r"\frac{\Gamma(s+1)\sin((s+1)\theta)}{r^{s+1}}"
        elif family == "x^s cos(bx)":
            output = L_xs_cos(t, b, s_or_nu)
            latex_formula = r"\frac{\Gamma(s+1)\cos((s+1)\theta)}{r^{s+1}}"
        elif family == "e^{ibx}":
            output = L_exp_ibx(t, b)
            latex_formula = r"\frac{e^{i\theta}}{r}"
        elif family == "e^{-bx}":
            output = L_exp_minus_bx(t, b)
            latex_formula = r"\frac{e^{-\phi}}{\bar r}"
        elif family == "e^{bx}":
            output = L_exp_plus_bx(t, b)
            latex_formula = r"\frac{e^{\phi}}{\bar r}"
        elif family == "x^n e^{-bx}":
            output = L_xn_exp_minus_bx(t, b, n)
            latex_formula = r"\frac{n!e^{-(n+1)\phi}}{\bar r^{n+1}}"
        elif family == "x^n e^{bx}":
            output = L_xn_exp_plus_bx(t, b, n)
            latex_formula = r"\frac{n!e^{(n+1)\phi}}{\bar r^{n+1}}"
        elif family == "sinh(bx)":
            output = L_sinh(t, b)
            latex_formula = r"\frac{\sinh\phi}{\bar r}"
        elif family == "cosh(bx)":
            output = L_cosh(t, b)
            latex_formula = r"\frac{\cosh\phi}{\bar r}"
        elif family == "x^n sinh(bx)":
            output = L_xn_sinh(t, b, n)
            latex_formula = r"\frac{n!\sinh((n+1)\phi)}{\bar r^{n+1}}"
        elif family == "x^n cosh(bx)":
            output = L_xn_cosh(t, b, n)
            latex_formula = r"\frac{n!\cosh((n+1)\phi)}{\bar r^{n+1}}"
        elif family == "sinc(bx)":
            output = L_sinc(t, b)
            latex_formula = r"\frac{\theta}{b}"
        elif family == "J0(bx)":
            output = L_J0(t, b)
            latex_formula = r"\frac{1}{r}"
        elif family == "Jν(bx)":
            output = L_Jv(t, b, s_or_nu)
            latex_formula = r"\frac{1}{r}\left(\tan\frac{\theta}{2}\right)^{\nu}"
        elif family == "sin(bx+ω)":
            output = L_phase_shifted_sin(t, b, s_or_nu)
            latex_formula = r"\frac{\sin(\theta+\omega)}{r}"
        else:
            output = L_phase_shifted_cos(t, b, s_or_nu)
            latex_formula = r"\frac{\cos(\theta+\omega)}{r}"

        st.latex(latex_formula)
        st.success(f"Transform value: {output}")

        try:
            pd = polar_data(t, b)
            info = {"r": pd.r, "theta": pd.theta}
            try:
                hd = hyperbolic_data(t, b)
                info.update({"r_bar": hd.r_bar, "phi": hd.phi})
            except Exception:
                pass
            st.write(info)
        except Exception:
            pass
    except Exception as e:
        st.error(str(e))
        
def section_summary_table():
    st.header("15. Summary Table")
    st.markdown("A table modeled on the one in the paper, with a practical classification suitable for the app.")
    
    st.subheader("Complete Kernel Classification (Mathematical Form)")

    st.markdown(r"""
| Kernel | $\mathcal{L}\{f\}(t)$ | Domain | Classification | Geometry | Angular lifting |
|---|---|---|---|---|---|
| $ \sin(bx) $ | $ \frac{\sin\theta}{r} $ | $t>0$ | bounded elementary | circular | yes |
| $ \cos(bx) $ | $ \frac{\cos\theta}{r} $ | $t>0$ | bounded elementary | circular | yes |
| $ x^n \sin(bx) $ | $ \frac{n!\sin((n+1)\theta)}{r^{n+1}} $ | $t>0$ | lifted oscillatory | circular | yes |
| $ x^n \cos(bx) $ | $ \frac{n!\cos((n+1)\theta)}{r^{n+1}} $ | $t>0$ | lifted oscillatory | circular | yes |
| $ x^s \sin(bx) $ | $ \frac{\Gamma(s+1)\sin((s+1)\theta)}{r^{s+1}} $ | $t>0$, $s>-1$ | fractional lifted | circular | yes |
| $ x^s \cos(bx) $ | $ \frac{\Gamma(s+1)\cos((s+1)\theta)}{r^{s+1}} $ | $t>0$, $s>-1$ | fractional lifted | circular | yes |
| $ e^{ibx} $ | $ \frac{e^{i\theta}}{r} $ | $t>0$ | complex oscillatory | circular | yes |
| $ e^{-bx} $ | $ \frac{e^{-\phi}}{r} $ | $t>\lvert b \rvert$ | unbounded exponential | hyperbolic | no |
| $ e^{bx} $ | $ \frac{e^{\phi}}{r} $ | $t>\lvert b \rvert$ | unbounded exponential | hyperbolic | no |
| $ \sinh(bx) $ | $ \frac{\sinh\phi}{r} $ | $t>\lvert b \rvert$ | hyperbolic elementary | hyperbolic | no |
| $ \cosh(bx) $ | $ \frac{\cosh\phi}{r} $ | $t>\lvert b \rvert$ | hyperbolic elementary | hyperbolic | no |
| $ x^n \sinh(bx) $ | $ \frac{n!\sinh((n+1)\phi)}{r^{n+1}} $ | $t>\lvert b \rvert$ | hyperbolic lifted in $\phi$ | hyperbolic | no |
| $ x^n \cosh(bx) $ | $ \frac{n!\cosh((n+1)\phi)}{r^{n+1}} $ | $t>\lvert b \rvert$ | hyperbolic lifted in $\phi$ | hyperbolic | no |
| $ \mathrm{sinc}(bx) $ | $ \frac{\theta}{b} $ | $t>0,\ b\neq 0$ | sector average | circular saturated | no |
| $ J_0(bx) $ | $ \frac{1}{r} $ | $t>0$ | full circular average | purely radial | no |
| $ J_\nu(bx) $ | $ \frac{\tan^\nu(\theta/2)}{r} $ | $t>0,\ \nu>-1$ | circular + analytic phase offset | circular saturated | no |
""")
   


def section_conclusion():
    st.header("16. Conclusion")
    st.markdown(
        """
This laboratory translates the full content of the paper into an interactive computational environment.

It contains:
- the circular / hyperbolic split,
- theorem-level formulas,
- proof-oriented verification tools,
- fractional lifting,
- phase shifting,
- differentiation operators,
- saturation tests,
- sinc and Bessel classification,
- and a full summary table.

The central message remains the same as in the paper:

> The Laplace transform admits a geometric duality.

Bounded oscillatory kernels are naturally angular, while unbounded kernels are naturally hyperbolic. Composite circular kernels such as sinc and Bessel functions occupy a saturated regime in which the angular freedom has already been consumed.
        """
    )


# ============================================================
# Sidebar navigation
# ============================================================
st.sidebar.title("Angular–Hyperbolic Laplace Lab")
st.sidebar.markdown("Interactive implementation of the full paper")

pages = {
    "Introduction": section_introduction,
    "Geometric Foundations": section_foundations,
    "Unified Theorem": section_unified_theorem,
    "Proofs and Low-Order Examples": section_proofs_and_examples,
    "Fractional Angular Lifting": section_fractional_lifting,
    "Hyperbolic Regime": section_hyperbolic_regime,
    "Angular Shifting": section_angular_shifting,
    "Horizontal Differentiation": section_horizontal_differentiation,
    "Composite Kernels": section_composite_kernels,
    "Negative Powers and Inverse sinc": section_negative_powers_and_inverse,
    "Phase-Shifted Oscillations": section_phase_shift_theorem,
    "Differential Structure": section_differential_structure,
    "Saturation Criterion": section_saturation_criterion,
    "Master Calculator": section_master_calculator,
    "Summary Table": section_summary_table,
    "Conclusion": section_conclusion,
}

page = st.sidebar.radio("Go to", list(pages.keys()))

st.title("A Unified Angular–Hyperbolic Representation of the Laplace Transform ")
st.caption("By: Rida Jamal Badawi Abu-Sokon , Amman-Jordan the version 1 preprint on https://orcid.org/0009-0008-3182-5300")

with st.expander("Read before use"):
    st.markdown(
        """
- Use $t > 0$ throughout.
- Hyperbolic formulas additionally require $|b| < t$.
- The app emphasizes the formulas and classifications stated in the paper.

        """
    )

pages[page]()
