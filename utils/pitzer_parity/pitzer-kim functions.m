function [lngamma_LiCl, lngamma_NaCl, lngamma_MgCl2, osmo_w] = pitzer_mix(m_LiCl, m_NaCl, m_MgCl2)
    %% ── Physical constants ─────────────────────────────────────────────────────
N_A      = 6.0221408e23;
kB       = 1.380649e-23;
eps0     = 8.854e-12;
e_charge = 1.602e-19;
eps_r    = 78;
rho_w    = 998;      % kg/m³
Temp     = 298.15;
%% ── Ionic charges ──────────────────────────────────────────────────────────
Z_Li =  1;  Z_Na =  1;  Z_Mg =  2;  Z_Cl = -1;

%% ── Derived concentrations ─────────────────────────────────────────────────
m_Li  = m_LiCl;
m_NaCl  = m_NaCl;
m_Mg  = m_MgCl2;
m_Cl  = m_LiCl + m_NaCl + 2*m_MgCl2;
m_tot = m_Li + m_NaCl + m_Mg + m_Cl;

I_tot = m_LiCl + m_NaCl + 3*m_MgCl2;    % ionic strength: (1/2)·Σ mᵢzᵢ²
Z_sum = m_Li + m_NaCl + 2*m_Mg + m_Cl;    % Σ mᵢ|zᵢ|

%% ── Debye-Hückel parameter ─────────────────────────────────────────────────
A_phi = (1/3)*(2*pi*N_A*rho_w)^0.5 * (e_charge^2/(4*pi*eps0*eps_r*kB*Temp))^1.5;
b     = 1.2;

%% ── Pitzer virial coefficients ─────────────────────────────────────────────
beta0_LiCl  = 0.14667;  beta1_LiCl  = 0.33703;  Cphi_LiCl  = 0.00393;
beta0_NaCl  = 0.07722;  beta1_NaCl  = 0.25183;  Cphi_NaCl  = 0.00106;
beta0_MgCl2 = 0.35372;  beta1_MgCl2 = 1.70054;  Cphi_MgCl2 = 0.00524;

%% ── Mixing parameters ──────────────────────────────────────────────────────
theta_LiNa = 0.0120;   theta_LiMg = 0.2198;   theta_NaMg = 0.0970;
Psi_LiNaCl = -0.0022;  Psi_LiMgCl = -0.01930;  Psi_NaMgCl = -0.0517;

%% ── Unsymmetric mixing (J integrals) ───────────────────────────────────────
% Li-Na have equal charge: thetaE = 0 identically (J terms cancel)
J0 = @(X) 0.25*X - 1 + (1/X)*integral(@(Y) Y.^2.*(1-exp(-X./Y.*exp(-Y))), 0, Inf);
J1 = @(X) 0.25*X -     (1/X)*integral(@(Y) Y.^2.*(1-(1+X./Y.*exp(-Y)).*exp(-X./Y.*exp(-Y))), 0, Inf);

X_LiLi = 6*Z_Li^2*A_phi*sqrt(I_tot);   % = X_NaNa since z_Li = z_Na
X_MgMg = 6*Z_Mg^2*A_phi*sqrt(I_tot);
X_LiMg = 6*Z_Li*Z_Mg*A_phi*sqrt(I_tot);  % = X_NaMg since z_Li = z_Na

thetaE_LiNa  = 0;   dthetaE_LiNa = 0;
thetaE_LiMg  = (Z_Li*Z_Mg/(4*I_tot))   * (J0(X_LiMg) - 0.5*J0(X_LiLi) - 0.5*J0(X_MgMg));
thetaE_NaMg  = (Z_Na*Z_Mg/(4*I_tot))   * (J0(X_LiMg) - 0.5*J0(X_LiLi) - 0.5*J0(X_MgMg));
dthetaE_LiMg = (Z_Li*Z_Mg/(8*I_tot^2)) * (J1(X_LiMg) - 0.5*J1(X_LiLi) - 0.5*J1(X_MgMg)) - thetaE_LiMg/I_tot;
dthetaE_NaMg = (Z_Na*Z_Mg/(8*I_tot^2)) * (J1(X_LiMg) - 0.5*J1(X_LiLi) - 0.5*J1(X_MgMg)) - thetaE_NaMg/I_tot;

%% ── Phi^phi (osmotic):   Φᵠ = θ + θᴱ + I·dθᴱ/dI ──────────────────────────
%% ── Phi   (activity):    Φ  = θ + θᴱ              ──────────────────────────
Phiphi_LiNa = theta_LiNa + thetaE_LiNa + I_tot*dthetaE_LiNa;
Phiphi_LiMg = theta_LiMg + thetaE_LiMg + I_tot*dthetaE_LiMg;
Phiphi_NaMg = theta_NaMg + thetaE_NaMg + I_tot*dthetaE_NaMg;

Phi_LiNa = theta_LiNa + thetaE_LiNa;
Phi_LiMg = theta_LiMg + thetaE_LiMg;
Phi_NaMg = theta_NaMg + thetaE_NaMg;

%% ══════════════════════════════════════════════════════════════════════════
%% OSMOTIC COEFFICIENT
%% ══════════════════════════════════════════════════════════════════════════
Bphi_LiCl  = beta0_LiCl  + beta1_LiCl  * exp(-2*sqrt(I_tot));
Bphi_NaCl  = beta0_NaCl  + beta1_NaCl  * exp(-2*sqrt(I_tot));
Bphi_MgCl2 = beta0_MgCl2 + beta1_MgCl2 * exp(-2*sqrt(I_tot));

phi_DH   = -A_phi * I_tot^1.5 / (1 + b*sqrt(I_tot));
bin_phi  = m_Li*m_Cl*(Bphi_LiCl  + Z_sum*Cphi_LiCl /(2*sqrt(abs(Z_Li*Z_Cl)))) + ...
           m_NaCl*m_Cl*(Bphi_NaCl  + Z_sum*Cphi_NaCl /(2*sqrt(abs(Z_Na*Z_Cl)))) + ...
           m_Mg*m_Cl*(Bphi_MgCl2 + Z_sum*Cphi_MgCl2/(2*sqrt(abs(Z_Mg*Z_Cl))));
tert_phi = m_Li*m_NaCl*(Phiphi_LiNa + m_Cl*Psi_LiNaCl) + ...
           m_Li*m_Mg*(Phiphi_LiMg + m_Cl*Psi_LiMgCl) + ...
           m_NaCl*m_Mg*(Phiphi_NaMg + m_Cl*Psi_NaMgCl);
osmo_w   = 1 + (2/m_tot)*(phi_DH + bin_phi + tert_phi);

%% ══════════════════════════════════════════════════════════════════════════
%% ACTIVITY COEFFICIENTS
%% ══════════════════════════════════════════════════════════════════════════

%% B (activity form): g(χ) where χ = 2√I
chi    = 2*sqrt(I_tot);
g_chi  =  2*(1-(1+chi)*exp(-chi)) / chi^2;
dg_chi = -2*(1-(1+chi+0.5*chi^2)*exp(-chi)) / chi^2;   % satisfies dB/dI = β₁·dg_chi/I

B_LiCl  = beta0_LiCl  + beta1_LiCl  * g_chi;
B_NaCl  = beta0_NaCl  + beta1_NaCl  * g_chi;
B_MgCl2 = beta0_MgCl2 + beta1_MgCl2 * g_chi;
dB_LiCl  = beta1_LiCl  * dg_chi / I_tot;
dB_NaCl  = beta1_NaCl  * dg_chi / I_tot;
dB_MgCl2 = beta1_MgCl2 * dg_chi / I_tot;

%% C (activity form): C = Cᵠ / (2√|z₊z₋|)
C_LiCl  = Cphi_LiCl  / (2*sqrt(abs(Z_Li*Z_Cl)));
C_NaCl  = Cphi_NaCl  / (2*sqrt(abs(Z_Na*Z_Cl)));
C_MgCl2 = Cphi_MgCl2 / (2*sqrt(abs(Z_Mg*Z_Cl)));

%% F_mix: DH term + Σ mᵢmⱼ·dB/dI + Σ mᵢmⱼ·dΦ/dI   (shared by all salts)
F_mix = -A_phi*(sqrt(I_tot)/(1+b*sqrt(I_tot)) + (2/b)*log(1+b*sqrt(I_tot))) + ...
        m_Li*m_Cl*dB_LiCl + m_NaCl*m_Cl*dB_NaCl + m_Mg*m_Cl*dB_MgCl2 + ...
        m_Li*m_NaCl*dthetaE_LiNa + m_Li*m_Mg*dthetaE_LiMg + m_NaCl*m_Mg*dthetaE_NaMg;

%% Shorthand: B + (Z/2)·C  for each salt
BC_LiCl  = B_LiCl  + (Z_sum/2)*C_LiCl;
BC_NaCl  = B_NaCl  + (Z_sum/2)*C_NaCl;
BC_MgCl2 = B_MgCl2 + (Z_sum/2)*C_MgCl2;

%% ln γ± — LiCl  (ν₊=ν₋=1 → average of ln γ_Li and ln γ_Cl)
lngamma_LiCl = abs(Z_Li*Z_Cl)*F_mix ...
    + (m_Li + m_Cl)*BC_LiCl ...
    + m_NaCl*(Phi_LiNa + BC_NaCl) ...
    + m_Mg*(Phi_LiMg + BC_MgCl2) ...
    + 0.5*(m_Li + m_Cl)*(m_NaCl*Psi_LiNaCl + m_Mg*Psi_LiMgCl) ...
    + 0.5*m_NaCl*m_Mg*Psi_NaMgCl;

%% ln γ± — NaCl  (ν₊=ν₋=1 → average of ln γ_Na and ln γ_Cl)
lngamma_NaCl = abs(Z_Na*Z_Cl)*F_mix ...
    + (m_NaCl + m_Cl)*BC_NaCl ...
    + m_Li*(Phi_LiNa + BC_LiCl) ...
    + m_Mg*(Phi_NaMg + BC_MgCl2) ...
    + 0.5*(m_NaCl + m_Cl)*(m_Li*Psi_LiNaCl + m_Mg*Psi_NaMgCl) ...
    + 0.5*m_Li*m_Mg*Psi_LiMgCl;

%% ln γ± — MgCl2  (ν₊=1, ν₋=2 → (ln γ_Mg + 2·ln γ_Cl)/3)
lngamma_MgCl2 = abs(Z_Mg*Z_Cl)*F_mix ...
    + (2/3)*m_Cl *BC_MgCl2 ...
    + (4/3)*m_Mg *BC_MgCl2 ...
    + (4/3)*m_Li *(BC_LiCl  + 0.5*Phi_LiMg) ...
    + (4/3)*m_NaCl *(BC_NaCl  + 0.5*Phi_NaMg) ...
    + (1/3)*m_Li*(m_Cl + 2*m_Mg)*Psi_LiMgCl ...
    + (1/3)*m_NaCl*(m_Cl + 2*m_Mg)*Psi_NaMgCl ...
    + (2/3)*m_Li*m_NaCl*Psi_LiNaCl;
end