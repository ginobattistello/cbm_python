function [gx, dgdx, dgdP] = g_valuePower(x, P, u, in)
% Risky-choice observation function with a POWER (CRRA) value function,
% matching CBM's POW_model exactly (benchmark/models.py).
%
%   U(sure)   = s^rho
%   U(gamble) = p * g^rho
%   P(choose gamble) = sigmoid( beta * (U(gamble) - U(sure)) )
%   rho  = exp(P(1)),  beta = exp(P(2))
%
% Set in.linear = true for the LINEAR model (rho fixed at 1, so P(1) is
% log beta and there is no curvature parameter) — that is CBM's LIN_model.
%
% This is a STATIC (observation-only) model: there is no hidden state to
% evolve, so VBA is called with dim.n = 0 and f_fname = []. The whole
% model lives here.
%
% IN:
%   - x  : (unused; no hidden states)
%   - P  : [log rho; log beta]  (or [log beta] when in.linear)
%   - u  : (1) sure amount, (2) gamble amount, (3) gamble probability
%   - in : struct, optional field .linear
% OUT:
%   - gx   : P(choose gamble)
%   - dgdx : empty (no states)
%   - dgdP : dgx/dP, size n_phi x 1

s = u(1); g = u(2); p = u(3);

isLinear = isstruct(in) && isfield(in, 'linear') && in.linear;

if isLinear
    rho  = 1;
    beta = exp(P(1));
else
    rho  = exp(P(1));
    beta = exp(P(2));
end

% Utilities and their difference
Ug = p * g^rho;
Us = s^rho;
dU = Ug - Us;

z  = beta * dU;
gx = 1 / (1 + exp(-z));

dgdx = [];

% d gx / d z
dz = gx * (1 - gx);

if isLinear
    % only beta; d z / d log beta = beta * dU
    dgdP = dz * beta * dU;
else
    % d dU / d rho  (note d/drho x^rho = x^rho * log x; x > 0 guaranteed
    % by the generator, which draws strictly positive amounts)
    ddU_drho = p * g^rho * log(g) - s^rho * log(s);
    % chain through rho = exp(P1): d rho / d P1 = rho
    dgdP = [dz * beta * ddU_drho * rho;
            dz * beta * dU];
end
