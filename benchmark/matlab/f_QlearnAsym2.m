function [fx, dfdx, dfdp] = f_QlearnAsym2(x, P, u, in)
% Asymmetric Q-learning with TWO INDEPENDENT learning rates, matching
% CBM's RL2_model exactly (benchmark/models.py).
%
%   delta = r - Q(a)
%   Q(a) <- Q(a) + alpha_pos * delta   if delta >= 0
%   Q(a) <- Q(a) + alpha_neg * delta   otherwise
%   alpha_pos = sigmoid(P(1)),  alpha_neg = sigmoid(P(2))
%
% WHY NOT VBA's OWN f_QlearningAsym: that one uses a single sigmoid with
% a signed offset, alpha = sigmoid(P1 + sign(delta)*P2), which is a
% DIFFERENT model family — it cannot represent every (alpha_pos,
% alpha_neg) pair, and its two parameters are not the two learning
% rates. Comparing it against CBM's RL2 would compare different models
% and attribute the difference to the toolbox. This file exists so all
% three benchmark arms fit the identical likelihood.
% (VBA's own file is left untouched; this is additive.)
%
% Note the >= on delta: a zero prediction error takes the POSITIVE rate,
% matching CBM's `if delta >= 0` branch exactly.
%
% IN:
%   - x: action values (2 x 1)
%   - P: [invsigmoid(alpha_pos); invsigmoid(alpha_neg)]
%   - u: (1) previous action (1 => option 1 chosen), (2) feedback
%   - in: [unused]
% OUT:
%   - fx: updated action values
%   - dfdx, dfdp: analytic gradients (VBA's convention: dfdp is
%     n_theta x n, dfdx is n x n)

n = numel(x);

% No feedback yet (first trial): identity map.
if isnan(u(2))
    fx   = x;
    dfdx = eye(n);
    dfdp = zeros(numel(P), n);
    return;
end

a = 2 - u(1);          % u(1)=1 (option 1) -> index 1; u(1)=0 -> index 2
r = u(2);
delta = r - x(a);

if delta >= 0
    alpha = VBA_sigmoid(P(1));
    which = 1;
else
    alpha = VBA_sigmoid(P(2));
    which = 2;
end

fx    = x;
fx(a) = x(a) + alpha * delta;

% d fx / d x
dfdx = eye(n);
dfdx(a, a) = 1 - alpha;

% d fx / d P — only the ACTIVE learning rate has a gradient on this
% trial; the inactive one is exactly zero (the branch is piecewise).
dfdp = zeros(numel(P), n);
dfdp(which, a) = alpha * (1 - alpha) * delta;
