% Run the MATLAB VBA arm of the three-way robustness benchmark.
%
% Reads the SAME datasets the Python arms read
% (benchmark/data/<grid>/*.mat, written by benchmark/simulate.py) and
% inverts each subject with VBA_NLStateSpaceModel under both candidate
% models, writing benchmark/results/vba_<grid>.mat for the shared
% analysis in benchmark/analyze.py.
%
% Usage (headless):
%   matlab -batch "grid='quick'; run('benchmark/run_vba_arm.m')"
%
% ---------------------------------------------------------------------
% WHAT IS AND IS NOT COMPARABLE  (read before interpreting results)
% ---------------------------------------------------------------------
% VBA does not fit the same object as CBM, and the benchmark must not
% pretend otherwise. Three deliberate differences:
%
% 1. INFERENCE. CBM does Laplace-MAP: point estimate + Hessian. VBA does
%    full variational Bayes on a state-space model, with the Q-values as
%    hidden STATES. To make VBA as close to CBM as possible we fix the
%    evolution to be deterministic (priors.a_alpha = Inf, b_alpha = 0),
%    which removes state noise and makes VBA's Q-recursion the same
%    deterministic recursion CBM uses. Without this VBA would be fitting
%    a strictly richer model and any comparison would be meaningless.
%
% 2. EVIDENCE SCALE. VBA's out.F is a variational free energy (a lower
%    bound on log p(y|m)); CBM's log_evidence is a Laplace approximation
%    to the same quantity. They are comparable in KIND (both approximate
%    log model evidence, both used the same way for model comparison)
%    but not identical in value. Model SELECTION (which model wins) is
%    the fair comparison; absolute nats are not.
%
% 3. RL2 PARAMETERIZATION. VBA's f_QlearningAsym uses a SINGLE sigmoid
%    with a signed offset:   alpha = sigmoid(P1 + sign(delta)*P2)
%    whereas CBM's RL2 uses TWO independent sigmoids:
%        alpha_pos = sigmoid(t1),  alpha_neg = sigmoid(t2)
%    These span different families (VBA's cannot represent every
%    (alpha_pos, alpha_neg) pair). Rather than silently compare
%    different models, this script ships its own f_QlearnAsym2.m
%    implementing CBM's two-sigmoid form exactly, so all three arms fit
%    the identical model. VBA's own version is left untouched.
%
% Also note VBA's g_softmax models P(a=1) with a sigmoid on beta*(Q1-Q2),
% which is algebraically the same 2-option softmax CBM uses.

if ~exist('grid', 'var'); grid = 'quick'; end

here     = fileparts(mfilename('fullpath'));
repo     = fileparts(here);
vba_path = fullfile(here, 'external', 'VBA-toolbox');
data_dir = fullfile(here, 'data', grid);
out_dir  = fullfile(here, 'results');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

addpath(genpath(vba_path));
addpath(fullfile(here, 'matlab'));   % our f_QlearnAsym2 / g_softmax2

manifest_txt = fileread(fullfile(data_dir, 'manifest.json'));
manifest     = jsondecode(manifest_txt);
fprintf('VBA arm: %d cells from %s\n', numel(manifest), data_dir);

rows = struct('cell', {}, 'generator', {}, 'n_trials', {}, ...
              'beta_cond', {}, 'fitted_model', {}, 'subject', {}, ...
              'est_alpha_pos', {}, 'est_alpha_neg', {}, 'est_beta', {}, ...
              'F', {}, 'seconds', {}, 'converged', {});

t_start = tic;
for ci = 1:numel(manifest)
    name = manifest(ci).name;
    d    = load(fullfile(data_dir, [name '.mat']));
    nsub = double(d.n_subjects);
    T    = double(d.n_trials);

    for mi = 1:2
        if mi == 1; model_name = 'RL'; else; model_name = 'RL2'; end
        t_cell = tic;
        for i = 1:nsub
            choices = double(d.choices(i, :));   % 0/1
            rewards = double(d.rewards(i, :));

            % VBA convention: y(t) = 1 if option 1 (our choice==0) was
            % taken, and u(:,t) carries the PREVIOUS action/feedback.
            y = double(choices == 0);
            u = zeros(2, T);
            u(1, 2:T) = y(1:T-1);            % previous action (1 => opt 1)
            u(2, 2:T) = rewards(1:T-1);      % previous feedback

            dim = struct('n', 2, 'n_phi', 1, 'n_theta', 1);
            if strcmp(model_name, 'RL2'); dim.n_theta = 2; end

            opt = struct();
            opt.sources          = struct('type', 1, 'out', 1);  % binomial
            opt.verbose          = 0;
            opt.DisplayWin       = 0;
            opt.priors.muX0      = [0; 0];
            opt.priors.SigmaX0   = zeros(2);
            % Deterministic evolution => same recursion as CBM (see note 1)
            opt.priors.a_alpha   = Inf;
            opt.priors.b_alpha   = 0;
            % Match CBM's prior: N(0, 10) on every fitted parameter
            opt.priors.muTheta    = zeros(dim.n_theta, 1);
            opt.priors.SigmaTheta = 10 * eye(dim.n_theta);
            opt.priors.muPhi      = zeros(dim.n_phi, 1);
            opt.priors.SigmaPhi   = 10 * eye(dim.n_phi);

            if strcmp(model_name, 'RL')
                f_fname = @f_Qlearn;
            else
                f_fname = @f_QlearnAsym2;   % CBM's two-sigmoid form
            end

            try
                [post, out] = VBA_NLStateSpaceModel(y, u, f_fname, ...
                                                    @g_softmax, dim, opt);
                if strcmp(model_name, 'RL')
                    a_pos = VBA_sigmoid(post.muTheta(1));
                    a_neg = a_pos;
                else
                    a_pos = VBA_sigmoid(post.muTheta(1));
                    a_neg = VBA_sigmoid(post.muTheta(2));
                end
                bet  = exp(post.muPhi(1));
                Fval = out.F(end);
                conv = 1;
            catch ME
                fprintf('  %s %s subj %d FAILED: %s\n', name, model_name, i, ME.message);
                a_pos = NaN; a_neg = NaN; bet = NaN; Fval = NaN; conv = 0;
            end

            rows(end+1) = struct( ...
                'cell', name, 'generator', d.generator, ...
                'n_trials', T, 'beta_cond', double(d.beta), ...
                'fitted_model', model_name, 'subject', i - 1, ...
                'est_alpha_pos', a_pos, 'est_alpha_neg', a_neg, ...
                'est_beta', bet, 'F', Fval, 'seconds', NaN, ...
                'converged', conv);  %#ok<SAGROW>
        end
        el = toc(t_cell);
        for k = (numel(rows) - nsub + 1):numel(rows)
            rows(k).seconds = el / nsub;
        end
        fprintf('  [%d/%d] %-24s %-4s %6.2fs  sum(F)=%10.2f\n', ...
                ci, numel(manifest), name, model_name, el, ...
                sum([rows(end-nsub+1:end).F], 'omitnan'));
    end
end

outfile = fullfile(out_dir, ['vba_' grid '.mat']);
save(outfile, 'rows', '-v7');
fprintf('\n%d rows -> %s  (%.1fs total)\n', numel(rows), outfile, toc(t_start));
