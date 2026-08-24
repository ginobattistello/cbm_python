% MATLAB VBA arm — generic driver for any benchmark grid.
%
% Reads benchmark/data/<grid>/*.mat (written by benchmark/simulate.py),
% dispatches each cell on its `family` field, inverts every subject under
% both candidate models of that family, and writes
% benchmark/results/vba_<grid>.mat.
%
%   matlab -batch "grid='boundary'; run('benchmark/run_vba_grid.m')"
%
% Families:
%   rl    — state-space (Q-values as hidden states), candidates RL / RL2
%   value — static risky choice,                     candidates LIN / POW
%
% Comparability notes are in run_vba_arm.m and run_vba_value.m; the same
% choices apply here (deterministic evolution for RL so VBA fits CBM's
% recursion; priors matched to CBM's N(0,10); VBA's out.F is a variational
% free energy, comparable in kind but not in value to a Laplace evidence).

if ~exist('grid', 'var'); grid = 'clean'; end

here     = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(here, 'external', 'VBA-toolbox')));
addpath(fullfile(here, 'matlab'));
data_dir = fullfile(here, 'data', grid);
out_dir  = fullfile(here, 'results');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

manifest = jsondecode(fileread(fullfile(data_dir, 'manifest.json')));
fprintf('VBA arm: %d cells from %s\n', numel(manifest), data_dir);

rows = struct('cell', {}, 'generator', {}, 'n_trials', {}, ...
              'beta_cond', {}, 'fitted_model', {}, 'subject', {}, ...
              'est_alpha_pos', {}, 'est_alpha_neg', {}, 'est_rho', {}, ...
              'est_beta', {}, 'F', {}, 'seconds', {}, 'converged', {});
t_start = tic;

for ci = 1:numel(manifest)
    name = manifest(ci).name;
    d    = load(fullfile(data_dir, [name '.mat']));
    nsub = double(d.n_subjects);
    T    = double(d.n_trials);
    fam  = strtrim(d.family);

    if strcmp(fam, 'rl')
        cand = {'RL', 'RL2'};
    else
        cand = {'LIN', 'POW'};
    end

    for mi = 1:2
        mn = cand{mi};
        tc = tic;
        for i = 1:nsub
            est_ap = NaN; est_an = NaN; est_rho = NaN;
            est_b = NaN; Fv = NaN; conv = 0;
            try
                if strcmp(fam, 'rl')
                    choices = double(d.choices(i, :));
                    rewards = double(d.rewards(i, :));
                    y = double(choices == 0);
                    u = zeros(2, T);
                    u(1, 2:T) = y(1:T-1);
                    u(2, 2:T) = rewards(1:T-1);
                    if strcmp(mn, 'RL'); nth = 1; ff = @f_Qlearn;
                    else;                nth = 2; ff = @f_QlearnAsym2; end
                    dim = struct('n', 2, 'n_phi', 1, 'n_theta', nth);
                    opt = struct();
                    opt.sources = struct('type', 1, 'out', 1);
                    opt.verbose = 0; opt.DisplayWin = 0;
                    opt.priors.muX0 = [0; 0];
                    opt.priors.SigmaX0 = zeros(2);
                    opt.priors.a_alpha = Inf; opt.priors.b_alpha = 0;
                    opt.priors.muTheta = zeros(nth, 1);
                    opt.priors.SigmaTheta = 10 * eye(nth);
                    opt.priors.muPhi = 0;
                    opt.priors.SigmaPhi = 10;
                    [post, out] = VBA_NLStateSpaceModel(y, u, ff, ...
                                                        @g_softmax, dim, opt);
                    est_ap = VBA_sigmoid(post.muTheta(1));
                    if strcmp(mn, 'RL')
                        est_an = est_ap;
                    else
                        est_an = VBA_sigmoid(post.muTheta(2));
                    end
                    est_b = exp(post.muPhi(1));
                else
                    y = double(d.chose(i, :));
                    u = [double(d.sure(i, :));
                         double(d.gamble(i, :));
                         double(d.prob(i, :))];
                    isLin = strcmp(mn, 'LIN');
                    if isLin; nphi = 1; else; nphi = 2; end
                    dim = struct('n', 0, 'n_theta', 0, 'n_phi', nphi);
                    opt = struct();
                    opt.sources = struct('type', 1, 'out', 1);
                    opt.verbose = 0; opt.DisplayWin = 0;
                    opt.inG = struct('linear', isLin);
                    opt.priors.muPhi = zeros(nphi, 1);
                    opt.priors.SigmaPhi = 10 * eye(nphi);
                    [post, out] = VBA_NLStateSpaceModel(y, u, [], ...
                                                        @g_valuePower, dim, opt);
                    if isLin
                        est_rho = 1; est_b = exp(post.muPhi(1));
                    else
                        est_rho = exp(post.muPhi(1));
                        est_b = exp(post.muPhi(2));
                    end
                end
                Fv = out.F(end); conv = 1;
            catch ME
                fprintf('  %s %s subj %d FAILED: %s\n', name, mn, i, ME.message);
            end

            rows(end+1) = struct('cell', name, ...
                'generator', strtrim(d.generator), 'n_trials', T, ...
                'beta_cond', double(d.beta), 'fitted_model', mn, ...
                'subject', i - 1, 'est_alpha_pos', est_ap, ...
                'est_alpha_neg', est_an, 'est_rho', est_rho, ...
                'est_beta', est_b, 'F', Fv, 'seconds', NaN, ...
                'converged', conv);  %#ok<SAGROW>
        end
        el = toc(tc);
        for k = (numel(rows) - nsub + 1):numel(rows)
            rows(k).seconds = el / nsub;
        end
        fprintf('  [%2d/%2d] %-12s %-4s %6.2fs  sum(F)=%10.2f\n', ...
                ci, numel(manifest), name, mn, el, ...
                sum([rows(end-nsub+1:end).F], 'omitnan'));
    end
end

save(fullfile(out_dir, ['vba_' grid '.mat']), 'rows', '-v7');
fprintf('\n%d rows -> vba_%s.mat  (%.1fs)\n', numel(rows), grid, toc(t_start));
